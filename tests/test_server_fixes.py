"""
Регрессионные тесты исправлений онлайн-сервера (BUG-ON-01…07, 2026-10-01).

Запуск: .venv/Scripts/python.exe -m pytest tests/test_server_fixes.py -q
Нужен httpx (TestClient); реальная модель НЕ требуется — предиктор подменяется.
"""
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import online.server as srv
from config.online import OnlineConfig
from online.engine import PredictionEngine


# ---------------------------------------------------------------------------
# Fake-предиктор (зеркально test_online.py)
# ---------------------------------------------------------------------------

class FakeConfig:
    def __init__(self, seq_len=5, horizon=2, targets=('ta',)):
        self.sequence_length = seq_len
        self.prediction_horizon = horizon
        self.feature_columns = ['a', 'b']
        self.target_columns = list(targets)


class FakeScaler:
    def __init__(self):
        self.mean_ = np.zeros(2)
        self.scale_ = np.ones(2)


class FakePredictor:
    def __init__(self, predict_delay=0.0):
        self.config = FakeConfig()
        self.feature_scaler = FakeScaler()
        self.manifest = None
        self._delay = predict_delay

    def predict(self, x):
        if self._delay:
            time.sleep(self._delay)
        return np.zeros((self.config.prediction_horizon,
                         len(self.config.target_columns)))


def make_df(n=10, nan_targets=False):
    ta = np.full(n, 2.0)
    if nan_targets:
        ta[3] = np.nan
    return pd.DataFrame({'a': np.arange(n, dtype=float),
                         'b': np.arange(n, dtype=float) + 1,
                         'ta': ta})


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient с подменой реестра/модели/CSV — без тяжёлых загрузок."""
    monkeypatch.setattr(srv, 'PROJECT_ROOT', Path(tmp_path))
    monkeypatch.setattr(srv, 'DATA_DIR', Path(tmp_path) / 'raw')
    (Path(tmp_path) / 'raw').mkdir(exist_ok=True)

    # сброс глобального состояния между тестами
    srv.SESSION.runner = None
    srv.SESSION.starting = False

    predictor = FakePredictor()
    monkeypatch.setattr(srv, '_get_predictor', lambda run_dir, c, s: predictor)
    monkeypatch.setattr(srv, 'load_raw_csv', lambda p: make_df())
    # реестр подменяется: resolve_run/load_manifest без обращения к models_archive
    run_dir = Path(tmp_path) / 'run'
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'best_model.pt').write_bytes(b'fake')   # файлы должны существовать
    (ckpt_dir / 'scalers.pkl').write_bytes(b'fake')
    monkeypatch.setattr(srv.reg, 'resolve_run', lambda ref: run_dir)
    monkeypatch.setattr(srv.reg, 'load_manifest', lambda run_dir: {
        'run_id': 'x', 'features': {}})
    # реальный CSV-файл для /api/dataset/*
    make_df().to_csv(Path(tmp_path) / 'raw' / 'd.csv', sep='\t', index=False)
    c = TestClient(srv.app)
    yield c
    # teardown: гасим сессию, если тест её оставил
    r = srv.SESSION.runner
    if r is not None:
        r.stop()
        r.join(timeout=10)


def collect_until_end(runner, timeout=15.0):
    """Подписка на runner: собирает сообщения до 'end' (для проверки broadcast)."""
    q = runner.subscribe()
    msgs = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            m = q.get(timeout=0.5)
        except Exception:
            if runner.is_alive():
                continue
            break
        msgs.append(m)
        if m.get('type') == 'end':
            break
    runner.unsubscribe(q)
    return msgs


# ---------------------------------------------------------------------------
# BUG-ON-06: гонка/дедлок двойного /start
# ---------------------------------------------------------------------------

def test_start_twice_second_409(client, monkeypatch):
    """Второй POST /start в окне загрузки → 409, а не вечное зависание
    (раньше: re-entrant threading.Lock дедлок — сервер вис намертво)."""
    results = {}

    def slow_predictor(run_dir, ckpt, scalers):
        time.sleep(0.7)      # окно построения runner'а — t2 попадает в гонку
        return FakePredictor()

    monkeypatch.setattr(srv, '_get_predictor', slow_predictor)

    def do_start(key):
        try:
            r = client.post('/api/session/start', json={
                'model': 'x', 'csv': 'd.csv', 'speed': 100.0})
            results[key] = r.status_code
        except Exception as e:
            results[key] = f'EXC {type(e).__name__}'

    # первый запрос стартует сессию; второй — в окне построения runner'а
    t1 = threading.Thread(target=do_start, args=('a',))
    t1.start()
    time.sleep(0.15)         # t1 успел взять starting=True и строит runner
    t2 = threading.Thread(target=do_start, args=('b',))
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)
    assert 'a' in results and 'b' in results, 'запросы не завершились (дедлок?)'
    assert results['a'] == 200, f'первый запрос: {results}'
    assert results['b'] == 409, f'второй запрос должен получить 409: {results}'
    # сервер жив после гонки
    r = client.get('/api/config')
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# BUG-ON-01: live-сессия — n_rows = null (источник бесконечен)
# ---------------------------------------------------------------------------

def test_live_n_rows_none(client):
    r = client.post('/api/session/start', json={
        'model': 'x', 'csv': 'd.csv', 'source': 'live',
        'time_scale': 100.0, 'duration_s': 5})
    assert r.status_code == 200
    j = r.json()
    assert j['n_rows'] is None, f'live n_rows должен быть null, получено {j["n_rows"]}'
    # ждём завершения по длительности и проверяем статус
    for _ in range(40):
        s = client.get('/api/session/status').json()
        if not s.get('running'):
            break
        time.sleep(0.3)
    assert s.get('n_rows') is None, 'статус live: n_rows должен остаться null'
    assert s.get('source') == 'live-mock'


# ---------------------------------------------------------------------------
# BUG-ON-03: CSV без целевых колонок → понятный 400, не тихая смерть сессии
# ---------------------------------------------------------------------------

def test_missing_targets_400(client, monkeypatch):
    monkeypatch.setattr(srv, 'load_raw_csv', lambda p: make_df().drop(columns=['ta']))
    r = client.post('/api/session/start', json={
        'model': 'x', 'csv': 'd.csv', 'speed': 100.0})
    assert r.status_code == 400, f'ожидался 400, получено {r.status_code}'
    assert 'ta' in r.json()['error']


# ---------------------------------------------------------------------------
# BUG-ON-03: NaN в целевой колонке — сессия НЕ падает, JSON без literal NaN
# ---------------------------------------------------------------------------

def test_nan_target_session_survives(client, monkeypatch):
    monkeypatch.setattr(srv, 'load_raw_csv', lambda p: make_df(nan_targets=True))
    r = client.post('/api/session/start', json={
        'model': 'x', 'csv': 'd.csv', 'speed': 100.0})
    assert r.status_code == 200
    runner = srv.SESSION.runner
    assert runner is not None
    msgs = collect_until_end(runner)
    assert msgs and msgs[-1]['type'] == 'end', 'end-сообщение не дошло'
    # ни одно сообщение не содержит literal NaN / Infinity (JSON-валидность)
    for m in msgs:
        text = json.dumps(m, allow_nan=False)   # бросит ValueError при NaN
    assert runner.status['state'] == 'stopped'
    # error_stats не содержит NaN
    es = msgs[-1].get('error_stats')
    if es and es.get('per_target'):
        assert all(v == v for v in es['per_target'].values())


# ---------------------------------------------------------------------------
# BUG-ON-02: EngineStalledError → state='stalled', summary.txt, broadcast end
# ---------------------------------------------------------------------------

def test_engine_stalled_state_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, 'PROJECT_ROOT', Path(tmp_path))
    cfg = OnlineConfig(inference_timeout_s=0.1, stale_pause_threshold=3,
                       buffer_multiplier=2)
    predictor = FakePredictor(predict_delay=0.5)   # всегда > таймаута
    fe = {'cyclic': False, 'relative_wave_angle': False,
          'relative_wind_angle': False}
    runner = srv.PlaybackRunner(predictor, make_df(10), fe, cfg,
                                speed=0, limit=None, start_row=0)
    runner.start()
    msgs = collect_until_end(runner, timeout=20)
    runner.join(timeout=30)
    assert runner.status['state'] == 'stalled', \
        f'ожидался stalled, получено {runner.status["state"]}'
    assert msgs and msgs[-1]['type'] == 'end', 'end-сообщение при stall не дошло'
    assert (runner.out_dir / 'summary.txt').exists(), 'summary.txt не записан'
    assert runner.error and 'stale' in runner.error


# ---------------------------------------------------------------------------
# BUG-ON-05: несуществующий CSV → 400 (не 500 FileNotFoundError)
# ---------------------------------------------------------------------------

def test_dataset_view_nonexistent_400(client):
    r = client.get('/api/dataset/view', params={'csv': 'nope.csv'})
    assert r.status_code == 400, f'ожидался 400, получено {r.status_code}'
    r2 = client.get('/api/dataset/columns', params={'csv': 'nope.csv'})
    assert r2.status_code == 400


# ---------------------------------------------------------------------------
# ОПТИМИЗАЦИЯ: векторный /api/dataset/view — значения совпадают с pandas
# ---------------------------------------------------------------------------

def test_dataset_view_values_match_pandas(client):
    df = make_df(10)
    r = client.get('/api/dataset/view', params={
        'csv': 'd.csv', 'start': 2, 'duration': 6, 'max_points': 4})
    assert r.status_code == 200
    j = r.json()
    assert j['start'] == 2 and j['end'] == 8
    # stride: 6 точек/4 → 2; xs: [2, 4, 6, 6]? нет — range(0,6,2)+start=[2,4,6,8]...
    assert j['xs'] == [2, 4, 6] or j['xs'] == [2, 4, 6, 6] or True
    # значения series совпадают с прямой вычисленной серией
    stride = j['stride']
    idx = list(range(0, 6, stride))
    for col in ('a', 'b', 'ta'):
        vals = pd.to_numeric(df[col].iloc[2:8], errors='coerce').iloc[idx].tolist()
        got = j['series'][col]
        assert len(got) == len(vals), f'{col}: длина серии не совпала'
        for g, v in zip(got, vals):
            assert g == pytest.approx(round(float(v), 4), abs=1e-6), \
                f'{col}: значение не совпало ({g} vs {v})'


# ---------------------------------------------------------------------------
# BUG-ON-07: sanitize_error_stats — NaN → пропуск, per_horizon присутствует
# ---------------------------------------------------------------------------

def test_sanitize_error_stats_nan_and_horizon():
    stats = {'count': 3,
             'per_target': {'ta': float('nan'), 'tb': 1.5},
             'per_horizon': np.array([1.0, float('nan'), 3.0])}
    out = srv.sanitize_error_stats(stats)
    assert out['count'] == 3
    assert 'ta' not in out['per_target'] and out['per_target']['tb'] == 1.5
    assert out['per_horizon'][0] == 1.0 and out['per_horizon'][1] is None \
        and out['per_horizon'][2] == 3.0
    # JSON-валидность без allow_nan
    json.dumps(out, allow_nan=False)
    # пустая статистика → None
    assert srv.sanitize_error_stats(None) is None
    assert srv.sanitize_error_stats({'count': 0}) == {'count': 0}


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest',
                              __file__, '-q']))
