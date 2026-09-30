"""
Smoke-тесты онлайн-системы (ШАГ 6): ядро стриминга, watchdog, finetune-экспорт.

Запуск: py -3.13 tests/test_online.py
(не требуют реальной модели: PredictionEngine работает с fake-predictor'ом)
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from config.online import OnlineConfig
from online.buffer import RingBuffer
from online.engine import PredictionEngine, EngineStalledError
from online.finetune import extract_segments


# ---------------------------------------------------------------------------
# Fake-предиктор: управляемая скорость predict, фиксированный выход
# ---------------------------------------------------------------------------

class FakeConfig:
    def __init__(self, seq_len=5, horizon=3, features=('a', 'b'), targets=('ta',)):
        self.sequence_length = seq_len
        self.prediction_horizon = horizon
        self.feature_columns = list(features)
        self.target_columns = list(targets)


class FakeScaler:
    def __init__(self):
        self.mean_ = np.zeros(2)
        self.scale_ = np.ones(2)


class FakePredictor:
    """manifest=None → ErrorMonitor отключён (ветка fallback тоже проверяется)."""

    def __init__(self, predict_delay=0.0, seq_len=5, horizon=3):
        self.config = FakeConfig(seq_len, horizon)
        self.feature_scaler = FakeScaler()
        self.manifest = None
        self._delay = predict_delay

    def predict(self, x):
        if self._delay:
            time.sleep(self._delay)
        return np.zeros((self.config.prediction_horizon,
                         self.config.output_dim if hasattr(self.config, 'output_dim')
                         else len(self.config.target_columns)))


def make_rows(n, val=1.0):
    return [{'a': val, 'b': val + 1, 'ta': val * 2} for _ in range(n)]


class UncertainPredictor(FakePredictor):
    """N4/C4: fake с predict_uncertain — фиксированная лента; скорость MC управляемая."""

    def __init__(self, mc_delay=0.0, std_val=0.5, **kw):
        super().__init__(**kw)
        self._mc_delay = mc_delay
        self._std_val = std_val
        self.mc_calls = 0

    def predict_uncertain(self, x, mc_samples=30):
        if self._mc_delay:
            time.sleep(self._mc_delay)
        self.mc_calls += 1
        h, n_t = self.config.prediction_horizon, len(self.config.target_columns)
        return {'mean': np.zeros((h, n_t)), 'std': np.full((h, n_t), self._std_val)}


def test_ring_buffer():
    buf = RingBuffer(maxlen=10)
    for i in range(25):
        buf.add({'a': i})
    assert len(buf) == 10, 'maxlen ограничивает рост (план §7)'
    assert buf.total_added == 25, 'счётчик всех добавленных строк'
    assert buf.window_df(5)['a'].tolist() == [20, 21, 22, 23, 24]
    assert buf.is_ready(10) and not buf.is_ready(11)
    print('  ✓ RingBuffer: maxlen, total_added, окно')


def test_engine_warmup_and_predict():
    cfg = OnlineConfig(inference_timeout_s=2.0)
    eng = PredictionEngine(FakePredictor(), {'cyclic': False,
                                             'relative_wave_angle': False}, cfg)
    res = [eng.on_tick(i, row) for i, row in enumerate(make_rows(4))]
    assert all(r.warmup for r in res), 'до заполнения окна — прогрев'
    res5 = eng.on_tick(4, make_rows(1)[0])
    assert res5.predicted and res5.preds is not None
    assert not eng.error_monitor_enabled, 'manifest без per-target MAE → монитор off'
    print('  ✓ Engine: прогрев, первый прогноз, монитор отключён без manifest')


def test_engine_uncertain_cadence():
    """N4/C4: MC-прогноз с cadence 1/10; лента персистентна между MC-тиками;
    статус «не верить прогнозу» по эмпирическим порогам."""
    cfg = OnlineConfig(uncertain_cadence=10, uncertain_mc_samples=5,
                       uncertain_max_std={'ta': 0.4}, inference_timeout_s=2.0)
    eng = PredictionEngine(UncertainPredictor(std_val=0.5),
                           {'cyclic': False, 'relative_wave_angle': False}, cfg)
    res = [eng.on_tick(i, row) for i, row in enumerate(make_rows(6))]
    # первый прогноз на tick 4 — не MC-тик, ленты ещё нет
    assert res[4].predicted and res[4].uncertain_std is None, 'до первого MC-тика ленты нет'
    r10 = eng.on_tick(10, make_rows(1)[0])
    assert r10.predicted and eng.predictor.mc_calls == 1, 'MC-тик: вызов predict_uncertain'
    assert r10.uncertain_std is not None and r10.uncertain_std.shape == (3, 1)
    assert r10.uncertain_mean is not None and r10.uncertain_mean.shape == (3, 1)
    assert r10.uncertain_alert, 'std 0.5 > порога 0.4 → «не верить прогнозу»'
    r11 = eng.on_tick(11, make_rows(1)[0])
    assert eng.predictor.mc_calls == 1, 'между MC-тиками вызовов нет (cadence)'
    assert r11.uncertain_std is not None and r11.uncertain_alert, \
        'лента и статус персистентны между MC-тиками'
    # низкий std → alert снимается только на следующем MC-тике
    eng2 = PredictionEngine(UncertainPredictor(std_val=0.3),
                            {'cyclic': False, 'relative_wave_angle': False}, cfg)
    for i, row in enumerate(make_rows(11)):
        r = eng2.on_tick(i, row)
    assert r.uncertain_std is not None and not r.uncertain_alert, \
        'std 0.3 <= порога 0.4 → прогнозу верить'
    print('  ✓ Engine: MC-Dropout cadence, персистентность ленты, статус «не верить»')


def test_engine_uncertain_stale_no_stall():
    """N4/C4: stale MC-вызова НЕ рвёт tick и НЕ ставит паузу основного прогноза
    (вспомогательный вызов, план §7)."""
    cfg = OnlineConfig(uncertain_cadence=1, uncertain_mc_samples=5,
                       inference_timeout_s=0.2, stale_pause_threshold=3)
    eng = PredictionEngine(UncertainPredictor(mc_delay=0.5, std_val=0.5),
                           {'cyclic': False, 'relative_wave_angle': False}, cfg)
    for i, row in enumerate(make_rows(6)):
        eng.on_tick(i, row)
    # MC на каждом tick'е выходит за таймаут — основной прогноз продолжается,
    # EngineStalledError НЕ возникает (stale-счётчик MC отдельный)
    raised = False
    try:
        for i, row in enumerate(make_rows(6), start=6):
            eng.on_tick(i, row)
    except EngineStalledError:
        raised = True
    assert not raised, 'stale MC не должен останавливать источник'
    assert eng._consecutive_stale == 0, 'stale-счётчик основного прогноза не тронут'
    print('  ✓ Engine: stale MC — tick продолжается, пауза не ставится')
def test_engine_watchdog():
    cfg = OnlineConfig(inference_timeout_s=0.2, stale_pause_threshold=3)
    eng = PredictionEngine(FakePredictor(predict_delay=0.5),
                           {'cyclic': False, 'relative_wave_angle': False}, cfg)
    for i, row in enumerate(make_rows(5)):
        eng.on_tick(i, row)
    # 3 подряд «висящих» прогноза → EngineStalledError (план §7)
    raised = False
    try:
        for i, row in enumerate(make_rows(5), start=5):
            eng.on_tick(i, row)
    except EngineStalledError:
        raised = True
    assert raised, 'stale_pause_threshold должен остановить источник'
    print('  ✓ Engine: watchdog — 3 подряд таймаутов → EngineStalledError')


def test_engine_error_gate_nan():
    cfg = OnlineConfig()
    eng = PredictionEngine(FakePredictor(),
                           {'cyclic': False, 'relative_wave_angle': False}, cfg)
    for i, row in enumerate(make_rows(6)):
        eng.on_tick(i, row)
    bad = eng.on_tick(6, {'a': float('nan'), 'b': 1.0, 'ta': 1.0})
    assert bad.skipped_reason == 'nan_window', 'NaN-окно → пропуск с причиной'
    # observe_actual копит реализованную ошибку (факт 2.0 vs прогноз 0.0)
    eng.observe_actual(5, make_rows(1)[0])
    st = eng.error_stats()
    assert st['count'] > 0 and st['per_target']['ta'] > 0
    print('  ✓ Engine: NaN-гейт + реализованная ошибка факт-vs-прогноз')


def test_finetune_extract_segments(tmp=Path('results/online/_test_ft')):
    tmp.mkdir(parents=True, exist_ok=True)
    n = 300
    df = pd.DataFrame({
        'tick': range(n), 'predicted': [1] * n,
        'input_anomaly': [0] * n, 'input_anomaly_features': [''] * n,
        'error_anomaly': [1 if 100 <= i < 110 or 200 <= i < 205 else 0 for i in range(n)],
        'error_ratio': [1.0] * n,
        'Roll(градусы)': np.arange(n) * 0.1,
        'Rudder State(градусы)': [0.0] * n,
    })
    csv = tmp / 'session.csv'
    df.to_csv(csv, index=False)
    out = extract_segments(csv, seq_len=10, all_rows=False)
    # сегмент 100-109 + 10 контекста → с 90; сегмент 200-204 → с 190
    assert len(out) == (110 - 90) + (205 - 190), f'ожидались расширенные сегменты, got {len(out)}'
    assert out['Roll(градусы)'].min() >= 90 * 0.1 - 1e-9
    # пустые маркеры → внятный отказ
    df2 = df.copy(); df2['error_anomaly'] = 0
    df2.to_csv(csv, index=False)
    try:
        extract_segments(csv, 10, False)
        raise AssertionError('должен был отказаться')
    except SystemExit as e:
        assert 'error_anomaly' in str(e)
    print('  ✓ finetune.extract_segments: расширение контекстом, отказ без маркеров')


if __name__ == '__main__':
    test_ring_buffer()
    test_engine_warmup_and_predict()
    test_engine_watchdog()
    test_engine_error_gate_nan()
    test_engine_uncertain_cadence()
    test_engine_uncertain_stale_no_stall()
    test_finetune_extract_segments()
    print('\nOK — все smoke-тесты онлайн-системы пройдены')
