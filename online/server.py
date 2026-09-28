"""
Локальный веб-сервер режима A «онлайн-playback» (ШАГ 3 плана).

Запуск (только локально, bind 127.0.0.1):
    .venv/Scripts/python.exe -m online.server [--port 8765]

REST API:
    GET  /api/models                 — список моделей реестра (с manifest)
    GET  /api/datasets               — CSV в data/raw/
    GET  /api/session/status         — состояние сессии
    POST /api/session/start          — {model, csv, speed, limit, start_row}
    POST /api/session/stop           — стоп
    POST /api/session/pause          — {paused: bool}
    POST /api/session/speed          — {speed: float | "max"}

WebSocket /ws: push tick-сообщений (backpressure: очередь ≤ 2, план §7).
Запись сессии — на сервере: закрытие браузера НЕ останавливает запись.
"""
import argparse
import asyncio
import base64
import csv as csv_mod
import json
import queue as py_queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config.online import OnlineConfig
import registry as reg
from online.engine import PredictionEngine, EngineStalledError
from online.sources import load_raw_csv, extract_fe_params, MockLiveSource
from run_inference import VesselPredictor_Inference

app = FastAPI(title="Vessel Motion Online — режим A")

from fastapi.staticfiles import StaticFiles
CFG = OnlineConfig()
STATIC = Path(__file__).parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC), name='static')  # uPlot vendor
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


# ===========================================================================
# Менеджер сессии (поток playback + рассылка WS)
# ===========================================================================

class SessionState:
    """Общая структура: runner-поток + подписчики WS + статус."""

    def __init__(self):
        self.runner: 'PlaybackRunner' = None
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self.lock:
            return self.runner is not None and self.runner.is_alive()


SESSION = SessionState()


class PlaybackRunner(threading.Thread):
    """
    Поток tick-цикла (зеркально online/playback.py, но с управлением):
    старт/стоп/пауза/скорость; запись сессии в results/online/<id>_playback/;
    broadcast в очереди подписчиков (Queue(maxlen<=2), drop-old).
    """

    def __init__(self, predictor, df_raw, fe_params, cfg: OnlineConfig,
                 speed: float, limit, start_row: int,
                 source_mode: str = 'playback', time_scale: float = 1.0,
                 seed: int = 42, duration_s: float | None = None,
                 anomaly_every: int | None = None,
                 motion_scale: float = 2.0):
        """
        source_mode: 'playback' — CSV по порядку (режим A); 'live' — MockLiveSource
                     (режим B, ШАГ 4).
        time_scale — ускорение live-режима относительно 1 Гц (1.0 = реальное время;
                     >1 — только для разработки/приёмки, в рейсе всегда 1.0).
        duration_s — лимит длительности сессии в «секундах данных» (план §7:
                     авто-stop; None — только max_session_hours).
        """
        super().__init__(daemon=True, name='playback')
        self.predictor = predictor
        self.df_raw = df_raw
        self.fe_params = fe_params
        self.cfg = cfg
        self.speed = float(speed)
        self.limit = limit
        self.start_row = start_row
        self.source_mode = source_mode
        self.time_scale = float(time_scale)
        self.seed = int(seed)
        self.duration_s = duration_s
        self.anomaly_every = anomaly_every
        self.motion_scale = motion_scale

        self._stop_flag = threading.Event()
        self._paused = threading.Event()  # set = пауза

        self.engine = PredictionEngine(predictor, fe_params, cfg)
        self.subscribers: list[py_queue.Queue] = []
        self.sub_lock = threading.Lock()

        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + (
            '_live' if source_mode == 'live' else '_playback')
        self.out_dir = PROJECT_ROOT / 'results' / 'online' / self.session_id
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # n_rows известен сразу (ответ /start до старта потока)
        df = df_raw.iloc[start_row:]
        if limit is not None:
            df = df.iloc[:limit]
        self.n_rows = len(df)

        self.status = {
            'session_id': self.session_id,
            'state': 'running',           # running | paused | stopped | stalled
            'tick': 0,
            'n_rows': 0,
            'predicted': 0,
            'input_anomaly_ticks': 0,
            'error_anomaly_ticks': 0,
        }
        self.error = None
        self.stats_snapshot = None       # engine.error_stats() — для /status

    # ------------------------------------------------------------------
    # Управление (REST)
    # ------------------------------------------------------------------

    def set_speed(self, speed: float):
        self.speed = float(speed)

    def set_paused(self, paused: bool):
        if paused:
            self._paused.set()
            self.status['state'] = 'paused'
        else:
            self._paused.clear()
            self.status['state'] = 'running'

    def stop(self):
        self._stop_flag.set()
        self._paused.clear()  # чтобы wait не висел на паузе

    # ------------------------------------------------------------------
    # Подписки WS (backpressure: drop-old, план §7)
    # ------------------------------------------------------------------

    def subscribe(self) -> py_queue.Queue:
        q = py_queue.Queue(maxsize=2)
        with self.sub_lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: py_queue.Queue):
        with self.sub_lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def _broadcast(self, msg: dict):
        with self.sub_lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(msg)
            except py_queue.Full:
                try:
                    q.get_nowait()   # выбрасываем старое
                    q.put_nowait(msg)
                except (py_queue.Empty, py_queue.Full):
                    pass

    # ------------------------------------------------------------------
    # Tick-цикл
    # ------------------------------------------------------------------

    def run(self):
        try:
            self._run_loop()
        except Exception as e:  # системная ошибка цикла — в статус
            self.error = str(e)
            self.status['state'] = 'stopped'
            raise

    def _run_loop(self):
        cfg = self.cfg
        engine = self.engine
        target_columns = engine.target_columns

        raw_columns = list(self.df_raw.columns)

        if self.source_mode == 'live':
            # Режим B: сидированный mock-сенсор (приёмка/разработка); строк —
            # бесконечно, стоп только по длительности/стопу (план §7)
            mock = MockLiveSource(self.df_raw, seed=self.seed,
                                   anomaly_every=self.anomaly_every,
                                   motion_scale=self.motion_scale)
            row_iter = ((i, r) for i, r in enumerate(mock.rows()))
            self.status['n_rows'] = None
            self.status['source'] = 'live-mock'
        else:
            df = self.df_raw.iloc[self.start_row:]
            if self.limit is not None:
                df = df.iloc[:self.limit]
            row_iter = enumerate(df.to_dict('records'))
            self.status['n_rows'] = len(df)
            self.status['source'] = 'playback-csv'

        # Лимит длительности (план §7): duration_s в «секундах данных»,
        # плюс жёсткий потолок max_session_hours
        max_ticks = None
        if self.duration_s is not None:
            max_ticks = int(self.duration_s / cfg.tick_seconds)
        max_ticks_hours = int(cfg.max_session_hours * 3600 / cfg.tick_seconds)
        max_ticks = min(t for t in (max_ticks, max_ticks_hours) if t is not None)

        session_path = self.out_dir / 'session.csv'
        fore_path = self.out_dir / 'forecasts.csv'

        snapshot = {
            'mode': f'web-{self.source_mode}',
            'session_id': self.session_id,
            'model_run_id': getattr(self.predictor, 'run_id', None),
            'start_row': self.start_row,
            'speed': self.speed,
            'time_scale': self.time_scale,
            'seed': self.seed,
            'duration_s': self.duration_s,
            'anomaly_every': self.anomaly_every,
            'motion_scale': self.motion_scale,
            'limit': self.limit,
            'sequence_length': engine.seq_len,
            'prediction_horizon': engine.horizon,
            'targets': target_columns,
            'online_config': cfg.to_dict(),
        }
        (self.out_dir / 'config_snapshot.json').write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')

        def log_event(level, msg):
            ts = datetime.now().isoformat(timespec='seconds')
            with open(self.out_dir / 'events.log', 'a', encoding='utf-8') as f:
                f.write(f'{ts} | {level} | {msg}\n')

        with open(session_path, 'w', encoding='utf-8', newline='') as f_session, \
             open(fore_path, 'w', encoding='utf-8', newline='') as f_fore:

            session_writer = csv_mod.writer(f_session)
            session_writer.writerow(['tick', 'predicted', 'input_anomaly',
                                     'input_anomaly_features', 'error_anomaly',
                                     'error_ratio'] + raw_columns)
            fore_writer = csv_mod.writer(f_fore)
            fore_writer.writerow(['tick', 'horizon_step'] +
                                 [f'pred_{c}' for c in target_columns])

            tick = 0
            for tick, row in row_iter:
                if self._stop_flag.is_set():
                    break
                if max_ticks is not None and tick >= max_ticks:
                    log_event('INFO', f'Достигнут лимит длительности: {tick} tick')
                    break
                while self._paused.is_set() and not self._stop_flag.is_set():
                    time.sleep(0.05)
                if self._stop_flag.is_set():
                    break

                result = engine.on_tick(tick, row)
                try:
                    engine.observe_actual(tick, row)
                except (KeyError, TypeError, ValueError) as e:
                    log_event('ERROR', f'tick {tick}: observe_actual: {e}')

                session_writer.writerow(
                    [tick, int(result.predicted), int(result.input_anomaly),
                     result.input_anomaly_features, int(engine.last_error_marker),
                     '' if engine.last_error_ratio is None
                        else f'{engine.last_error_ratio:.3f}']
                    + [row.get(c, '') for c in raw_columns])

                if result.predicted:
                    self.status['predicted'] += 1
                    for h in range(engine.horizon):
                        fore_writer.writerow(
                            [tick, h + 1]
                            + [round(float(v), 6) for v in result.preds[h]])

                if result.input_anomaly:
                    self.status['input_anomaly_ticks'] += 1
                if engine.last_error_marker:
                    self.status['error_anomaly_ticks'] += 1
                self.status['tick'] = tick + 1

                self._broadcast({
                    'type': 'tick',
                    'tick': tick,
                    'state': self.status['state'],
                    'warmup': result.warmup,
                    'predicted': result.predicted,
                    'input_anomaly': result.input_anomaly,
                    'input_anomaly_features': result.input_anomaly_features,
                    'error_anomaly': engine.last_error_marker,
                    'error_ratio': engine.last_error_ratio,
                    'inference_ms': result.inference_ms,
                    'actual': {c: float(row[c]) for c in target_columns},
                    'preds': None if result.preds is None
                    else [[round(float(v), 4) for v in row_]
                         for row_ in result.preds],
                })

                # ритм: привязка к монотонным часам, пауза не копит дрейф.
                # playback — speed; live — time_scale (1.0 = реальное время 1 Гц)
                pace = self.speed if self.source_mode == 'playback' else self.time_scale
                if pace > 0:
                    period = cfg.tick_seconds / pace
                    end_t = time.monotonic() + period
                    while not self._stop_flag.is_set() and time.monotonic() < end_t:
                        time.sleep(min(0.05, max(0.0, end_t - time.monotonic())))

        self.stats_snapshot = engine.error_stats()
        self.status['state'] = 'stopped' if self.status['state'] != 'stalled' \
            else self.status['state']
        self._broadcast({
            'type': 'end',
            'status': dict(self.status),
            'error_stats': sanitize_error_stats(self.stats_snapshot),
        })
        (self.out_dir / 'summary.txt').write_text(
            json.dumps({'status': self.status,
                        'error_stats': sanitize_error_stats(self.stats_snapshot)},
                       ensure_ascii=False, indent=2, default=float),
            encoding='utf-8')
        log_event('INFO', f'Сессия завершена: {dict(self.status)}')


# ===========================================================================
# REST
# ===========================================================================

class StartRequest(BaseModel):
    model: str
    csv: str
    speed: float = 1.0
    limit: int | None = None
    start_row: int = 0
    # Режим B (ШАГ 4):
    source: str = 'playback'          # 'playback' | 'live'
    seed: int = 42
    time_scale: float = 1.0           # live: 1.0 = реальное время; >1 — только dev
    duration_s: float | None = None   # лимит длительности (план §7)
    anomaly_every: int | None = None  # период «уникальных вставок» mock-сенсора
    motion_scale: float = 2.0         # амплитуда качки во вставке (сдвиг физики)


class PauseRequest(BaseModel):
    paused: bool


class SpeedRequest(BaseModel):
    speed: float  # 0 или отрицательное = max (без пауз)


@app.get('/')
async def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/models')
async def api_models():
    models = []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None:
            continue
        models.append({
            'run_id': m.get('run_id', run['dir'].name),
            'status': m.get('status'),
            'hypothesis': m.get('hypothesis', ''),
            'sequence_length': m.get('model', {}).get('sequence_length'),
            'prediction_horizon': m.get('model', {}).get('prediction_horizon'),
            'mae': (m.get('results', {}).get('physical', {})
                    .get('overall', {}).get('mae')),
        })
    return models


@app.get('/api/datasets')
async def api_datasets():
    return sorted([p.name for p in DATA_DIR.glob('*.csv')])


def sanitize_error_stats(stats):
    """error_stats → JSON-безопасный dict (numpy-массивы → списки/float)."""
    if not stats:
        return None
    out = {'count': int(stats.get('count', 0))}
    pt = stats.get('per_target')
    if pt:
        out['per_target'] = {k: float(v) for k, v in pt.items()}
    return out


@app.get('/api/session/status')
async def api_status():
    runner = SESSION.runner
    if runner is None:
        return {'running': False}
    return {
        'running': runner.is_alive(),
        **runner.status,
        'speed': runner.speed,
        'n_rows': getattr(runner, 'n_rows', runner.status.get('n_rows')),
        'error_stats': sanitize_error_stats(
            runner.stats_snapshot or
            (runner.engine.error_stats() if runner.engine else None)),
    }


@app.post('/api/session/start')
async def api_start(req: StartRequest):
    if SESSION.running:
        return JSONResponse({'error': 'Сессия уже запущена'}, status_code=409)

    run_dir = reg.resolve_run(req.model)
    manifest = reg.load_manifest(run_dir)
    if manifest is None:
        return JSONResponse({'error': f'У run нет manifest: {run_dir.name}'},
                            status_code=400)
    ckpt = run_dir / 'checkpoints' / 'best_model.pt'
    scalers = run_dir / 'checkpoints' / 'scalers.pkl'
    if not (ckpt.exists() and scalers.exists()):
        return JSONResponse({'error': 'Нет checkpoints/best_model.pt или scalers.pkl'},
                            status_code=400)

    predictor = VesselPredictor_Inference(str(ckpt), str(scalers))
    fe_params = extract_fe_params(manifest)
    df_raw = load_raw_csv(DATA_DIR / req.csv)

    runner = PlaybackRunner(predictor, df_raw, fe_params, CFG,
                            speed=max(0.0, req.speed), limit=req.limit,
                            start_row=req.start_row,
                            source_mode=req.source,
                            time_scale=max(0.0, req.time_scale),
                            seed=req.seed,
                            duration_s=req.duration_s,
                            anomaly_every=req.anomaly_every,
                            motion_scale=req.motion_scale)
    with SESSION.lock:
        SESSION.runner = runner
    runner.start()
    return {'session_id': runner.session_id, 'n_rows': runner.n_rows}


@app.post('/api/session/stop')
async def api_stop():
    if not SESSION.running:
        return JSONResponse({'error': 'Сессия не запущена'}, status_code=409)
    SESSION.runner.stop()
    return {'ok': True}


@app.post('/api/session/pause')
async def api_pause(req: PauseRequest):
    if not SESSION.running:
        return JSONResponse({'error': 'Сессия не запущена'}, status_code=409)
    SESSION.runner.set_paused(req.paused)
    return {'ok': True, 'paused': req.paused}


@app.post('/api/session/speed')
async def api_speed(req: SpeedRequest):
    if not SESSION.running:
        return JSONResponse({'error': 'Сессия не запущена'}, status_code=409)
    SESSION.runner.set_speed(max(0.0, req.speed))
    return {'ok': True, 'speed': SESSION.runner.speed}


# ===========================================================================
# WebSocket
# ===========================================================================

@app.websocket('/ws')
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    runner = SESSION.runner
    if runner is None or not runner.is_alive():
        await ws.send_json({'type': 'no_session'})
        await ws.close()
        return
    q = runner.subscribe()
    try:
        while True:
            try:
                msg = await asyncio.to_thread(q.get, timeout=30.0)
            except Exception:
                continue  # таймаут молчания — просто ждём дальше
            await ws.send_json(msg)
            if msg.get('type') == 'end':
                break
    except WebSocketDisconnect:
        pass
    finally:
        runner.unsubscribe(q)


# ===========================================================================
# ПАНЕЛЬ УПРАВЛЕНИЯ ПРОЕКТОМ (ШАГ 7): фоновые задачи + REST для всех действий
# ===========================================================================

import subprocess as sp
import uuid

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


class JobRunner(threading.Thread):
    """Фоновая subprocess-задача с журналом (план §7: bounded, с таймаутом)."""

    def __init__(self, job_id: str, kind: str, cmd: list, timeout_min: float = 600):
        super().__init__(daemon=True, name=f'job-{job_id[:8]}')
        self.id, self.kind, self.cmd = job_id, kind, cmd
        self.timeout_s = timeout_min * 60
        self.proc = None

    def run(self):
        job = JOBS[self.id]
        try:
            self.proc = sp.Popen(self.cmd, cwd=PROJECT_ROOT,
                                 stdout=sp.PIPE, stderr=sp.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
            for line in self.proc.stdout:
                job['log'].append(line.rstrip())
                if len(job['log']) > 2000:  # bound памяти
                    del job['log'][:1000]
            self.proc.wait(timeout=self.timeout_s)
            job['returncode'] = self.proc.returncode
            job['status'] = 'done' if self.proc.returncode == 0 else 'failed'
        except sp.TimeoutExpired:
            self.proc.kill()
            job['status'] = 'timeout'
            job['log'].append(f'WALL-CLOCK ТАЙМАУТ {self.timeout_s / 60:.0f} мин — процесс остановлен')
        except Exception as e:
            job['status'] = 'failed'
            job['log'].append(f'ОШИБКА: {e}')
        finally:
            job['finished'] = datetime.now().isoformat(timespec='seconds')


def start_job(kind: str, cmd: list, timeout_min: float = 600) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {'id': job_id, 'kind': kind, 'status': 'running',
                        'log': [], 'started': datetime.now().isoformat(timespec='seconds'),
                        'finished': None, 'returncode': None}
    JobRunner(job_id, kind, cmd, timeout_min).start()
    return job_id


# --- справка панели (детальные тексты «?» для UI) ---------------------------

def _config_snapshot() -> dict:
    from config.config import Config
    c = Config(verbose=False)
    return {
        'profile': c.profile,
        'sequence_length': c.sequence_length,
        'prediction_horizon': c.prediction_horizon,
        'n_features': c.input_dim,
        'n_targets': c.output_dim,
        'targets': c.target_columns,
        'learning_rate': c.learning_rate,
        'num_epochs': c.num_epochs,
        'batch_size': c.batch_size,
        'profiles': ['motion_prediction', 'motion_core_prediction',
                     'rot_prediction', 'speed_prediction', 'full_prediction'],
    }


class TrainRequest(BaseModel):
    profile: str = 'full_prediction'
    csv: str = 'your_data.csv'
    skip_rows: int = 4000
    train_segments: int = 9
    subset_seed: int | None = 123
    max_epochs: int = 100
    lr: float | None = None
    seed: int = 42
    notes: str = 'panel train'
    init_from: str | None = None      # warm start (дообучение из панели)
    fixed_scaler: bool = True


class SyntheticRequest(BaseModel):
    rows: int = 30000
    seed: int = 42
    fit_train_only: bool = True


class PromoteRequest(BaseModel):
    run_id: str


class DeleteRequest(BaseModel):
    run_id: str
    confirm: bool = False


class FinetuneRequest(BaseModel):
    session: str
    run_id: str
    min_rows: int | None = None
    max_epochs: int | None = None
    all_rows: bool = False
    skip_cooldown: bool = False


class PredictRequest(BaseModel):
    run_id: str
    csv: str
    start_row: int = 4000


class UploadRequest(BaseModel):
    name: str
    data_b64: str


@app.post('/api/datasets/upload')
async def api_upload(req: UploadRequest):
    """Добавить дата-сет от эксперимента: сохраняем в data/raw с валидацией колонок."""
    if '/' in req.name or '\\' in req.name or '..' in req.name:
        return JSONResponse({'error': 'недопустимое имя файла'}, status_code=400)
    try:
        raw = base64.b64decode(req.data_b64)
    except Exception:
        return JSONResponse({'error': 'неверный base64'}, status_code=400)
    text = raw.decode('utf-8', errors='replace')
    import io
    try:
        df = pd.read_csv(io.StringIO(text), sep='\t')
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            return JSONResponse({'error': f'не удалось разобрать CSV: {e}'}, status_code=400)
    required = ['Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)', 'SOG(узлы)', 'ROT(°/мин)']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return JSONResponse({'error': f'нет обязательных колонок: {missing}'},
                            status_code=400)
    out = DATA_DIR / req.name
    if not out.suffix:
        out = out.with_suffix('.csv')
    if out.exists():
        return JSONResponse({'error': f'{out.name} уже существует'}, status_code=409)
    df.to_csv(out, sep='\t', index=False)
    return {'ok': True, 'file': out.name, 'rows': len(df), 'columns': len(df.columns)}


@app.get('/api/config')
async def api_config():
    return _config_snapshot()


@app.post('/api/train')
async def api_train(req: TrainRequest):
    cmd = [sys.executable, 'run_training.py',
           '--data-path', str(DATA_DIR / req.csv),
           '--profile', req.profile,
           '--skip-rows', str(req.skip_rows),
           '--train-segments', str(req.train_segments),
           '--max-epochs', str(req.max_epochs),
           '--seed', str(req.seed),
           '--notes', f'panel: {req.notes}']
    if req.subset_seed is not None:
        cmd += ['--subset-seed', str(req.subset_seed)]
    if req.lr is not None:
        cmd += ['--lr', str(req.lr)]
    if req.fixed_scaler:
        cmd.append('--fixed-scaler')
    if req.init_from:
        cmd += ['--init-from', str(reg.resolve_run(req.init_from))]
    job = start_job('train', cmd, timeout_min=600)
    return {'job_id': job}


@app.post('/api/predict')
async def api_predict(req: PredictRequest):
    """Тестовый прогноз одним окном (инпрокесс, быстро)."""
    run_dir = reg.resolve_run(req.run_id)
    manifest = reg.load_manifest(run_dir)
    ckpt = run_dir / 'checkpoints' / 'best_model.pt'
    scalers = run_dir / 'checkpoints' / 'scalers.pkl'
    predictor = VesselPredictor_Inference(str(ckpt), str(scalers))
    fe = extract_fe_params(manifest)
    df = load_raw_csv(DATA_DIR / req.csv)
    if fe['cyclic'] or fe['relative_wave_angle']:
        from data.features import engineer_features_dataframe
        df = engineer_features_dataframe(df, cyclic=fe['cyclic'],
                                          relative_wave_angle=fe['relative_wave_angle'])
    seq, hor = predictor.config.sequence_length, predictor.config.prediction_horizon
    start = max(0, min(req.start_row, len(df) - seq - hor))
    hist_end = start + seq
    input_seq = df[predictor.config.feature_columns].iloc[start:hist_end].values
    preds = predictor.predict(np.asarray(input_seq, dtype=np.float64))
    actual = None
    mae = None
    t_end = hist_end + hor
    if t_end <= len(df):
        actual = df[predictor.config.target_columns].iloc[hist_end:t_end].values
        mae = {c: float(np.mean(np.abs(preds[:, i] - actual[:, i])))
               for i, c in enumerate(predictor.config.target_columns)}
    return {
        'run_id': run_dir.name,
        'targets': predictor.config.target_columns,
        'history': df[predictor.config.target_columns].iloc[
            max(0, hist_end - 120):hist_end].values.tolist(),
        'predictions': preds.tolist(),
        'actual': None if actual is None else actual.tolist(),
        'mae': mae,
        'start_row': start,
    }


@app.post('/api/synthetic')
async def api_synthetic(req: SyntheticRequest):
    out = DATA_DIR / f'synthetic_panel_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    cmd = [sys.executable, 'scripts/generate_synthetic_data.py',
           '--rows', str(req.rows), '--seed', str(req.seed),
           '--out', str(out), '--validate']
    if req.fit_train_only:
        cmd += ['--fit-train-only', '--skip-rows', '4000']
    return {'job_id': start_job('synthetic', cmd, timeout_min=120), 'out': str(out.name)}


@app.post('/api/finetune')
async def api_finetune(req: FinetuneRequest):
    session_path = PROJECT_ROOT / 'results' / 'online' / req.session
    cmd = [sys.executable, '-m', 'online.finetune',
           '--session', str(session_path), '--run-id', req.run_id]
    if req.min_rows is not None:
        cmd += ['--min-rows', str(req.min_rows)]
    if req.max_epochs is not None:
        cmd += ['--max-epochs', str(req.max_epochs)]
    if req.all_rows:
        cmd.append('--all-rows')
    if req.skip_cooldown:
        cmd.append('--skip-cooldown')
    return {'job_id': start_job('finetune', cmd, timeout_min=60)}


@app.get('/api/sessions')
async def api_sessions():
    base = PROJECT_ROOT / 'results' / 'online'
    out = []
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir() and (d / 'session.csv').exists():
                out.append({'name': d.name, 'has_csv': True})
    return out


@app.post('/api/models/promote')
async def api_promote(req: PromoteRequest):
    run_dir = reg.resolve_run(req.run_id)
    reg.update_manifest(run_dir, {'status': 'production'})
    return {'ok': True, 'run_id': run_dir.name, 'status': 'production'}


@app.post('/api/models/delete')
async def api_delete(req: DeleteRequest):
    if not req.confirm:
        return JSONResponse({'error': 'Требуется confirm=true'}, status_code=400)
    try:
        reg.delete_run(req.run_id)
        return {'ok': True}
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


@app.get('/api/jobs')
async def api_jobs():
    with JOBS_LOCK:
        return sorted(JOBS.values(), key=lambda j: j['started'], reverse=True)


@app.get('/api/jobs/{job_id}')
async def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return JSONResponse({'error': 'нет такой задачи'}, status_code=404)
        return {**job, 'log': job['log'][-80:]}


def main():
    if hasattr(__import__('sys').stdout, 'reconfigure'):
        import sys
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    import uvicorn
    print(f'▶ Онлайн-система (режим A): http://127.0.0.1:{args.port}')
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
