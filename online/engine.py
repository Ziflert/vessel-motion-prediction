"""
PredictionEngine — движок онлайн-прогноза (ШАГ 1 плана).

Поток данных на один tick (см. план §3):
    raw-строка → RingBuffer → инженерия окна (та же, что при обучении)
    → проверка NaN → predict() под WATCHDOG → прогноз [horizon, targets]
    → регистрация прогноза для последующей сверки «факт vs прогноз».

Watchdog (план §7): один вызов predict ограничен inference_timeout_s.
Превышение → tick помечается stale, исполнитель пересоздаётся (поток-зомби
бросается; их число ограничено stale_pause_threshold, после чего источник
должен быть остановлен вызывающей стороной).
"""
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.online import OnlineConfig
from data.features import engineer_features_dataframe
from online.buffer import RingBuffer
from online.anomaly import (InputAnomalyDetector, ErrorMonitor,
                            baseline_mae_from_manifest)


class EngineStalledError(RuntimeError):
    """N подряд stale-прогнозов — источник обязан остановиться (план §7)."""


@dataclass
class TickResult:
    """Результат обработки одного tick'а источника."""
    tick: int
    warmup: bool = False          # окно ещё не заполнено
    predicted: bool = False       # прогноз получен
    preds: np.ndarray = None      # [horizon, n_targets] в физических единицах
    skipped_reason: str = ''      # 'nan_window' | 'timeout' | ''
    inference_ms: float = 0.0
    input_anomaly: bool = False   # маркер: вход вне train-распределения (ШАГ 2)
    input_anomaly_features: str = ''
    error_anomaly: bool = False   # маркер: модель устойчиво ошибается (ШАГ 2)
    error_ratio: float = None     # во сколько раз ошибка выше базовой MAE модели


class PredictionEngine:
    def __init__(self, predictor, fe_params: dict, cfg: OnlineConfig):
        """
        predictor — VesselPredictor_Inference (run_inference.py)
        fe_params — {'cyclic': bool, 'relative_wave_angle': bool} из manifest
        """
        self.predictor = predictor
        self.fe = dict(fe_params)
        self.cfg = cfg

        self.seq_len = predictor.config.sequence_length
        self.horizon = predictor.config.prediction_horizon
        self.feature_columns = list(predictor.config.feature_columns)
        self.target_columns = list(predictor.config.target_columns)

        maxlen = max(self.seq_len, 600) * cfg.buffer_multiplier
        self.buffer = RingBuffer(maxlen=maxlen)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='infer')
        self._consecutive_stale = 0

        # --- ШАГ 2: детекторы аномалий ---
        self.input_detector = InputAnomalyDetector(
            predictor.config.feature_columns, predictor.feature_scaler, cfg)
        baseline = baseline_mae_from_manifest(
            getattr(predictor, 'manifest', None), predictor.config.target_columns)
        self.error_monitor = None
        if baseline is not None:
            self.error_monitor = ErrorMonitor(baseline, cfg)
        self.error_monitor_enabled = self.error_monitor is not None
        self.last_error_marker = False
        self.last_error_ratio = None

        # Реализованная ошибка «факт vs прогноз» (заполняется observe_actual)
        self._forecasts = {}   # tick -> np.ndarray [horizon, n_targets]
        self._reset_error_stats()

    # ------------------------------------------------------------------
    # Основной поток
    # ------------------------------------------------------------------

    def on_tick(self, tick: int, raw_row: dict) -> TickResult:
        self.buffer.add(raw_row)

        result = TickResult(tick=tick)
        result.input_anomaly, result.input_anomaly_features = self._check_input(tick)

        if not self.buffer.is_ready(self.seq_len):
            result.warmup = True
            return result

        input_seq = self._engineer_window()
        if input_seq is None:
            result.skipped_reason = 'nan_window'
            return result

        preds, ms, stale = self._run_predict(input_seq)
        if stale:
            result.skipped_reason = 'timeout'
            return result

        result.predicted = True
        result.preds = preds
        result.inference_ms = ms
        self._forecasts[tick] = preds
        self._cleanup_forecasts(tick)
        return result

    # ------------------------------------------------------------------
    # Сверка «факт vs прогноз»
    # ------------------------------------------------------------------

    def observe_actual(self, tick: int, raw_row: dict):
        """
        Фактические значения целей на tick пришли: сверяем со ВСЕМИ прогнозами,
        сделанными на предыдущих tick'ах для этого момента (lag 1..horizon),
        затем обновляем монитор ошибки (ШАГ 2).
        """
        actual = np.array([float(raw_row[c]) for c in self.target_columns],
                          dtype=np.float64)
        tick_err = np.zeros(len(self.target_columns))
        tick_count = 0
        for h in range(1, self.horizon + 1):
            preds = self._forecasts.get(tick - h)
            if preds is None:
                continue
            err = np.abs(preds[h - 1] - actual)           # [n_targets]
            self._err_sum_targets += err
            self._err_count += 1
            self._err_sum_horizon[h - 1] += float(np.mean(err))
            self._err_horizon_count[h - 1] += 1
            tick_err += err
            tick_count += 1

        # ШАГ 2: монитор ошибки «модель устойчиво ошибается»
        if tick_count and self.error_monitor is not None:
            ratio_mean = tick_err / tick_count
            self.error_monitor.observe(tick, ratio_mean)
            self.last_error_marker = self.error_monitor.marker
            self.last_error_ratio = self.error_monitor.last_ratio
        elif not self.error_monitor_enabled:
            self.last_error_ratio = None

    def _cleanup_forecasts(self, current_tick: int):
        """Удаляем прогнозы, которые уже не понадобятся (bound памяти, план §7)."""
        horizon = self.horizon
        for t in [t for t in self._forecasts if t < current_tick - horizon - 1]:
            del self._forecasts[t]

    # ------------------------------------------------------------------
    # Прогноз окна (инженерия + NaN-гейт + watchdog)
    # ------------------------------------------------------------------

    def _engineer_window(self):
        window = self.buffer.window_df(self.seq_len)
        engineered = engineer_features_dataframe(
            window,
            cyclic=self.fe['cyclic'],
            relative_wave_angle=self.fe['relative_wave_angle'],
        )
        input_df = engineered[self.feature_columns]
        if input_df.isna().values.any():
            return None
        return np.asarray(input_df.values, dtype=np.float64)

    def _check_input(self, tick: int):
        """ШАГ 2: аномалия входа на последней строке (без прогрева окна)."""
        last = self.buffer.window_df(1)
        if last.empty:
            return False, ''
        try:
            engineered = engineer_features_dataframe(
                last, cyclic=self.fe['cyclic'],
                relative_wave_angle=self.fe['relative_wave_angle'])
            row = engineered[self.feature_columns]
        except KeyError:
            return False, ''
        if row.isna().values.any():
            return False, ''
        result = self.input_detector.check(
            np.asarray(row.values[0], dtype=np.float64), tick)
        # разделитель ';' — имена признаков пишутся в CSV (запятая сломала бы колонки;
        # урок приёмки ШАГА 4: статус-счётчик говорил 64, а CSV-колонка была пуста)
        return result['marker'], ';'.join(result['features'])

    def _run_predict(self, input_seq: np.ndarray):
        """Возвращает (preds, ms, stale). stale=True — таймаут watchdog'а."""
        future = self._executor.submit(self.predictor.predict, input_seq)
        t0 = time.monotonic()
        try:
            preds = future.result(timeout=self.cfg.inference_timeout_s)
        except FutureTimeoutError:
            # Исполнитель занят «висящим» вызовом: пересоздаём (поток-зомби
            # остаётся, их число ограничено stale_pause_threshold — план §7).
            self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=1,
                                                 thread_name_prefix='infer')
            self._consecutive_stale += 1
            if self._consecutive_stale >= self.cfg.stale_pause_threshold:
                raise EngineStalledError(
                    f"{self._consecutive_stale} подряд stale-прогнозов "
                    f"(timeout={self.cfg.inference_timeout_s}s)")
            return None, 0.0, True

        self._consecutive_stale = 0
        ms = (time.monotonic() - t0) * 1000.0
        return np.asarray(preds), ms, False

    # ------------------------------------------------------------------
    # Статистика (для summary.txt / UI)
    # ------------------------------------------------------------------

    def _reset_error_stats(self):
        n_targets = len(self.target_columns)
        self._err_sum_targets = np.zeros(n_targets)
        self._err_count = 0
        self._err_sum_horizon = np.zeros(self.horizon)
        self._err_horizon_count = np.zeros(self.horizon, dtype=np.int64)

    def error_stats(self):
        """Усреднённые ошибки «факт vs прогноз» (физические единицы)."""
        if self._err_count == 0:
            return {'count': 0}
        per_target = {
            col: float(self._err_sum_targets[i] / self._err_count)
            for i, col in enumerate(self.target_columns)
        }
        horizon_counts = np.maximum(self._err_horizon_count, 1)
        per_horizon = self._err_sum_horizon / horizon_counts
        return {
            'count': int(self._err_count),
            'per_target': per_target,
            'per_horizon': per_horizon,
        }

    @property
    def consecutive_stale(self) -> int:
        return self._consecutive_stale
