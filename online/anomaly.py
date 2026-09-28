"""
Детекторы «нетипичных» данных (ШАГ 2 плана; ADR-3, ADR-5).

Два НЕЗАВИСИМЫХ маркера (не смешивать!):

1. `input_anomaly`  — статистический выброс ВХОДА относительно train
   (z-score по engineered-признакам, статистики из scaler'а модели).
   Сигнал оператору «условия непохожи на обучающие» / «проверить датчик».
   НЕ является критерием дообучения (ADR-5: выброс может быть поломкой).

2. `error_anomaly`  — устойчивое превышение ошибки «факт vs прогноз»
   над порогом k × (MAE модели per-target, из manifest). Именно это —
   маркер «уникальных данных»: сама модель сообщает, что не умеет
   такое прогнозировать. Критерий дообучения (с участием человека).

Оба маркера проходят через hysteresis (анти-дребезг) + cooldown.
"""
from collections import deque

import numpy as np


# ============================================================================
# Анти-дребезг (план §6)
# ============================================================================

class Hysteresis:
    """
    Симметричный фильтр статуса:
    - включение: N подряд срабатываний;
    - выключение: N подряд несрабатываний;
    - смена статуса не чаще, чем раз в cooldown_ticks (план §7).
    """

    def __init__(self, consecutive: int = 2, cooldown_ticks: int = 10):
        if consecutive < 1:
            raise ValueError(f"consecutive must be >= 1, got {consecutive}")
        self.consecutive = int(consecutive)
        self.cooldown_ticks = int(cooldown_ticks)
        self.state = False
        self._run_true = 0
        self._run_false = 0
        self._last_flip_tick = -10**9

    def update(self, triggered: bool, tick: int) -> bool:
        self._run_true = self._run_true + 1 if triggered else 0
        self._run_false = self._run_false + 1 if not triggered else 0

        if self._run_true >= self.consecutive and not self.state:
            self._flip(True, tick)
        elif self._run_false >= self.consecutive and self.state:
            self._flip(False, tick)
        return self.state

    def _flip(self, value: bool, tick: int) -> None:
        if tick - self._last_flip_tick < self.cooldown_ticks:
            return  # cooldown: слишком частая смена запрещена (план §7)
        self.state = value
        self._last_flip_tick = tick


# ============================================================================
# 1. Аномалия входа (z-score по статистикам train)
# ============================================================================

class InputAnomalyDetector:
    """
    z-score на engineered-признаках последней строки окна.
    Статистики (mean/std) — из feature_scaler модели, т.е. train-распределения:
    порог «насколько условия непохожи на обучающие» без всякого обучения.

    Ограничение (ADR-3): корреляции признаков не учитываются
    (Mahalanobis — план v1.1). Вырожденные признаки (std=0, как
    Wave.direction в записи) не дают срабатываний.
    """

    def __init__(self, feature_columns, feature_scaler, cfg):
        self.feature_columns = list(feature_columns)
        self.mean = np.asarray(feature_scaler.mean_, dtype=np.float64)
        self.scale = np.asarray(feature_scaler.scale_, dtype=np.float64)
        if len(self.mean) != len(self.feature_columns):
            raise ValueError(
                f"scaler размером {len(self.mean)} не совпадает с числом "
                f"признаков {len(self.feature_columns)}")
        self.cfg = cfg

    def check(self, x: np.ndarray, tick: int) -> dict:
        """
        x — engineered-признаки последней строки [n_features].
        Возвращает {'triggered': bool, 'features': [...], 'max_z': float}.
        """
        z = np.abs(np.asarray(x, dtype=np.float64) - self.mean) / np.where(
            self.scale == 0, 1.0, self.scale)
        over = z > self.cfg.input_anomaly_z
        triggered = bool(over.any())
        features = [self.feature_columns[i]
                    for i in np.flatnonzero(over)][:5]
        result = {
            'triggered': triggered,
            'features': features,
            'max_z': float(np.nanmax(z)) if z.size else 0.0,
        }
        if not hasattr(self, '_hyst'):
            self._hyst = Hysteresis(self.cfg.hysteresis_consecutive,
                                    int(self.cfg.marker_cooldown_s))
        result['marker'] = self._hyst.update(result['triggered'], tick)
        return result


# ============================================================================
# 2. Аномалия ошибки (rolling-MAE vs базовый MAE модели)
# ============================================================================

class ErrorMonitor:
    """
    На каждый tick получает усреднённую по упреждениям ошибку per-target,
    нормирует на per-target MAE модели (из manifest, физические единицы) и
    сравнивает ОТНОШЕНИЕ КАЖДОЙ ЦЕЛИ с порогом error_anomaly_k.

    Срабатывание — по МАКСИМАЛЬНОМУ per-target ratio (не по среднему):
    среднее по 8 целям размывает сигнал (урок приёмки ШАГА 4: сдвиг физики
    на 2–3 целях тонул в базе, где Velocity.Rolling baseline = 71.6°/мин).

    ratio_max = max_over_targets(err_per_target / baseline_mae_per_target)
    triggered := ratio_max > k

    База (baseline) отсутствует в manifest → монитор отключён (marker=False),
    это логируется на стороне вызывающего (engine).
    """

    def __init__(self, baseline_per_target, cfg):
        self.baseline = np.maximum(
            np.asarray(baseline_per_target, dtype=np.float64), 1e-9)
        self.cfg = cfg
        self._hyst = Hysteresis(cfg.hysteresis_consecutive,
                               int(cfg.marker_cooldown_s))
        self.window = deque(maxlen=cfg.error_window_ticks)
        self.last_ratio = None
        self.n_observations = 0

    @property
    def threshold(self) -> float:
        return float(self.cfg.error_anomaly_k)

    def observe(self, tick: int, err_per_target: np.ndarray) -> dict:
        """err_per_target — MAE этого tick'а по каждой цели (физ. единицы)."""
        err = np.asarray(err_per_target, dtype=np.float64)
        ratios = err / self.baseline
        ratio_max = float(np.max(ratios))
        ratio_mean = float(np.mean(ratios))
        self.window.append(ratio_max)
        self.n_observations += 1
        self.last_ratio = ratio_max
        self.last_ratio_mean = ratio_mean
        triggered = ratio_max > self.threshold
        return {
            'triggered': bool(triggered),
            'ratio': ratio_max,
            'ratio_mean': ratio_mean,
            'marker': self._hyst.update(triggered, tick),
        }

    @property
    def marker(self) -> bool:
        return self._hyst.state


# ============================================================================
# Извлечение базовой ошибки per-target из manifest
# ============================================================================

def baseline_mae_from_manifest(manifest: dict, target_columns) -> list | None:
    """
    Per-target MAE модели в физических единицах:
    manifest['results']['physical']['per_target'][col]['mae'].
    Возвращает None, если данных нет (монитор отключается).
    """
    try:
        per_target = manifest['results']['physical']['per_target']
        return [float(per_target[c]['mae']) for c in target_columns]
    except (KeyError, TypeError, ValueError):
        return None
