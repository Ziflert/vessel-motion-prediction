"""
Источники данных онлайн-системы.

Контракт (ADR-7): источник выдаёт RAW-строку (dict) той же схемы, что колонки
train-CSV, ДО инженерии признаков. Инженерию система применяет сама той же
`engineer_features_dataframe`, что при обучении (параметры — из manifest модели).

- TestSource  — playback CSV: одна строка за tick (ШАГ 1).
- LiveSource  — интерфейс реального источника (реализация — ШАГ 4).
"""
import time
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# ============================================================================
# Загрузка CSV — зеркально run_training.load_data (тот же sniffing формата)
# ============================================================================

def load_raw_csv(data_path) -> pd.DataFrame:
    """CSV с fallback-ами (tab/utf-16 и т.п.) + drop 'time'. Без инженерии."""
    data_path = Path(data_path)
    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except Exception:
        try:
            df = pd.read_csv(data_path, sep='\t')
        except Exception:
            df = pd.read_csv(data_path)

    if 'time' in df.columns:
        df = df.drop('time', axis=1)
    return df


def extract_fe_params(manifest: dict) -> dict:
    """Параметры инженерии признаков из manifest модели (fallback: без FE)."""
    eng = (manifest or {}).get('features', {}).get('feature_engineering')
    if eng:
        return {
            'cyclic': eng['cyclic_encoding'],
            'relative_wave_angle': eng['relative_wave_angle'],
        }
    return {'cyclic': False, 'relative_wave_angle': False}


# ============================================================================
# Режим A: playback CSV
# ============================================================================

class TestSource:
    """
    Playback готового CSV с ритмом 1 строка/tick.

    speed > 0  — пауза tick_seconds/speed между строками (привязка к
                  time.monotonic: пропуски, а не «догон» — нет дрейфа).
    speed <= 0 — «max»: без пауз (быстрый прогон, bound = конец файла).
    """

    def __init__(self, df_raw: pd.DataFrame, speed: float = 1.0,
                 limit: int = None, start_row: int = 0, tick_seconds: float = 1.0):
        if start_row < 0:
            raise ValueError(f"start_row must be >= 0, got {start_row}")
        df = df_raw.iloc[start_row:]
        if limit is not None:
            df = df.iloc[:limit]
        self._rows = df.to_dict('records')
        self._speed = float(speed)
        self._tick_seconds = float(tick_seconds)
        self.n_rows = len(self._rows)

    def __iter__(self):
        rows = self._rows
        if not rows:
            return
        period = self._tick_seconds / self._speed if self._speed > 0 else 0.0
        start = time.monotonic()
        for i, row in enumerate(rows):
            if period > 0:
                target = start + i * period
                sleep_s = target - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            yield i, row


# ============================================================================
# Режим B: live-источники
# ============================================================================

class LiveSource(ABC):
    """
    Интерфейс источника реальных данных (сенсоры).

    get_row() должен вернуть:
      - dict «колонка → значение» (raw-схема train-CSV);
      - None — данных нет (tick пропускается по таймауту источника).
    Блокировка допускается не дольше timeout_s (внутренний контракт источника).
    Контракт для реальных сенсоров — docs/ONLINE_API.md (ADR-7).
    """

    @abstractmethod
    def get_row(self, timeout_s: float) -> dict | None:
        raise NotImplementedError


class FileLiveSource(LiveSource):
    """CSV как «сенсор»: те же данные, но без знания будущего в UI (план §5)."""

    def __init__(self, df_raw: pd.DataFrame, start_row: int = 0):
        self._rows = df_raw.iloc[start_row:].to_dict('records')
        self._i = 0

    def get_row(self, timeout_s: float = 1.0) -> dict | None:
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._rows)


class MockLiveSource(LiveSource):
    """
    Сидированный генератор raw-строк для разработки/приёмки режима B (план §5).

    Физика: циклический проход по реальному CSV с сид-зависимого старта —
    динамика качки непрерывна (не перемешиваем!). Опционально — ПЕРИОДИЧЕСКИЕ
    «уникальные вставки»: ступенька руля (±25°, в пределах диапазона train ±35°),
    на которой модель, обученная на штатной динамике, должна ошибаться →
    error_anomaly для приёмки контура дообучения.

    Детерминизм: один seed → одна и та же последовательность строк и вставок
    (приёмка ШАГА 4: маркеры воспроизводятся повторным прогоном).
    """

    def __init__(self, df_raw: pd.DataFrame, seed: int = 42,
                 anomaly_every: int | None = None, anomaly_len: int = 40,
                 rudder_step_deg: float = 25.0, motion_scale: float = 2.0):
        self._df = df_raw
        self._anomaly_every = anomaly_every
        self._anomaly_len = anomaly_len
        self._rudder_step = rudder_step_deg
        self._motion_scale = motion_scale
        self._motion_cols = [c for c in df_raw.columns if c.startswith(
            ('Pitch', 'Roll', 'Vertical', 'Velocity.', 'ROT'))]
        rng = np.random.default_rng(seed)
        self._start = int(rng.integers(0, max(1, len(df_raw) - 1000)))
        self._burst_sign = float(rng.choice([-1.0, 1.0]))
        self._i = 0

    def rows(self):
        """Бесконечный генератор строк (стоп — на стороне цикла сессии)."""
        n = len(self._df)
        i = self._start
        while True:
            row = dict(self._df.iloc[i % n])
            # «уникальная вставка»: ступенька руля (в пределах диапазона train ±35°):
            # первые 60% вставки руль в одном положении, дальше — в противоположном
            in_burst = False
            if self._anomaly_every:
                phase = (i - self._start) % self._anomaly_every
                in_burst = phase < self._anomaly_len
            if in_burst:
                sign = self._burst_sign if phase < self._anomaly_len * 0.6 else -self._burst_sign
                row['Rudder Order(градусы)'] = float(row.get('Rudder Order(градусы)', 0)) + sign * self._rudder_step
                row['Rudder State(градусы)'] = float(row.get('Rudder State(градусы)', 0)) + sign * self._rudder_step
                # Сдвиг физики: амплитуда качки ×motion_scale — режим, которого
                # нет в train. Именно этот сегмент — «уникальные данные»
                # (модель не может его предсказывать → error_anomaly)
                for c in self._motion_cols:
                    row[c] = float(row[c]) * self._motion_scale
            yield row
            i += 1
            self._i = i

    def get_row(self, timeout_s: float = 1.0) -> dict | None:
        return next(self.rows())

    @property
    def position(self) -> int:
        return self._i
