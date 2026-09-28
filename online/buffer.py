"""
«Память» системы: скользящее окно raw-строк (ADR-4).

Модель — Seq2Seq БЕЗ скрытого состояния между прогнозами: контекст
держится только внутри окна sequence_length=120 строк (2 мин при 1 Гц).
Храним RAW-строки (до инженерии признаков) — инженерия применяется к окну
на каждом прогнозе той же функцией, что при обучении (ADR-7).
"""
from collections import deque
import pandas as pd


class RingBuffer:
    """Ограниченный буфер последних строк (deque с maxlen — расти не может)."""

    def __init__(self, maxlen: int):
        if maxlen < 1:
            raise ValueError(f"RingBuffer: maxlen must be >= 1, got {maxlen}")
        self._rows = deque(maxlen=int(maxlen))
        self._total_added = 0
        self._maxlen = int(maxlen)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def total_added(self) -> int:
        """Сколько строк прошло через буфер за всё время (для диагностики)."""
        return self._total_added

    def add(self, row: dict) -> None:
        self._total_added += 1
        self._rows.append(dict(row))

    def is_ready(self, need: int) -> bool:
        """Готово ли окно требуемой длины (прогрев)."""
        return len(self._rows) >= need

    def window_df(self, n: int) -> pd.DataFrame:
        """DataFrame из последних n raw-строк (порядок хронологический)."""
        n = min(n, len(self._rows))
        return pd.DataFrame(list(self._rows)[-n:])

    def last_row(self):
        return self._rows[-1] if self._rows else None

    def __len__(self) -> int:
        return len(self._rows)
