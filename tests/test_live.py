"""
Unit-тесты ШАГА 4 — live-источники (online/sources.py).

Запуск: py -3.13 tests/test_live.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

from online.sources import MockLiveSource, FileLiveSource


def make_df(n=2000):
    return pd.DataFrame({
        'Rudder Order(градусы)': np.zeros(n),
        'Rudder State(градусы)': np.zeros(n),
        'Roll(градусы)': np.sin(np.linspace(0, 50, n)),
        'ROT(°/мин)': np.zeros(n),
        'time': np.arange(n),
    })


def test_mock_determinism():
    df = make_df()
    a = [dict(r) for _, r in zip(range(50), MockLiveSource(df, seed=7).rows())]
    b = [dict(r) for _, r in zip(range(50), MockLiveSource(df, seed=7).rows())]
    c = [dict(r) for _, r in zip(range(50), MockLiveSource(df, seed=8).rows())]
    assert a == b, 'один seed → одна и та же последовательность'
    assert a != c, 'разные seed → разные последовательности'
    print('  ✓ MockLiveSource: детерминизм по seed')


def test_mock_burst():
    df = make_df()
    src = MockLiveSource(df, seed=42, anomaly_every=100, anomaly_len=10,
                         rudder_step_deg=25.0, motion_scale=2.0)
    rows = [dict(r) for _, r in zip(range(300), src.rows())]
    bursts = [i for i, r in enumerate(rows)
              if abs(r['Rudder Order(градусы)']) > 1e-6]
    assert bursts, 'вставки должны быть'
    # фазы вставок: [0..9], [100..109], [200..209]
    expect = set(list(range(0, 10)) + list(range(100, 110)) + list(range(200, 210)))
    assert set(bursts) == expect, f'фазы вставок не совпали: {sorted(set(bursts))[:5]}...'
    # знак внутри вставки меняется (ступенька): 0..5 один, 6..9 противоположный
    assert abs(rows[0]['Rudder Order(градусы)'] + rows[7]['Rudder Order(градусы)']) < 1e-6
    # вне вставок строка идентична исходной
    assert rows[50]['Roll(градусы)'] == df.iloc[src._start + 50]['Roll(градусы)']
    # амплитуда качки во вставке ×motion_scale (сдвиг физики)
    assert abs(rows[3]['Roll(градусы)'] - 2.0 * df.iloc[src._start + 3]['Roll(градусы)']) < 1e-9
    print('  ✓ MockLiveSource: вставки периодичны, ступенька + сдвиг амплитуды ×2')


def test_file_live_source():
    df = make_df(10)
    src = FileLiveSource(df)
    got = [src.get_row(1.0) for _ in range(12)]
    assert len([r for r in got if r is not None]) == 10
    assert got[-1] is None and src.exhausted
    print('  ✓ FileLiveSource: отдаёт строки и честно иссякает (None)')


if __name__ == '__main__':
    test_mock_determinism()
    test_mock_burst()
    test_file_live_source()
    print('\nOK — все тесты ШАГА 4 пройдены')
