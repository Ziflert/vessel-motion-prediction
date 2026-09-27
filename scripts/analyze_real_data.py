"""
Анализ реальных данных для исследования «объём выборки vs качество».

Строит статистический портрет записи, который потом используется:
  1) как ТЗ для генератора синтетики (scripts/generate_synthetic_data.py);
  2) как доказательная база в RESEARCH_LOG.md (что именно мы имитируем).

Считает:
  - базовые статистики ключевых колонок;
  - спектральные пики (естественные периоды качки);
  - время автокорреляции (быстро/медленно меняющиеся величины);
  - шум (высокочастотный остаток после сглаживания);
  - приращения за 1 с (характер «быстрых изменений»);
  - кросс-корреляции (Roll↔волна, Руль→ROT);
  - состав режимов волнения (доли времени).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from run_training import load_data


def welch_psd(x, fs=1.0, nperseg=512):
    x = np.asarray(x, float)
    x = x - x.mean()
    step = nperseg // 2
    win = np.hanning(nperseg)
    segs = []
    for s in range(0, len(x) - nperseg + 1, step):
        segs.append(x[s:s + nperseg] * win)
    segs = np.array(segs)
    fft = np.fft.rfft(segs, axis=1)
    psd = (np.abs(fft) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1 / fs)
    psd = psd * 2 / (fs * (win ** 2).sum())
    psd[0] /= 2
    return freqs, psd


def peak_period(x, fmin=1 / 60, fmax=0.5):
    f, p = welch_psd(x)
    mask = (f >= fmin) & (f <= fmax)
    if not mask.any():
        return float('nan'), float('nan')
    i = np.argmax(p[mask])
    f_peak = f[mask][i]
    return 1 / f_peak if f_peak > 0 else float('nan'), f_peak


def autocorr_time(x, max_lag=300):
    """Первый лаг, где ACF падает ниже 1/e (сек)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    var = x.var()
    if var == 0:
        return float('nan')
    for lag in range(1, max_lag):
        ac = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        if ac < 1 / np.e:
            return float(lag)
    return float(max_lag)


def noise_std(x, window=9):
    """Оценка шума: std остатка после скользящего среднего (высокие частоты)."""
    s = pd.Series(x).rolling(window, center=True, min_periods=1).mean().values
    return float(np.std(np.asarray(x, float) - s))


def cross_corr_lag(a, b, max_lag=60):
    """Лаг (сек) максимума взаимной корреляции и его величина."""
    a = np.asarray(a, float); a = a - a.mean()
    b = np.asarray(b, float); b = b - b.mean()
    best, best_lag = 0.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.corrcoef(a[lag:], b[:len(b) - lag])[0, 1]
        else:
            c = np.corrcoef(a[:len(a) + lag], b[-lag:])[0, 1]
        if np.isfinite(c) and c > best:
            best, best_lag = c, lag
    return best_lag, best


def main():
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        try:
            _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    df = load_data(PROJECT_ROOT / 'data' / 'raw' / 'your_data.csv')
    out = PROJECT_ROOT / 'results' / 'analysis_real_data'
    out.mkdir(parents=True, exist_ok=True)

    lines = ['=' * 76, 'ПОРТРЕТ РЕАЛЬНЫХ ДАННЫХ (data/raw/your_data.csv)',
             f'Строк: {len(df)} (~{len(df) / 3600:.2f} ч @ 1 Гц)', '=' * 76, '']

    key_cols = [
        'Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)',
        'Velocity.Rolling(°/мин)', 'Velocity.Pitching(°/мин)',
        'Wave.Highest(метры)', 'Swell(метры)', 'Wave.direction(градусы)',
        'Wind.direction(градусы)', 'Rudder State(градусы)', 'Rudder Order(градусы)',
        'ROT(°/мин)', 'SOG(узлы)', 'Course(градусы)', 'RPM(Обороты в минуту)',
    ]

    lines.append('--- Базовые статистики (физические единицы) ---')
    stat_rows = []
    for c in key_cols:
        x = df[c].values.astype(float)
        row = {'column': c, 'mean': np.mean(x), 'std': np.std(x),
               'min': np.min(x), 'max': np.max(x),
               'p05': np.percentile(x, 5), 'p95': np.percentile(x, 95),
               'autocorr_s': autocorr_time(x), 'noise_std': noise_std(x)}
        stat_rows.append(row)
        lines.append(f"{c:32s} mean={row['mean']:8.3f} std={row['std']:7.3f} "
                     f"[{row['min']:8.2f};{row['max']:8.2f}] "
                     f"τ_corr={row['autocorr_s']:5.0f}s noise={row['noise_std']:6.4f}")
    pd.DataFrame(stat_rows).to_csv(out / 'stats.csv', index=False)

    lines += ['', '--- Спектральные пики качки (период, с) ---']
    spec = {}
    for c in ['Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)']:
        T, f_peak = peak_period(df[c].values)
        spec[c] = {'period_s': T, 'freq_hz': f_peak}
        lines.append(f"{c:32s} T = {T:6.1f} с (f = {f_peak:.4f} Гц)")

    lines += ['', '--- Быстрые изменения: приращения за 1 с ---']
    for c in ['Roll(градусы)', 'Pitch(градусы)', 'Rudder State(градусы)', 'ROT(°/мин)']:
        d = np.diff(df[c].values.astype(float))
        lines.append(f"{c:32s} Δ1s: std={np.std(d):7.4f}  p99|Δ|={np.percentile(np.abs(d), 99):7.4f}  "
                     f"max|Δ|={np.max(np.abs(d)):7.3f}")

    lines += ['', '--- Кросс-корреляции (лаг в секундах, величина) ---']
    pairs = [
        ('Roll(градусы)', 'Wave.Highest(метры)'),
        ('Roll(градусы)', 'Wave.speed(узлы)'),
        ('Vertical(Метр)', 'Wave.Highest(метры)'),
        ('ROT(°/мин)', 'Rudder State(градусы)'),
        ('SOG(узлы)', 'RPM(Обороты в минуту)'),
    ]
    for a, b in pairs:
        lag, cc = cross_corr_lag(df[a].values, df[b].values)
        lines.append(f"{a:26s} ↔ {b:26s} lag={lag:+4d}s  corr={cc:.3f}")

    lines += ['', '--- Режимы волнения (доли времени) ---']
    wh = df['Wave.Highest(метры)'].values.astype(float)
    for name, lo, hi in [('calm <0.5м', 0, 0.5), ('moderate 0.5-1.5м', 0.5, 1.5),
                         ('rough 1.5-2.5м', 1.5, 2.5), ('severe >2.5м', 2.5, 1e9)]:
        frac = np.mean((wh >= lo) & (wh < hi))
        lines.append(f"  {name:22s} {frac * 100:5.1f}%")

    rel = (df['Wave.direction(градусы)'].astype(float) - df['Course(градусы)'].astype(float)
           + 180) % 360 - 180
    arel = np.abs(rel)
    for name, lo, hi in [('following <30°', 0, 30), ('quartering 30-60°', 30, 60),
                         ('beam 60-120°', 60, 120), ('quartering 120-150°', 120, 150),
                         ('head >150°', 150, 181)]:
        frac = np.mean((arel >= lo) & (arel < hi))
        lines.append(f"  {name:22s} {frac * 100:5.1f}%")

    lines += ['', '--- Руль: статистика манёвров ---']
    ro = df['Rudder Order(градусы)'].values.astype(float)
    changes = np.abs(np.diff(ro)) > 0.5
    idx = np.nonzero(changes)[0]
    if len(idx) > 1:
        gaps = np.diff(idx)
        lines.append(f"  Число перекладок руля: {len(idx)} (в среднем каждые {np.mean(gaps):.0f} с, "
                     f"медиана {np.median(gaps):.0f} с)")
    lines.append(f"  Доля времени с |Rudder Order|>1°: {np.mean(np.abs(ro) > 1) * 100:.1f}%")

    report = '\n'.join(lines)
    print(report)
    (out / 'report.txt').write_text(report, encoding='utf-8')

    # Машинные параметры для генератора
    gen_params = {
        'n_rows': int(len(df)),
        'specs': {c: {'mean': float(np.mean(df[c].values.astype(float))),
                      'std': float(np.std(df[c].values.astype(float)))} for c in key_cols},
        'spectra': {c: {'period_s': spec[c]['period_s'], 'freq_hz': spec[c]['freq_hz']}
                    for c in spec},
    }
    import json
    (out / 'generator_params.json').write_text(
        json.dumps(gen_params, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n✓ Сохранено: {out}")


if __name__ == '__main__':
    main()
