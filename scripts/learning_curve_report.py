"""
Отчёт по исследованию «объём обучающей выборки → качество прогноза».

Собирает все прогоны с пометками E1/E2/E3 в notes (см. RESEARCH_LOG.md §3),
строит сводную таблицу (физические метрики, mean±std по сидам), аппроксимирует
learning curve степенным законом MAE(n) = a·n^(−b) + c (Hestness 2017) и
строит график.

Запуск: python scripts/learning_curve_report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import registry as reg


def collect_runs(prefixes=('E1', 'E2', 'E3')):
    rows = []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None:
            continue
        notes = m.get('hypothesis', '') or ''
        if not notes.startswith(prefixes):
            continue
        res = m.get('results', {})
        phys = res.get('physical', {})
        if not phys:
            continue
        overall = phys.get('overall', {})
        data = m.get('data', {})
        # фактический объём train (строк)
        n_train = data.get('rows_train', 0)
        rows.append({
            'run_id': m['run_id'],
            'exp': notes.split()[0],
            'notes': notes,
            'seed': m.get('training', {}).get('seed'),
            'n_train_rows': n_train,
            'mae': overall.get('mae'),
            'rmse': overall.get('rmse'),
            'r2': overall.get('r2'),
            'skill': (res.get('skill_vs_persistence') or {}).get('overall'),
            'per_target_mae': {k: v['mae'] for k, v in phys.get('per_target', {}).items()},
        })
    return pd.DataFrame(rows)


def power_law_fit(n, y):
    """MAE(n) = a*n^(-b) + c, подбор через scipy.curve_fit с разумными границами."""
    from scipy.optimize import curve_fit

    def f(n, a, b, c):
        return a * np.power(n, -b, dtype=float) + c

    try:
        popt, _ = curve_fit(f, np.asarray(n, float), np.asarray(y, float),
                            p0=[float(y.max()) * 0.5, 0.3, float(y.min()) * 0.8],
                            bounds=([1e-6, 0.02, 0.0],
                                    [1e4, 3.0, float(y.min()) * 0.99]),
                            maxfev=50000)
        pred = f(np.asarray(n, float), *popt)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return popt, r2
    except Exception:
        return None, float('nan')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    import argparse
    parser = argparse.ArgumentParser(description='Learning curve report')
    parser.add_argument('--prefix', type=str, default='E1',
                        help='Префикс notes серии (E1 — старый пайплайн E1/E2/E3; '
                             'F1 — новый пайплайн v2-канон, Ф0.5/Ф1)')
    prefix = parser.parse_args().prefix

    df = collect_runs(prefixes=(prefix,) if prefix != 'E1' else ('E1', 'E2', 'E3'))
    if df.empty:
        print(f'Нет прогонов с префиксом {prefix}')
        return

    new_pipeline = prefix != 'E1'
    out = PROJECT_ROOT / (f'results/learning_curve_{prefix.lower()}' if new_pipeline
                          else 'results/learning_curve')
    out.mkdir(parents=True, exist_ok=True)

    region_note = ('ПОЛНАЯ запись (Ф0.5): спокойный 0–4000 + активный 4000+, '
                   'minimal_prediction, fixed scaler, test фиксирован' if new_pipeline
                   else 'активный регион записи: строки 4000+, fixed scaler, test фиксирован')
    lines = ['=' * 84,
             f'LEARNING CURVE ({prefix}): качество vs объём обучающей выборки',
             f'({region_note})',
             '=' * 84, '']

    e1 = df[df['exp'] == prefix]
    agg = e1.groupby('n_train_rows').agg(
        mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
        r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
        skill_mean=('skill', 'mean'), n_runs=('mae', 'count')).reset_index()
    agg = agg.sort_values('n_train_rows')

    lines.append(f"{'train, строк':>12s} {'MAE mean':>9s} {'±std':>6s} {'R² mean':>8s} {'±std':>6s} "
                 f"{'skill':>6s} {'сидов':>5s}")
    lines.append('-' * 60)
    for _, r in agg.iterrows():
        lines.append(f"{int(r['n_train_rows']):12d} {r['mae_mean']:9.3f} "
                     f"{(r['mae_std'] if np.isfinite(r['mae_std']) else 0):6.3f} "
                     f"{r['r2_mean']:8.3f} "
                     f"{(r['r2_std'] if np.isfinite(r['r2_std']) else 0):6.3f} "
                     f"{r['skill_mean']:6.3f} {int(r['n_runs']):5d}")

    # Степенной закон
    popt, fit_r2 = power_law_fit(agg['n_train_rows'].values, agg['mae_mean'].values)
    if popt is not None:
        a, b, c = popt
        lines += ['', f'Аппроксимация MAE(n) = a·n^(-b) + c:',
                  f'  a = {a:.3f}, b = {b:.3f}, c = {c:.3f}   (R² аппроксимации = {fit_r2:.4f})',
                  f'  Интерпретация: при увеличении train в 2 раза MAE падает на '
                  f'{(1 - 2 ** -b) * 100:.1f}%; асимптота (потолок данных) MAE ≈ {c:.2f}']

    # E2 / E3 (только старый пайплайн; Ф1 — синтетика отдельной серией)
    e3 = df[df['exp'] == 'E3'].copy() if 'E3' in df['exp'].values else e1.iloc[0:0].copy()
    lines += ['', '--- E2: обучение ТОЛЬКО на синтетике ---']
    for _, r in df[df['exp'] == 'E2'].iterrows():
        lines.append(f"  {r['notes']:28s} MAE={r['mae']:.3f}  R²={r['r2']:.3f}  skill={r['skill']:+.3f}")

    lines += ['', '--- E3: немного реального + синтетика (аугментация) ---']
    e3['syn_rows'] = e3['notes'].str.extract(r'syn(\d+)').astype(float)
    e3 = e3.sort_values('syn_rows')
    for _, r in e3.iterrows():
        lines.append(f"  {r['notes']:28s} MAE={r['mae']:.3f}  R²={r['r2']:.3f}  skill={r['skill']:+.3f}")

    base1 = e1[e1['n_train_rows'] == e1['n_train_rows'].min()]['mae'].mean()
    if new_pipeline:
        report = '\n'.join(lines)
        print(report)
        (out / 'learning_curve_report.txt').write_text(report, encoding='utf-8')
        df.to_csv(out / 'all_f1_runs.csv', index=False)
        agg.to_csv(out / 'f1_aggregated.csv', index=False)
        print(f'\n✓ Сохранено: {out}')
        return
    basefull = e1[e1['n_train_rows'] == e1['n_train_rows'].max()]['mae'].mean()
    if len(e3):
        best_aug = e3.loc[e3['mae'].idxmin()]
        lines += ['',
                  f'Сравнение: 700 реальных строк alone          MAE = {base1:.3f}',
                  f'           700 реальных + {int(best_aug["syn_rows"])} синтетики  MAE = {best_aug["mae"]:.3f}',
                  f'           {int(e1["n_train_rows"].max())} реальных (полный train)      MAE = {basefull:.3f}']

    report = '\n'.join(lines)
    print(report)
    (out / 'learning_curve_report.txt').write_text(report, encoding='utf-8')
    df.to_csv(out / 'all_study_runs.csv', index=False)  # E-пайплайн (F1 уходит раньше)
    agg.to_csv(out / 'e1_aggregated.csv', index=False)

    # График
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    xs = agg['n_train_rows'].values
    ax.errorbar(xs, agg['mae_mean'], yerr=agg['mae_std'].fillna(0), fmt='o-',
                color='tab:blue', capsize=4, label=f'Только реальные данные ({prefix}, mean±std)')
    if popt is not None:
        xs_fit = np.linspace(xs.min(), xs.max(), 200)
        ax.plot(xs_fit, popt[0] * xs_fit ** (-popt[1]) + popt[2], '--',
                color='tab:blue', alpha=0.5,
                label=f'Степенной закон a·n^-b+c (b={popt[1]:.2f})')
    for _, r in e3.iterrows():
        ax.scatter(700 + r['syn_rows'], r['mae'], marker='s', color='tab:orange', s=60)
    if len(e3):
        ax.scatter([], [], marker='s', color='tab:orange',
                   label='700 реальных + синтетика (E3)')
    for _, r in df[df['exp'] == 'E2'].iterrows():
        ax.scatter(r['n_train_rows'], r['mae'], marker='^', color='tab:green', s=70)
    ax.scatter([], [], marker='^', color='tab:green', label='Только синтетика (E2)')

    ax.set_xlabel('Объём обучающей выборки, строк (1 Гц)')
    ax.set_ylabel('Test MAE, физические единицы (8 целей)')
    ax.set_title('Качество прогноза качки vs объём обучающей выборки\n'
                 f'(test фиксирован, {region_note})')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / 'learning_curve.png', dpi=150)
    print(f'\n✓ Сохранено: {out}')


if __name__ == '__main__':
    main()
