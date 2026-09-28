"""
Пункт 3 (Tasks fm user): анализ результатов sweep-эксперимента.

Собирает прогоны с notes 'SWEEP <вариант> K=<k>' (+ базовые 'E1 segsK seed42'),
строит:
  1. сводную таблицу MAE/R²/skill по (вариант × размер выборки);
  2. эффект относительно baseline (ΔMAE) на «малом» и «большом» наборе данных;
  3. per-target MAE таблицу;
  4. график (сгруппированные бары MAE по вариантам, отдельно для K=2 и K=9).

Запуск: python scripts/sweep_report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import registry as reg


def collect():
    rows = []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None:
            continue
        notes = m.get('hypothesis') or ''
        phys = m.get('results', {}).get('physical', {})
        if not phys or not notes.startswith(('SWEEP', 'E1 ')):
            continue
        variant, k = None, None
        seed = (m.get('training') or {}).get('seed') or 42
        if notes.startswith('SWEEP'):
            parts = notes.split()
            variant, k = parts[1], int(parts[2].split('=')[1])
        else:
            # E1 segsK seedN — базовые точки (все сиды, для мультисида)
            variant, k = 'baseline', int(notes.split('segs')[1].split()[0])
        overall = phys.get('overall', {})
        rows.append({
            'variant': variant, 'K': k, 'seed': seed, 'run_id': m['run_id'],
            'n_train': m.get('data', {}).get('rows_train'),
            'mae': overall.get('mae'), 'rmse': overall.get('rmse'), 'r2': overall.get('r2'),
            'skill': (m.get('results', {}).get('skill_vs_persistence') or {}).get('overall'),
            'per_target': {t: v['mae'] for t, v in phys.get('per_target', {}).items()},
        })
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    df = collect()
    if df.empty:
        print('Нет прогонов SWEEP/E1')
        return
    # мультисид: дедуп по (вариант, K, сид), затем mean±std по сидам
    df = df.drop_duplicates(subset=['variant', 'K', 'seed'], keep='last')
    df['per_target'] = df['per_target'].apply(lambda d: d if isinstance(d, dict) else {})
    df = df.groupby(['variant', 'K']).agg(
        n_train=('n_train', 'first'), mae=('mae', 'mean'), mae_std=('mae', 'std'),
        r2=('r2', 'mean'), r2_std=('r2', 'std'), skill=('skill', 'mean'),
        seeds=('seed', lambda s: sorted(s)),
        run_id=('run_id', 'last'),
        per_target=('per_target', lambda ss: pd.DataFrame(
            [p for p in ss if p]).mean().to_dict() if any(p for p in ss) else {}),
    ).reset_index()

    out = PROJECT_ROOT / 'results' / 'sweep'
    out.mkdir(parents=True, exist_ok=True)

    lines = ['=' * 88,
             'SWEEP: влияние коэффициентов на качество прогноза (физические единицы)',
             'test фиксирован; активный регион; fixed scaler; мультисид mean±std',
             '=' * 88, '']

    table_rows, per_target_rows = [], []
    for k in sorted(df['K'].unique()):
        sub = df[df['K'] == k]
        base = sub[sub['variant'] == 'baseline']
        base_mae = float(base['mae'].iloc[0]) if len(base) else np.nan
        lines.append(f'--- Размер выборки K={k} '
                     f'({int(sub["n_train"].iloc[0])} строк) | baseline MAE = {base_mae:.3f} ---')
        lines.append(f'{"вариант":12s} {"MAE±std":>15s} {"ΔMAE":>8s} {"R²":>7s} {"skill":>7s} {"сиды":>12s}')
        for _, r in sub.sort_values('mae').iterrows():
            delta = r['mae'] - base_mae if np.isfinite(base_mae) else np.nan
            mark = ' ←' if r['variant'] == 'baseline' else ('  ✓' if delta < -0.05 else
                   ('  ✗' if delta > 0.05 else ''))
            mae_s = f'{r["mae"]:.3f}±{r["mae_std"]:.2f}' if np.isfinite(r['mae_std']) else f'{r["mae"]:.3f}'
            seeds_s = ','.join(str(s) for s in r['seeds'])
            lines.append(f'{r["variant"]:12s} {mae_s:>15s} {delta:+8.3f} '
                         f'{r["r2"]:7.3f} {r["skill"]:+7.3f} {seeds_s:>12s}{mark}')
            table_rows.append({'K': k, 'variant': r['variant'], 'mae': r['mae'],
                               'mae_std': r['mae_std'], 'delta_mae': delta, 'r2': r['r2'],
                               'skill': r['skill'], 'n_seeds': len(r['seeds']),
                               'run_id': r['run_id']})
            pt = {'variant': r['variant'], 'K': k, **r['per_target']}
            per_target_rows.append(pt)
        lines.append('')

    report = '\n'.join(lines)
    print(report)
    (out / 'sweep_report.txt').write_text(report, encoding='utf-8')
    pd.DataFrame(table_rows).to_csv(out / 'sweep_table.csv', index=False)
    pd.DataFrame(per_target_rows).to_csv(out / 'sweep_per_target.csv', index=False)

    # График: MAE по вариантам, панели для K=2 и K=9
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ks = sorted(df['K'].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(7 * len(ks), 5.5), sharey=False)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        sub = df[df['K'] == k].sort_values('mae')
        colors = ['tab:orange' if v == 'baseline' else 'tab:blue' for v in sub['variant']]
        ax.barh(sub['variant'], sub['mae'], color=colors)
        ax.set_xlabel('Test MAE (физ. ед.)')
        ax.set_title(f'K={k} сегментов ({int(sub["n_train"].iloc[0])} строк)')
        ax.grid(alpha=0.3, axis='x')
        for i, (_, r) in enumerate(sub.iterrows()):
            ax.text(r['mae'], i, f" {r['mae']:.2f}", va='center', fontsize=9)
    fig.suptitle('Влияние коэффициентов на качество прогноза качки', fontsize=13)
    plt.tight_layout()
    plt.savefig(out / 'sweep_chart.png', dpi=150)
    print(f'✓ Сохранено: {out}')


if __name__ == '__main__':
    main()
