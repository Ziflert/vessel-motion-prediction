"""
A3 (IDEAS.md): отчёт по матрице горизонтов прогноза 10/20/30 — кривая «ошибка от упреждения».

Собирает прогоны 'HORIZON <вариант> H=<h> seed<s>' + легаси 'SWEEP <вариант> K=9 seed<s>'
(те же условия при H=20, мультисид A1). Строит (mean±std по сидам):
  1. кривую «MAE от упреждения» по шагам 1..H для каждого (вариант × H) + persistence;
  2. сводную таблицу overall MAE/R²/skill по (вариант × H);
  3. фигуры fig6 (кривая от упреждения — ключевой график для СППР) и
     fig7 (цена горизонта: overall MAE и skill по H) — PDF+PNG 300 dpi,
     дублируются в results/article/figs/.

Запуск: .venv/Scripts/python.exe scripts/horizon_report.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import registry as reg

OUT = PROJECT_ROOT / 'results' / 'horizons'
FIGS = PROJECT_ROOT / 'results' / 'article' / 'figs'

# --- журнальный стиль (как article_figures.py) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

VARIANT_LABEL_OLD = {
    'baseline': 'baseline (MSE + 0.5·Huber + 0.1·smooth, Roll=2)',
    'smooth0': 'smooth0 (без гладкости, MSE + 0.5·Huber, Roll=2)',
}
VARIANT_LABEL_NEW = {
    'baseline': 'baseline (MSE + 0.5·Huber + 0.1·smooth, Roll=2)',
    'roll_w4': 'roll_w4 (лучший свип Ф3: Roll=4)',
}
H_COLOR = {10: '#1f77b4', 20: '#d62728', 30: '#2ca02c'}
H_MARKER = {10: 'o', 20: 's', 30: '^'}

TARGETS_SHORT = {
    'Pitch(градусы)': 'Pitch, °',
    'Roll(градусы)': 'Roll, °',
    'Vertical(Метр)': 'Vertical, м',
    'Velocity.Rolling(°/мин)': 'Vel.Roll, °/мин',
    'ROT(°/мин)': 'ROT, °/мин',
    'SOG(узлы)': 'SOG, уз',
}


def save(fig, out_dir, name, also_article=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png')
    fig.savefig(out_dir / f'{name}.pdf')
    if also_article:
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / f'{name}.png')
        fig.savefig(FIGS / f'{name}.pdf')
    plt.close(fig)
    print(f'  ✓ {name}.png/.pdf' + (' (+ article/figs)' if also_article else ''))


def collect(prefix='HORIZON'):
    rows = []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None:
            continue
        notes = m.get('hypothesis') or ''
        variant, h = None, None
        if notes.startswith(prefix):
            parts = notes.split()
            variant, h = parts[1], int(parts[2].split('=')[1])
        elif prefix == 'HORIZON' and notes.startswith('SWEEP'):
            parts = notes.split()
            k = int(parts[2].split('=')[1])
            if k == 9:  # полная выборка = те же условия при H=20 (мультисид A1)
                variant, h = parts[1], 20
        elif prefix == 'F4' and notes.startswith('F3 '):
            # H=20 переиспользуется из Ф3 (только варианты Ф4: baseline, roll_w4)
            parts = notes.split()
            k = int(parts[2].split('=')[1])
            if k == 9 and parts[1] in ('baseline', 'roll_w4'):
                variant, h = parts[1], 20
        elif prefix == 'F4' and notes.startswith('F1 minimal k9'):
            # H=20 baseline переиспользуется из Ф1 (3 сида)
            variant, h = 'baseline', 20
        if variant is None:
            continue
        res = m.get('results', {})
        phys = res.get('physical', {})
        if not phys:
            continue
        seed = (m.get('training') or {}).get('seed') or 42
        rows.append({
            'variant': variant, 'H': h, 'seed': seed, 'run_id': m['run_id'],
            'mae': phys.get('overall', {}).get('mae'),
            'rmse': phys.get('overall', {}).get('rmse'),
            'r2': phys.get('overall', {}).get('r2'),
            'skill': (res.get('skill_vs_persistence') or {}).get('overall'),
            'per_lead': phys.get('per_horizon_mae'),
            'persist_lead': (res.get('baselines', {}).get('persistence', {}) or {}).get('per_horizon_mae'),
            'per_lead_target': phys.get('per_horizon_per_target_mae'),
        })
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    import argparse
    parser = argparse.ArgumentParser(description='Horizon matrix report')
    parser.add_argument('--prefix', type=str, default='HORIZON',
                        help='Префикс notes (HORIZON — старый пайплайн; F4 — новый v2-канон)')
    prefix = parser.parse_args().prefix

    df = collect(prefix)
    if df.empty:
        print(f'Нет прогонов {prefix}/легаси')
        return

    vlabel = VARIANT_LABEL_OLD if prefix == 'HORIZON' else VARIANT_LABEL_NEW
    if prefix != 'HORIZON':
        OUT = PROJECT_ROOT / f'results/horizons_{prefix.lower()}'

    # дедуп (та же политика, что sweep_report: keep='last' — согласовно с агрегатом A1)
    df = df.drop_duplicates(subset=['variant', 'H', 'seed'], keep='last')

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- агрегат overall по (вариант × H) ----
    agg = df.groupby(['variant', 'H']).agg(
        mae=('mae', 'mean'), mae_std=('mae', 'std'),
        r2=('r2', 'mean'), r2_std=('r2', 'std'),
        skill=('skill', 'mean'), skill_std=('skill', 'std'),
        seeds=('seed', lambda s: sorted(s)),
        run_ids=('run_id', lambda s: ';'.join(sorted(s))),
    ).reset_index()
    agg.to_csv(OUT / 'horizon_overall.csv', index=False)

    # ---- кривая MAE от упреждения (long CSV) ----
    curve_rows = []
    for _, r in agg.iterrows():
        sub = df[(df['variant'] == r['variant']) & (df['H'] == r['H'])]
        leads = np.arange(1, int(r['H']) + 1)
        arr = np.array([p for p in sub['per_lead'] if p], dtype=float)  # [seeds, H]
        parr = np.array([p for p in sub['persist_lead'] if p], dtype=float)
        for i, step in enumerate(leads):
            curve_rows.append({
                'variant': r['variant'], 'H': int(r['H']), 'lead_step': int(step),
                'lead_s': int(step),  # 1 Гц, prediction_step=1
                'mae_mean': arr[:, i].mean(), 'mae_std': arr[:, i].std(ddof=1) if len(arr) > 1 else 0.0,
                'persist_mean': parr[:, i].mean(), 'persist_std': parr[:, i].std(ddof=1) if len(parr) > 1 else 0.0,
                'skill_mean': 1 - arr[:, i].mean() / parr[:, i].mean() if parr[:, i].mean() > 0 else np.nan,
            })
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(OUT / 'horizon_curve.csv', index=False)

    # ================= FIG6: кривая «ошибка от упреждения» =================
    variants = list(vlabel.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(9.0, 3.6), sharey=True)
    for ax, v in zip(axes, variants):
        for h in [10, 20, 30]:
            sub = curve[(curve['variant'] == v) & (curve['H'] == h)]
            if sub.empty:
                continue
            c, mk = H_COLOR[h], H_MARKER[h]
            ax.plot(sub['lead_s'], sub['mae_mean'], color=c, marker=mk, ms=3.5,
                    lw=1.4, label=f'модель H={h} (шагов)')
            ax.fill_between(sub['lead_s'], sub['mae_mean'] - sub['mae_std'],
                            sub['mae_mean'] + sub['mae_std'], color=c, alpha=0.13)
            ax.axvline(h, color=c, ls=':', lw=0.8, alpha=0.6)
        # persistence (общий бейзлайн, фиксированный тест → одинаков у всех сидов)
        sub0 = curve[(curve['variant'] == v) & (curve['H'] == curve['H'].max())]
        if not sub0.empty:
            ax.plot(sub0['lead_s'], sub0['persist_mean'], color='#7f7f7f', ls='--', lw=1.2,
                    label='persistence (наивный)')
        ax.set_xlabel('Упреждение, с (1 Гц)')
        ax.set_title(vlabel[v], fontsize=9)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.legend(loc='upper left', framealpha=0.9)
    axes[0].set_ylabel('MAE (физ. ед., все цели)')
    fig.suptitle('Кривая «ошибка от упреждения»: матрица горизонтов 10/20/30 '
                 '(K=9 ≈ 5949 строк, mean±std по 3 сидам)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, OUT, 'fig6_horizon_curve', also_article=(prefix == 'HORIZON'))

    # ================= FIG7: цена горизонта (overall) =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    x = np.arange(len([10, 20, 30]))
    w = 0.38
    for j, v in enumerate(variants):
        sub = agg[agg['variant'] == v].set_index('H').reindex([10, 20, 30])
        offs = (j - 0.5) * w
        ax1.bar(x + offs, sub['mae'], w, yerr=sub['mae_std'], capsize=3,
                color=['#1f77b4', '#d62728', '#2ca02c'][j % 3], alpha=0.85,
                label=vlabel[v].split('(')[0].strip())
        for xi, (mae, sk) in zip(x + offs, zip(sub['mae'], sub['skill'])):
            if np.isfinite(sk):
                ax1.annotate(f'S={sk:.2f}', (xi, mae), textcoords='offset points',
                             xytext=(0, 4 + (3 if j else 0)), ha='center', fontsize=7)
        ax2.bar(x + offs, sub['skill'], w, yerr=sub['skill_std'], capsize=3,
                color=['#1f77b4', '#d62728', '#2ca02c'][j % 3], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(['H=10', 'H=20', 'H=30'])
    ax1.set_ylabel('MAE (физ. ед., усреднено по горизонту)')
    ax1.set_xlabel('Горизонт прогноза')
    ax1.legend()
    ax2.set_xticks(x); ax2.set_xticklabels(['H=10', 'H=20', 'H=30'])
    ax2.set_ylabel('Skill vs persistence')
    ax2.set_xlabel('Горизонт прогноза')
    ax2.axhline(0, color='#7f7f7f', lw=0.8)
    fig.suptitle('Цена горизонта: overall-качество на 10/20/30 шагов упреждения '
                 f'({"Ф4, минимальный пайплайн" if prefix != "HORIZON" else "K=9"}, mean±std по 3 сидам)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, OUT, 'fig7_horizon_overall', also_article=(prefix == 'HORIZON'))

    # ================= Текстовый отчёт =================
    conditions_note = (
        'ПОЛНАЯ запись (Ф0.5), minimal_prediction, fixed scaler, subset-seed 123, '
        '≤100 эпох; 1 Гц (1 шаг упреждения = 1 с); mean±std по 3 сидам {42,43,44}. '
        'H=20 переиспользованы из Ф1/Ф3.' if prefix != 'HORIZON' else
        'K=9 сегментов (~5949 строк), skip-rows 4000, fixed scaler, subset-seed 123, '
        '≤100 эпох; 1 Гц (1 шаг упреждения = 1 с); mean±std по 3 сидам {42,43,44}. '
        'Прогоны H=20 переиспользованы из мультисида A1 (SWEEP <v> K=9 seed<s>).')
    lines = [
        '=' * 78,
        'A3 — МАТРИЦА ГОРИЗОНТОВ ПРОГНОЗА 10/20/30 (кривая «ошибка от упреждения»)',
        '=' * 78, '',
        f'Условия: {conditions_note}', '',
    ]
    for v in variants:
        lines += [f'--- {vlabel[v]} ---']
        for _, r in agg[agg['variant'] == v].iterrows():
            lines.append(
                f"H={int(r['H']):>2}: MAE {r['mae']:.2f} ± {r['mae_std']:.2f}   "
                f"R² {r['r2']:.3f} ± {r['r2_std']:.3f}   Skill {r['skill']:+.3f} ± {r['skill_std']:.3f}"
            )
        lines.append('')
    lines += ['--- Кривая MAE от упреждения (mean±std по сидам; физ. ед., все цели) ---']
    for v in variants:
        for h in [10, 20, 30]:
            sub = curve[(curve['variant'] == v) & (curve['H'] == h)]
            if sub.empty:
                continue
            cells = '  '.join(f"{row.mae_mean:.2f}±{row.mae_std:.2f}" for row in sub.itertuples())
            lines.append(f'{v} H={h:>2}: {cells}')
    lines.append('')
    lines += ['--- MAE ключевых целей на максимальном шаге упреждения (mean по сидам) ---']
    for v in variants:
        for h in [10, 20, 30]:
            sub = df[(df['variant'] == v) & (df['H'] == h)]
            plts = [p for p in sub['per_lead_target'] if p]
            if not plts:
                continue
            keys = list(plts[0].keys())
            parts = []
            for t in keys:
                vals = [p[t] for p in plts if t in p]
                if vals:
                    parts.append(f"{TARGETS_SHORT.get(t, t)}={np.mean(vals):.2f}")
            lines.append(f'{v} H={h}: ' + '  '.join(parts))
    lines += ['', '=' * 78,
              'Интерпретация для СППР (CONCEPT §4.2): горизонт, на который можно верить прогнозу,',
              'определяется ростом MAE по упреждению; бейзлайн persistence — нижняя планка полезности.',
              'Артефакты: results/horizons/ (horizon_overall.csv, horizon_curve.csv, fig6, fig7);',
              'фигуры продублированы в results/article/figs/.', '=' * 78]
    with open(OUT / 'horizon_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'  ✓ horizon_report.txt')

    # краткая сводка в консоль
    print('\n--- overall (mean±std по сидам) ---')
    for v in variants:
        for _, r in agg[agg['variant'] == v].iterrows():
            print(f"{v:>9} H={int(r['H']):>2}: MAE {r['mae']:.2f}±{r['mae_std']:.2f}  "
                  f"R² {r['r2']:.3f}±{r['r2_std']:.3f}  Skill {r['skill']:+.3f}±{r['skill_std']:.3f}")


if __name__ == '__main__':
    main()
