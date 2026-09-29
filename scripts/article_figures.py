"""
A2 (IDEAS.md): журнальный набор рисунков статьи.

Строит 5 фигур из ГОТОВЫХ данных (реестр + results/*.csv), PDF+PNG 300 dpi,
в results/article/figs/:
  fig1 — learning curve + степенной закон (E1, mean±std);
  fig2 — ΔMAE sweep-вариантов по объёму выборки (мультисид, mean±std);
  fig3 — per-target сравнение: реал 700 / реал 5949 / синтетика 30k / аугмент 700+17.5k;
  fig4 — MAE по режимам волнения;
  fig5 — дрейф периода качки по мере развития шторма (по сырой записи).

Запуск: .venv/Scripts/python.exe scripts/article_figures.py
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import registry as reg

import argparse

OUT = PROJECT_ROOT / 'results' / 'article' / 'figs'
OUT_NEW = PROJECT_ROOT / 'results' / 'article' / 'figs_v2'

# --- журнальный стиль ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

C_MAIN = '#1f77b4'
C_ACCENT = '#d62728'
C_GREY = '#7f7f7f'


def save(fig, name):
    out_dir = OUT if _use_old else OUT_NEW
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png')
    fig.savefig(out_dir / f'{name}.pdf')
    plt.close(fig)
    print(f'  ✓ {name}.png/.pdf -> {out_dir.name}/')


_use_old = True  # переключается в main (--new-pipeline)


# ----------------------------------------------------------------------------
def fig1_learning_curve(new=False):
    src = (PROJECT_ROOT / 'results/learning_curve/e1_aggregated.csv' if not new
           else PROJECT_ROOT / 'results/learning_curve_f1/f1_aggregated.csv')
    df = pd.read_csv(src)
    n = df['n_train_rows'].values.astype(float)
    mae, std = df['mae_mean'].values, df['mae_std'].values
    label = ('E1: mean ± std (2 сида)' if not new
             else 'Ф1: mean ± std (3 сида, минимальный пайплайн, ПОЛНАЯ запись)')

    # степенной закон MAE = a * n^b (fit по логам, по средним)
    b, loga = np.polyfit(np.log(n), np.log(mae), 1)
    a = np.exp(loga)
    nfit = np.linspace(n.min() * 0.8, n.max() * 1.6, 200)

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    ax.errorbar(n, mae, yerr=std, fmt='o', color=C_MAIN, capsize=3,
                label=label)
    ax.plot(nfit, a * nfit ** b, '--', color=C_ACCENT,
            label=f'степенной закон: {a:.1f}·n^({b:.2f})')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Объём обучающей выборки n, строк (сек при 1 Гц)')
    ax.set_ylabel('Test MAE, физ. ед. (агрег. по 8 целям)')
    ax.set_title('а) Кривая обучения: ошибка от объёма базы')
    ax.legend(title='MAE — меньше лучше', title_fontsize=8)
    for x, y in zip(n, mae):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(6, -10), fontsize=8)
    save(fig, 'fig1_learning_curve')
    return a, b


# ----------------------------------------------------------------------------
def sweep_collect(prefix='SWEEP'):
    """Мультисид-агрегация <prefix> + baseline из реестра (mean±std по сидам)."""
    rows = []
    for run in reg.list_runs():
        m = run['manifest']
        if not m or not (m.get('hypothesis') or '').startswith((prefix, 'E1 ')):
            continue
        notes = m['hypothesis']
        phys = m.get('results', {}).get('physical', {})
        overall = phys.get('overall', {})
        mae = overall.get('mae')
        if mae is None:
            continue
        seed = m.get('training', {}).get('seed') or 42
        if notes.startswith(prefix):
            parts = notes.split()
            variant, k = parts[1], int(parts[2].split('=')[1])
        else:
            variant, k = 'baseline', int(notes.split('segs')[1].split()[0])
        rows.append({'variant': variant, 'K': k, 'seed': seed, 'mae': mae})
    df = pd.DataFrame(rows).drop_duplicates(subset=['variant', 'K', 'seed'])
    agg = df.groupby(['variant', 'K'])['mae'].agg(['mean', 'std', 'count']).reset_index()
    return agg


def fig2_sweep(new=False):
    agg = sweep_collect('F3' if new else 'SWEEP')
    order = ['smooth0', 'huber0', 'roll_w4', 'lr2e-4', 'baseline', 'no-attn', 'roll_w0.5']
    ks = sorted(agg['K'].unique())
    title_let = {'2': 'а', '9': 'б'}

    # вертикальная компоновка: K=2 сверху, K=9 снизу; бары горизонтальные,
    # подписи вариантов — по оси Y (не наезжают друг на друга)
    fig, axes = plt.subplots(len(ks), 1, figsize=(5.5, 3.1 * len(ks)), sharex=True)
    if len(ks) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks):
        sub = agg[agg['K'] == k].set_index('variant').reindex(
            [v for v in order if v in set(agg['variant'])]).reset_index()
        base = float(sub.loc[sub['variant'] == 'baseline', 'mean'].iloc[0])
        delta = sub['mean'] - base
        colors = [C_GREY if v == 'baseline' else (C_ACCENT if d > 0 else C_MAIN)
                  for v, d in zip(sub['variant'], delta)]
        y = np.arange(len(sub))
        ax.barh(y, delta, color=colors)
        err = sub['std'].fillna(0)
        ax.errorbar(delta, y, xerr=err, fmt='none', ecolor='k', capsize=3, lw=0.8)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub['variant'])
        ax.invert_yaxis()
        ax.set_title(f'{title_let.get(str(k), "")}) K={k} сегментов '
                     f'({"мало" if k == min(ks) else "много"} данных)', loc='left')
        for i, d in enumerate(delta):
            off = 0.04 if d >= 0 else -0.04
            ax.text(d + off, i, f'{d:+.2f}', va='center',
                    ha='left' if d >= 0 else 'right', fontsize=8)
    axes[-1].set_xlabel('ΔMAE относительно baseline (меньше — лучше; планки — std по сидам)')
    axes[0].legend(handles=[
        Patch(fc=C_MAIN, label='вариант лучше baseline (ΔMAE < 0)'),
        Patch(fc=C_ACCENT, label='вариант хуже baseline (ΔMAE > 0)'),
        Patch(fc=C_GREY, label='baseline (контроль)'),
        Line2D([], [], color='k', lw=1, marker='|', linestyle='None',
               label='± std по сидам (42/43/44)'),
    ], loc='lower right', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    save(fig, 'fig2_sweep_delta')


# ----------------------------------------------------------------------------
def fig3_per_target(new=False):
    if new:
        # Новый пайплайн (Ф1/Ф2): Реал 700 vs Реал 5949 vs 5949 без КУ-признаков (ablation Ф2)
        rows = []
        for run in reg.list_runs():
            m = run['manifest']
            if not m:
                continue
            notes = m.get('hypothesis') or ''
            if not notes.startswith(('F1 ', 'F2 ')):
                continue
            phys = m.get('results', {}).get('physical', {})
            if not phys:
                continue
            eng = m['features'].get('feature_engineering') or {}
            rel = bool(eng.get('relative_wave_angle') and eng.get('relative_wind_angle'))
            rows.append({'exp': notes.split()[0], 'n': m['data']['rows_train'],
                         'rel': rel, 'seed': m['training']['seed'],
                         'per_target_mae': {k: v['mae'] for k, v in phys.get('per_target', {}).items()}})
        df = pd.DataFrame(rows).drop_duplicates(subset=['exp', 'n', 'rel', 'seed'])
        cond = {
            'Реал 700': (df['exp'] == 'F1') & (df['n'] == 700),
            'Реал 5949': (df['exp'] == 'F1') & (df['n'] == 5949),
            '5949 без КУ (Ф2)': (df['exp'] == 'F2') & (df['n'] == 5949),
        }
    else:
        df = pd.read_csv(PROJECT_ROOT / 'results/learning_curve/all_study_runs.csv')
        df['per_target_mae'] = df['per_target_mae'].apply(ast.literal_eval)
        cond = {
            'Реал 700': (df['exp'] == 'E1') & (df['n_train_rows'] == 700),
            'Реал 5949': (df['exp'] == 'E1') & (df['n_train_rows'] == 5949),
            'Синтетика 30k': (df['exp'] == 'E2') & df['notes'].str.contains('30k'),
            '700+17.5k син.': (df['exp'] == 'E3') & df['notes'].str.contains('17500'),
        }
    short = {'Pitch(градусы)': 'Pitch,°', 'Roll(градусы)': 'Roll,°',
             'Vertical(Метр)': 'Vertical,м', 'Velocity.Rolling(°/мин)': 'Vel.Roll,°/мин',
             'ROT(°/мин)': 'ROT,°/мин', 'SOG(узлы)': 'SOG,уз'}
    data = {}
    for name, mask in cond.items():
        sub = df[mask]
        if sub.empty:
            continue
        pts = [pd.Series(p) for p in sub['per_target_mae']]
        data[name] = pd.concat(pts, axis=1).mean(axis=1)

    all_t = sorted(set().union(*[set(v.index) for v in data.values()]))
    use = [t for t in all_t if t in short]
    x = np.arange(len(use))
    w = 0.2
    colors = ([C_GREY, C_MAIN, '#ff7f0e'] if new
              else [C_GREY, C_MAIN, '#2ca02c', '#ff7f0e'])

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (name, series) in enumerate(data.items()):
        vals = [series.get(t, np.nan) for t in use]
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=colors[i])
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([short[t] for t in use])
    ax.set_ylabel('Test MAE, физ. ед. (агрег. по 8 целям, log)')
    ax.set_title('г) Ошибка по целям: объём реальных данных' +
                 (' vs без КУ-признаков (ablation Ф2)' if new else
                  ' vs синтетика vs аугментация'))
    ax.legend(ncol=2, title='Обучающая выборка', title_fontsize=8)
    plt.tight_layout()
    save(fig, 'fig3_per_target')


# ----------------------------------------------------------------------------
def fig4_regimes(new=False):
    if new:
        # per-regime лучшего прогона Ф1 (k9-s42): лёгкий режим — отдельный режим (Ф0.5)
        src = next((PROJECT_ROOT / 'results' / 'regimes').glob('20260929-*f1-minimal-k9-s42*/regimes.csv'))
    else:
        src = next((PROJECT_ROOT / 'results').glob('regimes_*/regimes.csv'))
    df = pd.read_csv(src)
    df = df[df['regime'] != 'ALL'].sort_values('mae')
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.barh(df['regime'], df['mae'], color=C_MAIN,
            label='Test MAE, физ. ед. (агрег. по 8 целям)')
    ax.set_xlabel('Test MAE, физ. ед. (агрег. по 8 целям) — меньше лучше')
    ax.set_title('д) Качество прогноза по режимам волнения')
    for i, (m, n_) in enumerate(zip(df['mae'], df['n_values'])):
        ax.text(m, i, f' {m:.2f} (n={n_})', va='center', fontsize=8)
    ax.legend(loc='lower right', fontsize=8)
    save(fig, 'fig4_regimes')


# ----------------------------------------------------------------------------
def fig5_period_drift(new=False):
    """Медианный период качки Roll по чанкам записи: пики через zero-crossing."""
    data_path = PROJECT_ROOT / 'data/raw/your_data.csv'
    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except Exception:
        try:
            df = pd.read_csv(data_path, sep='\t')
        except Exception:
            df = pd.read_csv(data_path)
    roll = df['Roll(градусы)'].values.astype(float)

    chunk, periods_t, periods = 700, [], []
    for i in range(0, len(roll) - chunk, chunk):
        seg = roll[i:i + chunk] - roll[i:i + chunk].mean()
        # число пересечений нуля вверх -> период = 2 * длительность / n_cross
        up = np.sum((seg[:-1] < 0) & (seg[1:] >= 0))
        if up >= 3:
            periods_t.append(i + chunk / 2)
            periods.append(2 * chunk / up)

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(periods_t, periods, 'o-', color=C_MAIN, ms=4,
            label='медианный период качки Roll\n(чанки 700 строк, zero-crossing)')
    ax.axvline(4000, color=C_GREY, ls=':', lw=1,
               label=('строка 4000 — граница спокойного/активного региона '
                      '(оба в обучении, Ф0.5)' if new else
                      'строка 4000 — начало активного шторма\n(регион исследования)'))
    if not new:
        ax.annotate('начало активного шторма', (4000, ax.get_ylim()[1] * 0.92),
                    xytext=(5, 0), textcoords='offset points', fontsize=8, color=C_GREY)
    ax.set_xlabel('Строка записи (сек при 1 Гц)')
    ax.set_ylabel('Период качки Roll, с (меньше — чаще)')
    ax.set_title('е) Дрейф собственного периода качки по мере развития шторма')
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'fig5_period_drift')


# ----------------------------------------------------------------------------
def main():
    global _use_old
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='Journal figures')
    parser.add_argument('--new-pipeline', action='store_true',
                        help='Ф5: перегенерация с прогонов нового пайплайна '
                             '(Ф1/Ф2/Ф3/Ф4) -> results/article/figs_v2/')
    args = parser.parse_args()
    new = args.new_pipeline
    _use_old = not new
    print('Журнальные рисунки ->', OUT if _use_old else OUT_NEW)
    a, b = fig1_learning_curve(new=new)
    print(f'  степенной закон: MAE = {a:.1f}·n^({b:.3f})')
    fig2_sweep(new=new)
    fig3_per_target(new=new)
    fig4_regimes(new=new)
    fig5_period_drift(new=new)
    print('Готово.')


if __name__ == '__main__':
    main()
