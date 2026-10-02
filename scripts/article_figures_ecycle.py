# -*- coding: utf-8 -*-
"""Фигуры E-цикла для статьи (fig8–fig10, план §2.1).

Строит из манифестов models_archive (источник правды) PDF+PNG 300 dpi в
results/article/figs_v2/:
  fig8 — дозовая кривая E3TDOSE (MAE и skill vs доза синтетики, K=2)
         + контроли E3TBAD/E3TFULL;
  fig9 — learning curve E1T на реале 71k (K=2..16 + FULL): MAE + skill;
  fig10 — сравнение датасетов 12к (your_data, шторм) ↔ 71к (real_w5w6w7,
          попутная): skill и R² (НЕ физические MAE — разные сценарии).

Запуск: .venv/Scripts/python.exe -X utf8 scripts/article_figures_ecycle.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = PROJECT_ROOT / 'results' / 'article' / 'figs_v2'

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
C_GREEN = '#2ca02c'


def load_series(prefixes):
    """Собрать physical overall + skill по манифестам с hypothesis-префиксами."""
    out = []
    for p in sorted(glob.glob(str(PROJECT_ROOT / 'models_archive' / '*' / 'manifest.json'))):
        with open(p, encoding='utf-8') as f:
            m = json.load(f)
        h = m.get('hypothesis', '')
        if any(h.startswith(x) for x in prefixes):
            r = m['results']
            out.append({
                'hypothesis': h,
                'mae': r['physical']['overall']['mae'],
                'r2': r['physical']['overall']['r2'],
                'skill': r['skill_vs_persistence']['overall'],
            })
    return out


def ms(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)
    print(f'  saved {name}.pdf/.png')


def fig8():
    """Дозовая кривая E3TDOSE при K=2 + контроли."""
    dose_runs = {
        0: load_series(['E1T transas-real K=2 ']),          # дозa 0 = E1T K=2
        5000: load_series(['E3TDOSE K=2 dose=5000']),
        10000: load_series(['E3TDOSE K=2 dose=10000']),
        30000: load_series(['E3T transas K=2 +syn30k']),    # 30k = E3T K=2
        70000: load_series(['E3TDOSE K=2 dose=70000']),
    }
    bad = load_series(['E3TBAD'])
    full = load_series(['E3TFULL'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # (a) дозовая кривая
    doses = sorted(dose_runs)
    mae_m, mae_s, sk_m, sk_s = [], [], [], []
    for d in doses:
        mm, ss = ms([r['mae'] for r in dose_runs[d]])
        km, ks = ms([r['skill'] for r in dose_runs[d]])
        mae_m.append(mm); mae_s.append(ss); sk_m.append(km); sk_s.append(ks)
    ax1.errorbar(doses, mae_m, yerr=mae_s, marker='o', ms=5, capsize=3,
                 color=C_MAIN, label='MAE (physical overall)')
    ax1.set_xscale('symlog', linthresh=1000)
    ax1.set_xticks(doses)
    ax1.set_xticklabels(['0', '5k', '10k', '30k', '70k'])
    ax1.set_xlabel('Доза синтетики, строк (K=2 реала)')
    ax1.set_ylabel('MAE, физ. ед.')
    ax1b = ax1.twinx()
    ax1b.errorbar(doses, sk_m, yerr=sk_s, marker='s', ms=4, capsize=3,
                  color=C_ACCENT, ls='--', label='Skill vs persistence')
    ax1b.set_ylabel('Skill vs persistence', color=C_ACCENT)
    ax1b.tick_params(axis='y', colors=C_ACCENT)
    ax1b.grid(False)
    lines = ax1.get_legend_handles_labels()[0] + ax1b.get_legend_handles_labels()[0]
    ax1.legend(lines, ['MAE (physical overall)', 'Skill vs persistence'],
               loc='lower left', framealpha=0.9)
    ax1.set_title('(а) Дозовая кривая аугментации')

    # (b) контроли
    groups = [
        ('Реал K=2\n(доза 0)', dose_runs[0], C_MAIN),
        ('+синт 30k\n(своя калибровка)', dose_runs[30000], C_GREEN),
        ('+синт 30k\n(чужая калибровка,\nE3TBAD)', bad, C_ACCENT),
        ('Реал FULL 49.7k\n+синт 71k\n(E3TFULL)', full, C_GREY),
    ]
    xs = np.arange(len(groups))
    for x, (label, runs, color) in zip(xs, groups):
        mm, ss = ms([r['mae'] for r in runs])
        ax2.bar(x, mm, yerr=ss, capsize=3, color=color, alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax2.set_ylabel('MAE, физ. ед.')
    ax2.set_title('(б) Контроли: калибровка и объём')
    fig.tight_layout()
    save(fig, 'fig8_dose_controls')


def fig9():
    """Learning curve E1T на реале 71k."""
    runs = load_series(['E1T transas-real'])
    groups = {
        1400: 'K=2', 2800: 'K=4', 5600: 'K=8', 11200: 'K=16', 49700: 'FULL',
    }
    by = {}
    for r in runs:
        n = int(r['hypothesis'].split('K=')[1].split(' ')[0]) * 700 if 'K=' in r['hypothesis'] else 49700
        by.setdefault(n, []).append(r)
    ns = sorted(by)
    mae_m, mae_s, sk_m, sk_s = [], [], [], []
    for n in ns:
        mm, ss = ms([r['mae'] for r in by[n]])
        km, ks = ms([r['skill'] for r in by[n]])
        mae_m.append(mm); mae_s.append(ss); sk_m.append(km); sk_s.append(ks)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(ns, mae_m, yerr=mae_s, marker='o', ms=5, capsize=3, color=C_MAIN,
                label='MAE (physical overall)')
    for n, mm, ss in zip(ns, mae_m, mae_s):
        label = groups[n]
        ax.annotate(label, (n, mm), textcoords='offset points', xytext=(6, 7), fontsize=8)
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([f'{n//1000}k' if n >= 1000 else str(n) for n in ns])
    ax.set_xlabel('Объём train, строк (реал w5–w7)')
    ax.set_ylabel('MAE, физ. ед.')
    axb = ax.twinx()
    axb.errorbar(ns, sk_m, yerr=sk_s, marker='s', ms=4, capsize=3, ls='--',
                 color=C_ACCENT, label='Skill vs persistence')
    axb.set_ylabel('Skill vs persistence', color=C_ACCENT)
    axb.tick_params(axis='y', colors=C_ACCENT)
    axb.grid(False)
    lines = ax.get_legend_handles_labels()[0] + axb.get_legend_handles_labels()[0]
    ax.legend(lines, ['MAE (physical overall)', 'Skill vs persistence'], loc='center right')
    ax.set_title('Learning curve на второй записи (E1T)')
    fig.tight_layout()
    save(fig, 'fig9_e1t_curve')


def fig10():
    """Сравнение датасетов 12к ↔ 71к: skill и R² (НЕ MAE)."""
    # старый датасет: production F3 roll_w4 K=9 (5949 строк, шторм)
    old = load_series(['F3 roll_w4 K=9'])
    # новый датасет: E1T FULL (49700 строк, попутная) и E3TFULL
    new_full = load_series(['E1T transas-real FULL'])
    new_aug = load_series(['E3TFULL'])
    # сопоставимая точка объёма: E1T K=8 (~5.6k строк)
    new_k8 = load_series(['E1T transas-real K=8 '])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    groups = [
        ('Шторм 12к строк\n(train 5.9k)\nF3 roll_w4 K=9', old),
        ('Попутная 71к строк\n(train 5.6k)\nE1T K=8', new_k8),
        ('Попутная 71к строк\n(train 49.7k)\nE1T FULL', new_full),
        ('Попутная 71к + синт 71k\n(train 49.7k)\nE3TFULL', new_aug),
    ]
    for ax, key, title, ylab in (
        (axes[0], 'skill', 'Skill vs persistence', 'Skill'),
        (axes[1], 'r2', 'R² (physical overall)', 'R²'),
    ):
        for x, (label, runs) in enumerate(groups):
            mm, ss = ms([r[key] for r in runs])
            ax.bar(x, mm, yerr=ss, capsize=3,
                   color=[C_GREY, C_MAIN, C_GREEN, C_GREEN][x], alpha=0.85)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.axhline(0, color='k', lw=0.6)
    fig.suptitle('Сравнение датасетов по относительным метрикам '
                 '(разные погодные сценарии — физические MAE несопоставимы)',
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, 'fig10_dataset_comparison')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('E-cycle figures ->', OUT)
    fig8()
    fig9()
    fig10()


if __name__ == '__main__':
    main()
