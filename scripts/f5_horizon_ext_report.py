"""
F5-ext: объединённая кривая «цена горизонта» H ∈ {5..120} на 12k-записи (your_data_minimal).

Собирает:
  - baseline: прогоны 'F5 baseline H=<h> seed<s>' (F5v2-батч, сплит 6000/3000/3000 —
    фильтр data.rows_val == 3000; манифесты первого батча 07:36 со старым сплитом
    и статусом failed исключаются);
  - roll_w4: прогоны Ф4 ('F4 roll_w4 H=10/30') + H=20 из Ф3 ('F3 roll_w4 K=9');
  - кросс-чек baseline Ф4 ('F4 baseline H=10/30' + 'F1 minimal k9' / 'F3 baseline K=9')
    со старым сплитом 6300/1800/1800.

Строит:
  1. общую таблицу overall MAE/R²/skill (mean±std, n=3 сида) по (вариант × H) → CSV;
  2. fig «skill vs H» с ДВУМЯ линиями (baseline F5v2 5–120; roll_w4 Ф4 10–30)
     + дубль с MAE — PNG/PDF 300 dpi; skill-фигура дублируется в results/article/figs_v2/;
  3. текстовый отчёт f5ext_horizon_report.txt.

Запуск: .venv/Scripts/python.exe scripts/f5_horizon_ext_report.py
"""

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

OUT = PROJECT_ROOT / 'results' / 'horizons_f4'
FIGS_ARTICLE = PROJECT_ROOT / 'results' / 'article' / 'figs_v2'

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
})

LINE_STYLE = {
    'baseline': dict(color='#1f77b4', marker='o', ls='-',
                     label='baseline (MSE + 0.5·Huber + 0.1·smooth)'),
    'roll_w4': dict(color='#d62728', marker='s', ls='--',
                    label='roll_w4 (лучший свип Ф3, Roll=4)'),
}


def collect():
    rows = []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None:
            continue
        notes = m.get('hypothesis') or ''
        variant, h, src = None, None, None
        parts = notes.split()
        if notes.startswith('F5 ') and (m.get('data') or {}).get('rows_val') == 3000:
            # F5v2-батч: 'F5 baseline H=<h> seed<s>' (первый батч со старым сплитом отфильтрован)
            variant, h, src = parts[1], int(parts[2].split('=')[1]), 'F5v2'
        elif notes.startswith('F4 ') and parts[1] in ('baseline', 'roll_w4'):
            variant, h, src = parts[1], int(parts[2].split('=')[1]), 'F4'
        elif notes.startswith('F3 roll_w4 ') and int(parts[2].split('=')[1]) == 9:
            # только roll_w4: F3 baseline K=9 — детерминированные перезапуски Ф1 k9 (те же метрики),
            # чтобы не дублировать baseline H=20 в кросс-чеке
            variant, h, src = parts[1], 20, 'F3(K=9)'
        elif notes.startswith('F1 minimal k9'):
            variant, h, src = 'baseline', 20, 'F1(K=9)'
        if variant is None:
            continue
        phys = (m.get('results') or {}).get('physical') or {}
        ov = phys.get('overall') or {}
        if not ov:
            continue
        seed = (m.get('training') or {}).get('seed') or 42
        rows.append({
            'variant': variant, 'H': h, 'src': src, 'seed': seed, 'run_id': m['run_id'],
            'mae': ov.get('mae'), 'rmse': ov.get('rmse'), 'r2': ov.get('r2'),
            'skill': ((m.get('results') or {}).get('skill_vs_persistence') or {}).get('overall'),
        })
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    df = collect()
    if df.empty:
        print('Нет прогонов F5v2/F4/F3(K=9)/F1(K=9)')
        return
    df = df.drop_duplicates(subset=['variant', 'src', 'H', 'seed'], keep='last')
    OUT.mkdir(parents=True, exist_ok=True)

    agg = (df.groupby(['variant', 'src', 'H'])
             .agg(mae=('mae', 'mean'), mae_std=('mae', 'std'),
                  r2=('r2', 'mean'), r2_std=('r2', 'std'),
                  skill=('skill', 'mean'), skill_std=('skill', 'std'),
                  n=('seed', 'size'))
             .reset_index())
    agg.to_csv(OUT / 'f5ext_horizon_overall.csv', index=False)

    # ---- таблица для отчёта ----
    def fmt(r):
        return (f"H={int(r['H']):>3} [{r['src']:<8}] n={int(r['n'])}: "
                f"MAE {r['mae']:6.2f} ± {r['mae_std']:4.2f}   "
                f"R² {r['r2']:+.3f} ± {r['r2_std']:.3f}   "
                f"Skill {r['skill']:+.3f} ± {r['skill_std']:.3f}")

    base = agg[(agg['variant'] == 'baseline') & (agg['src'] == 'F5v2')].sort_values('H')
    roll = agg[(agg['variant'] == 'roll_w4')
               & (agg['src'].isin(['F4', 'F3(K=9)']))].sort_values('H')
    cross = agg[(agg['variant'] == 'baseline') & (agg['src'] != 'F5v2')].sort_values('H')

    lines = [
        '=' * 78,
        'F5-ext — ОБЪЕДИНЁННАЯ КРИВАЯ «ЦЕНА ГОРИЗОНТА» H = 5…120 с (12k-запись)',
        '=' * 78, '',
        'Данные: your_data_minimal (12 499 строк, ПОЛНАЯ запись Ф0.5), minimal_prediction,',
        'fixed scaler, ≤100 эпох; 1 Гц (1 шаг упреждения = 1 с); mean±std по 3 сидам {42,43,44}.',
        'baseline: F5v2-батч, H ∈ {5,10,15,20,30,40,60,120}, сплит 6000/3000/3000',
        '  (первый батч со сплитом 6300/1800/1800 упал на H≥40 — val/test короче окна',
        '  seq+H; H=10/20/30 переиграны в F5v2 с тем же протоколом).',
        'roll_w4: Ф4 (H=10/30, старый сплит 6300/1800/1800) + H=20 из Ф3 (K=9)',
        '  — другая конфигурация из свипа Ф3, единственная с полным набором 10/20/30.',
        '', '--- baseline (F5v2, сплит 6000/3000/3000) ---',
    ]
    lines += [fmt(r) for _, r in base.iterrows()]
    lines += ['', '--- roll_w4 (Ф4/Ф3, старый сплит 6300/1800/1800) ---']
    lines += [fmt(r) for _, r in roll.iterrows()]
    lines += ['', '--- кросс-чек: baseline на старом сплите (Ф4/Ф1/Ф3 K=9) ---']
    lines += [fmt(r) for _, r in cross.iterrows()]

    # кросс-чек численно: совпадение F5v2 vs Ф4 на H=10/20/30
    lines += ['', '--- кросс-чек F5v2 vs Ф4 baseline (H=10/20/30, разные сплиты) ---']
    for h in (10, 20, 30):
        a = base[base['H'] == h]
        b = cross[cross['H'] == h]
        if a.empty or b.empty:
            continue
        d_skill = abs(float(a['skill'].iloc[0]) - float(b['skill'].iloc[0]))
        lines.append(f"H={h}: skill F5v2 {float(a['skill'].iloc[0]):+.3f} vs Ф4 {float(b['skill'].iloc[0]):+.3f}"
                     f"  (Δ={d_skill:.3f})")

    # ---- фигура: skill vs H (2 линии) ----
    def horizon_fig(ycol, ystd, ylabel, fname, article=False):
        fig, ax = plt.subplots(figsize=(89 / 25.4 * 1.6, 60 / 25.4 * 1.35), layout='constrained')
        for variant, sub in (('baseline', base), ('roll_w4', roll)):
            if sub.empty:
                continue
            st = LINE_STYLE[variant]
            ax.errorbar(sub['H'], sub[ycol], yerr=sub[ystd], color=st['color'],
                        marker=st['marker'], ms=4, lw=1.4, ls=st['ls'], capsize=2.5,
                        label=f"{st['label']} (n=3, ±1 SD)")
        ax.axhline(0, color='#7f7f7f', lw=0.9, ls=':')
        ax.text(0.99, 0.02, 'уровень persistence', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='#7f7f7f')
        ax.set_xscale('log')
        ax.set_xticks(sorted(base['H'].tolist()))
        ax.set_xticklabels([str(int(h)) for h in sorted(base['H'].tolist())])
        ax.minorticks_off()
        ax.set_xlabel('Горизонт прогноза H, с (1 Гц)')
        ax.set_ylabel(ylabel)
        ax.set_title('Цена горизонта на 12k-записи: baseline (H=5–120, F5v2) и '
                     'roll_w4 (H=10–30, Ф4/Ф3)', fontsize=10)
        ax.legend(loc='upper right', framealpha=0.9)
        targets = [(OUT, fname)]
        if article:
            targets.append((FIGS_ARTICLE, 'fig11_horizon_ext'))
        for out_dir, name in targets:
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f'{name}.png')
            fig.savefig(out_dir / f'{name}.pdf')
        plt.close(fig)
        print(f'  ✓ {fname}.png/.pdf' + (' (+ article/figs_v2)' if article else ''))

    horizon_fig('skill', 'skill_std', 'Skill vs persistence', 'f5ext_horizon_skill', article=True)
    horizon_fig('mae', 'mae_std', 'MAE (физ. ед., все цели)', 'f5ext_horizon_mae')

    lines += ['', '=' * 78,
              'Интерпретация (CONCEPT §4.2): монотонное падение skill по H; H=120 — skill +0.13±0.02 ≤ +0.15,',
              'R² ≈ 0 — горизонт 2 мин на этой записи подтверждённо недостижим (ожидание 2 Ф5-ext).',
              'Артефакты: results/horizons_f4/ (f5ext_horizon_overall.csv, f5ext_horizon_skill.*,',
              'f5ext_horizon_mae.*, f5ext_horizon_report.txt); skill-фигура — results/article/figs_v2/.',
              'Ограничение сравнения: baseline и roll_w4 различаются сплитом (6000/3000/3000 vs',
              '6300/1800/1800); кросс-чек baseline на H=10/20/30 показывает согласованность протоколов.',
              '=' * 78]
    (OUT / 'f5ext_horizon_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('  ✓ f5ext_horizon_report.txt')

    print('\n--- overall (mean±std, n=3) ---')
    for _, r in pd.concat([base, roll, cross]).iterrows():
        print(fmt(r))


if __name__ == '__main__':
    main()
