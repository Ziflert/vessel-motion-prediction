"""
A3 (IDEAS.md): матрица горизонтов прогноза 10/20/30 — кривая «ошибка от упреждения»,
ключевой график для СППР (CONCEPT §4.2: на сколько минут вперёд можно верить прогнозу).

Сетка: prediction_horizon ∈ {10, 20, 30} × варианты {baseline, smooth0} × сиды {42,43,44},
на полной выборке (K=9 сегментов ≈ 5949 строк). Все прочие условия идентичны исследованию
RESEARCH_LOG (skip-rows 4000, fixed scaler, subset-seed 123, ≤100 эпох).

Прогоны H=20 уже есть в мультисиде A1 (RESEARCH_LOG §5.5.1, 'SWEEP <variant> K=9 seed<s>')
— переиспользуются (skip). Новыми остаются 12 прогонов: H∈{10,30} × 2 варианта × 3 сида.

Мультисид обязателен по уроку §5.5.1: одиночный сид систематически переоценивает эффекты
(smooth0 −0.97 → −0.34 при 3 сидах); для кривой mean±std по сидам.

Запуск:  .venv/Scripts/python.exe scripts/horizon_sweep.py           # resumable
         .venv/Scripts/python.exe scripts/horizon_sweep.py --only 10,30
Отчёт:   .venv/Scripts/python.exe scripts/horizon_report.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg

PYTHON = sys.executable

# Варианты loss-конфигураций (как в sweep.py)
GRID = [
    {'name': 'baseline', 'args': []},                        # контроль сверки с E1/A1
    {'name': 'smooth0',  'args': ['--smooth-weight', '0']},  # без smoothness-loss (лучший кандидат §5.5.1)
]

HORIZONS = [10, 20, 30]  # шагов упреждения (1 Гц → 10/20/30 с)
SEEDS = [42, 43, 44]
K = 9  # полная выборка: 9 train-сегментов (RESEARCH_LOG §3)


def run_name(variant: str, h: int) -> str:
    return f'HORIZON {variant} H={h}'


def already_done(variant: str, h: int, seed: int) -> bool:
    """Готовые прогоны: новые 'HORIZON <v> H=<h> seed<s>' ИЛИ легаси
    'SWEEP <v> K=9 seed<s>' (это те же условия при h=20, мультисид A1)."""
    for run in reg.list_runs():
        m = run['manifest']
        if not m:
            continue
        hyp = (m.get('hypothesis') or '')
        if hyp == f'{run_name(variant, h)} seed{seed}':
            return True
        if h == 20 and hyp == f'SWEEP {variant} K={K} seed{seed}':
            return True
    return False


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='Запятая-список горизонтов (по умолчанию 10,20,30)')
    args = parser.parse_args()

    horizons = [int(h) for h in args.only.split(',')] if args.only else HORIZONS

    total = len(GRID) * len(horizons) * len(SEEDS)
    print(f'Сетка: {len(GRID)} вариантов × горизонты {horizons} × сиды {SEEDS} = {total} прогонов')
    done = skipped = failed = 0

    for h in horizons:
        for seed in SEEDS:
            for v in GRID:
                if already_done(v['name'], h, seed):
                    print(f'-- skip (уже есть): {v["name"]} H={h} seed={seed}')
                    skipped += 1
                    continue
                cmd = [PYTHON, 'run_training.py',
                       '--skip-rows', '4000',
                       '--train-segments', str(K), '--subset-seed', '123',
                       '--fixed-scaler', '--max-epochs', '100',
                       '--prediction-horizon', str(h),
                       '--seed', str(seed),
                       '--notes', f'{run_name(v["name"], h)} seed{seed}'] + v['args']
                print(f'>>> {v["name"]} H={h} seed={seed}')
                r = subprocess.run(cmd, cwd=PROJECT_ROOT)
                if r.returncode != 0:
                    print(f'!! ПРОВАЛ: {v["name"]} H={h} seed={seed} (код {r.returncode})')
                    failed += 1
                else:
                    done += 1

    print(f'\nГотово: выполнено {done}, пропущено {skipped}, провалено {failed}.')
    print('Отчёт: .venv/Scripts/python.exe scripts/horizon_report.py')


if __name__ == '__main__':
    main()
