"""
A3 (IDEAS.md): матрица горизонтов прогноза 10/20/30 — кривая «ошибка от упреждения»,
ключевой график для СППР (CONCEPT §4.2: на сколько минут вперёд можно верить прогнозу).

Сетка: prediction_horizon ∈ {10, 20, 30} × варианты {baseline, smooth0} × сиды {42,43,44},
на полной выборке (K=9 сегментов ≈ 5949 строк). Все прочие условия идентичны исследованию
RESEARCH_LOG (fixed scaler, subset-seed 123, ≤100 эпох).

Старый пайплайн (skip-rows 4000, A3-предшественник): варианты {baseline, smooth0},
notes 'HORIZON ...'; H=20 переиспользуется из мультисида A1 ('SWEEP <v> K=9 seed<s>').
Новый пайплайн v2-канон (--data-minimal, Ф4): варианты {baseline, roll_w4 — лучшая
конфигурация Ф3}, notes 'F4 ...'; H=20 переиспользуется из Ф1 (baseline k9) и Ф3
(roll_w4 K=9).

Запуск:  .venv/Scripts/python.exe scripts/horizon_sweep.py --data-minimal --prefix F4
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
GRID_OLD = [
    {'name': 'baseline', 'args': []},                        # контроль сверки с E1/A1
    {'name': 'smooth0',  'args': ['--smooth-weight', '0']},  # лучший кандидат старого свипа §5.5.1
]
GRID_NEW = [
    {'name': 'baseline', 'args': []},                        # контроль сверки с Ф1/Ф3
    {'name': 'roll_w4',  'args': ['--roll-weight', '4']},    # лучшая конфигурация Ф3 (K=9)
]

HORIZONS = [10, 20, 30]  # шагов упреждения (1 Гц → 10/20/30 с)
SEEDS = [42, 43, 44]
K = 9  # полная выборка: 9 train-сегментов (RESEARCH_LOG §3)


def run_name(variant: str, h: int, prefix: str = 'HORIZON') -> str:
    return f'{prefix} {variant} H={h}'


def already_done(variant: str, h: int, seed: int, prefix: str = 'HORIZON',
                 data_minimal: bool = False) -> bool:
    """Готовые прогоны: новые '<prefix> <v> H=<h> seed<s>' ИЛИ переиспользуемые:
    старый пайплайн h=20 — 'SWEEP <v> K=9 seed<s>' (мультисид A1);
    новый пайплайн h=20 — baseline из Ф1 ('F1 minimal k9 s<s>') и Ф3
    ('F3 baseline K=9 seed<s>'), roll_w4 из Ф3 ('F3 roll_w4 K=9 seed<s>')."""
    for run in reg.list_runs():
        m = run['manifest']
        if not m:
            continue
        hyp = (m.get('hypothesis') or '')
        if hyp == f'{run_name(variant, h, prefix)} seed{seed}':
            return True
        if h == 20 and not data_minimal and hyp == f'SWEEP {variant} K={K} seed{seed}':
            return True
        if h == 20 and data_minimal:
            if variant == 'baseline' and (hyp == f'F1 minimal k9 s{seed}' or
                                          hyp == f'F3 baseline K=9 seed{seed}'):
                return True
            if variant == 'roll_w4' and hyp == f'F3 roll_w4 K=9 seed{seed}':
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
    parser.add_argument('--prefix', type=str, default='HORIZON',
                        help='Префикс notes (HORIZON — старый пайплайн; F4 — новый v2-канон)')
    parser.add_argument('--data-minimal', action='store_true',
                        help='Новый пайплайн: ПОЛНАЯ запись your_data_minimal.csv + '
                             'профиль minimal_prediction (вместо skip-rows 4000)')
    args = parser.parse_args()

    horizons = [int(h) for h in args.only.split(',')] if args.only else HORIZONS
    grid = GRID_NEW if args.data_minimal else GRID_OLD

    total = len(grid) * len(horizons) * len(SEEDS)
    print(f'Сетка: {len(grid)} вариантов × горизонты {horizons} × сиды {SEEDS} = {total} прогонов '
          f'(пайплайн: {"v2-канон minimal" if args.data_minimal else "старый skip-rows 4000"})')
    done = skipped = failed = 0

    for h in horizons:
        for seed in SEEDS:
            for v in grid:
                if already_done(v['name'], h, seed, args.prefix, args.data_minimal):
                    print(f'-- skip (уже есть): {v["name"]} H={h} seed={seed}')
                    skipped += 1
                    continue
                data_flags = (['--data-path', 'data/raw/your_data_minimal.csv',
                               '--profile', 'minimal_prediction'] if args.data_minimal
                              else ['--skip-rows', '4000'])
                cmd = ([PYTHON, 'run_training.py'] + data_flags +
                       ['--train-segments', str(K), '--subset-seed', '123',
                        '--fixed-scaler', '--max-epochs', '100',
                        '--prediction-horizon', str(h),
                        '--seed', str(seed),
                        '--notes', f'{run_name(v["name"], h, args.prefix)} seed{seed}'] + v['args'])
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
