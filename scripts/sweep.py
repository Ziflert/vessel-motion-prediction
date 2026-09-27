"""
Пункт 2 (Tasks fm user): автоматический перебор коэффициентов.

Сетка: варианты loss/target-весов и гиперпараметров × 2 размера обучающей выборки
(K=2 сегмента ≈ 1400 строк «мало», K=9 ≈ 5949 строк «много»). Все прочие условия
идентичны исследованию RESEARCH_LOG (skip-rows 4000, fixed scaler, subset-seed 123,
seed 42, ≤100 эпох). Базовые точки = прогоны E1 (notes начинаются с 'E1').

Запуск:  python scripts/sweep.py            # вся сетка (resumable: готовое пропускается)
         python scripts/sweep.py --only roll_w4,lrl        # подмножество вариантов
Отчёт:   python scripts/sweep_report.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg

PYTHON = sys.executable

# Варианты коэффициентов (переопределения поверх базового профиля full_prediction)
GRID = [
    {'name': 'baseline',  'args': []},                       # повтор профиля (контроль сверки с E1)
    {'name': 'roll_w4',   'args': ['--roll-weight', '4']},   # крен в 4 раза важнее
    {'name': 'roll_w0.5', 'args': ['--roll-weight', '0.5']}, # крен понижен
    {'name': 'huber0',    'args': ['--huber-weight', '0']},  # чистый MSE
    {'name': 'huber1.5',  'args': ['--huber-weight', '1.5']},# робастная составляющая усилена
    {'name': 'smooth0',   'args': ['--smooth-weight', '0']}, # без гладкости
    {'name': 'lr1e-3',    'args': ['--lr', '0.001']},
    {'name': 'lr2e-4',    'args': ['--lr', '0.0002']},
    {'name': 'no-attn',   'args': ['--no-attention']},       # архитектурный коэффициент
]

SIZES = [2, 9]  # число train-сегментов (см. RESEARCH_LOG §3)


def run_name(variant: str, k: int) -> str:
    return f'SWEEP {variant} K={k}'


def already_done(variant: str, k: int) -> bool:
    for run in reg.list_runs():
        m = run['manifest']
        if m and (m.get('hypothesis') or '') == run_name(variant, k):
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
                        help='Запятая-список имён вариантов (по умолчанию вся сетка)')
    parser.add_argument('--seeds', type=str, default='42')
    args = parser.parse_args()

    variants = GRID if not args.only else [g for g in GRID if g['name'] in args.only.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    total = len(variants) * len(SIZES) * len(seeds)
    print(f'Сетка: {len(variants)} вариантов × размеры {SIZES} × сиды {seeds} = {total} прогонов')
    done = skipped = 0

    for k in SIZES:
        for seed in seeds:
            for v in variants:
                if already_done(v['name'], k) and seed == 42:
                    print(f'-- skip (уже есть): {v["name"]} K={k}')
                    skipped += 1
                    continue
                cmd = [PYTHON, 'run_training.py',
                       '--skip-rows', '4000',
                       '--train-segments', str(k), '--subset-seed', '123',
                       '--fixed-scaler', '--max-epochs', '100',
                       '--seed', str(seed),
                       '--notes', f'{run_name(v["name"], k)} seed{seed}'] + v['args']
                print(f'>>> {v["name"]} K={k} seed={seed}')
                r = subprocess.run(cmd, cwd=PROJECT_ROOT)
                if r.returncode != 0:
                    print(f'!! ПРОВАЛ: {v["name"]} K={k} seed={seed} (код {r.returncode})')
                else:
                    done += 1

    print(f'\nГотово: выполнено {done}, пропущено {skipped}.')
    print('Отчёт: python scripts/sweep_report.py')


if __name__ == '__main__':
    main()
