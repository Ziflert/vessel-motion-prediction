"""
A1 (IDEAS.md): добивка недостающих прогонов мультисида.
Запускает только отсутствующие конфигурации (проверка по реестру):
  - smooth0/huber0 K=9 seed44 (сорвались по таймауту сессии)
  - baseline K={2,9} seed{43,44} (контроль для mean±std)
Resumable: при повторном запуске пропускает уже завершённое.

Запуск: .venv/Scripts/python.exe scripts/a1_finish.py
Лог:    results/a1_finish.log
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg

PYTHON = sys.executable
JOBS = [
    # (variant, extra_args)
    ('smooth0',  ['--smooth-weight', '0']),
    ('huber0',   ['--huber-weight', '0']),
    ('baseline', []),
]


def finished(variant: str, k: int, seed: int) -> bool:
    for run in reg.list_runs():
        m = run['manifest']
        if not m or not (m.get('hypothesis') or '').startswith('SWEEP'):
            continue
        parts = m['hypothesis'].split()
        if len(parts) < 3:
            continue
        cfg, kk = parts[1], parts[2].split('=')[1]
        if (cfg == variant and kk == str(k)
                and m.get('training', {}).get('seed') == seed
                and m.get('results', {}).get('physical', {}).get('overall', {}).get('mae') is not None):
            return True
    return False


def main():
    tasks = []
    for variant, extra in JOBS:
        for k in [2, 9]:
            for seed in [43, 44]:
                if not finished(variant, k, seed):
                    tasks.append((variant, k, seed, extra))
    print(f'К выполнению: {len(tasks)} прогонов')
    for variant, k, seed, extra in tasks:
        cmd = [PYTHON, 'run_training.py',
               '--skip-rows', '4000',
               '--train-segments', str(k), '--subset-seed', '123',
               '--fixed-scaler', '--max-epochs', '100',
               '--seed', str(seed),
               '--notes', f'SWEEP {variant} K={k} seed{seed}'] + extra
        print(f'>>> {variant} K={k} seed={seed}', flush=True)
        r = subprocess.run(cmd, cwd=PROJECT_ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f'    код {r.returncode}', flush=True)
    print('Готово.')


if __name__ == '__main__':
    main()
