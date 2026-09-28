"""
Починка «зависших» статусов прогонов (уроки тестирования панели 2026-09-28).

Прогон, оборванный по таймауту/сбою, оставался со статусом 'running' навсегда:
  - он показывался [running] в панели и его нельзя было использовать (500 при прогнозе);
  - experiments.csv не отражал реального состояния.

Правило: статус 'running' + нет results.physical.overall.mae в manifest → 'failed'.
Реестр experiments.csv синхронизируется (столбец status).

Запуск: .venv/Scripts/python.exe scripts/fix_stale_runs.py [--dry-run]
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    dry = '--dry-run' in sys.argv

    fixed, ok = [], []
    for run in reg.list_runs():
        m = run['manifest']
        if m is None or m.get('status') != 'running':
            continue
        mae = (m.get('results', {}).get('physical', {}).get('overall', {}) or {}).get('mae')
        if mae is None:
            fixed.append(run)
        else:
            ok.append(run)  # завершён, но статус не обновился — такого быть не должно

    print(f'Зависших «running» без результатов: {len(fixed)}')
    for r in fixed:
        print(f'  {r["run_id"]}: running → failed  ({(r["manifest"].get("hypothesis") or "")[:60]})')
        if not dry:
            reg.update_manifest(r['dir'], {'status': 'failed'})
    print(f'«running», но с результатами (требуют ручной проверки): {len(ok)}')
    for r in ok:
        print(f'  {r["run_id"]}')

    # синхронизация experiments.csv
    if fixed and not dry:
        path = reg.registry_path()
        rows = reg.read_registry()
        fixed_ids = {r['run_id'] for r in fixed}
        for row in rows:
            if row.get('run_id') in fixed_ids:
                row['status'] = 'failed'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=reg.REGISTRY_COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print('experiments.csv синхронизирован.')
    print('Готово.' if not dry else '(dry-run — ничего не изменено)')


if __name__ == '__main__':
    main()
