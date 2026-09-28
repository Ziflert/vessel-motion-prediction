"""
Finetune-контур (ШАГ 5 плана; ADR-5, ADR-6): дообучение модели на
«уникальных» данных онлайн-сессии — ОФФЛАЙН, только по явной команде.

Запуск:
    .venv/Scripts/python.exe -m online.finetune --session <session_dir|session.csv> \
        --run-id <base_run_id> [--all-rows] [--skip-cooldown] ...

Пайплайн (план §8):
    1. cooldown-проверка (анти-спам дообучений, план §7);
    2. извлечение сегментов с маркером error_anomaly (+ контекст окна до них);
    3. проверка min-объёма (finetune_min_rows);
    4. дообучение = run_training subprocess: базовые данные + сегменты
       (--extra-train-csv) + warm start (--init-from) + пониженный LR,
       под watchdog'ами: max_epochs, patience, wall-clock timeout;
    5. валидационный гейт: test MAE новой модели (тот же фиксированный holdout,
       что у базовой) <= базового × (1 + tolerance). Гейт не пройден →
       статус 'rejected-finetune' в manifest (модель останется кандидатом,
       автосвитча нет — ADR-6);
    6. cooldown-файл обновляется ТОЛЬКО при успешном гейте.

Оценку «на сегментах сессии» не делаем гейтом: новая модель дообучена на них —
метрика вырождается (leakage). Гейт — только честный holdout базовой модели.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config.online import OnlineConfig
import registry as reg


FT_DIR = PROJECT_ROOT / 'data' / 'online_finetune'
STATE_FILE = PROJECT_ROOT / 'results' / 'online' / '.finetune_state.json'


def force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(description='Дообучение модели на уникальных данных сессии')
    p.add_argument('--session', required=True,
                   help='Путь к директории сессии (results/online/<id>) или к session.csv')
    p.add_argument('--run-id', required=True, help='Базовый run (warm start + гейт)')
    p.add_argument('--all-rows', action='store_true',
                   help='Дообучаться на всей сессии, а не только на error_anomaly-сегментах')
    p.add_argument('--data', default='data/raw/your_data.csv',
                   help='Базовый дата-сет обучения (тот же, что у базовой модели)')
    p.add_argument('--skip-rows', type=int, default=4000,
                   help='skip-rows базовых данных (как в исследованиях — активный регион)')
    p.add_argument('--train-segments', type=int, default=9,
                   help='K train-сегментов базовых данных (default 9 — полная выборка)')
    p.add_argument('--lr-frac', type=float, default=0.3,
                   help='LR дообучения = доля от базового LR (default 0.3)')
    p.add_argument('--max-epochs', type=int, default=None, help='default из OnlineConfig')
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--min-rows', type=int, default=None)
    p.add_argument('--wall-clock-min', type=float, default=None)
    p.add_argument('--cooldown-min', type=float, default=None)
    p.add_argument('--gate-tolerance', type=float, default=0.02,
                   help='Гейт: новый test MAE <= базовый × (1 + tolerance)')
    p.add_argument('--skip-cooldown', action='store_true')
    p.add_argument('--skip-gate', action='store_true',
                   help='Не отбраковывать по гейту (только для отладки)')
    return p.parse_args()


def check_cooldown(cooldown_min: float) -> str | None:
    """Возвращает причину отказа или None."""
    if not STATE_FILE.exists():
        return None
    try:
        st = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        last = datetime.fromisoformat(st['last_finished'])
        waited = (datetime.now() - last).total_seconds() / 60
        if waited < cooldown_min:
            return (f'cooldown: прошло {waited:.0f} мин из {cooldown_min:.0f} '
                    f'(последнее дообучение {st.get("last_run_id")}). '
                    f'Либо подождите, либо --skip-cooldown.')
    except (KeyError, ValueError, json.JSONDecodeError):
        pass
    return None


def extract_segments(session_csv: Path, seq_len: int, all_rows: bool) -> pd.DataFrame:
    """
    Сегменты с маркером error_anomaly, расширенные на seq_len строк контекста
    ДО начала сегмента (окну нужна история) и слитые в непрерывные куски.
    """
    df = pd.read_csv(session_csv)
    for col in ('error_anomaly', 'input_anomaly', 'predicted'):
        if col not in df.columns:
            raise SystemExit(f'В session.csv нет колонки {col} — не та сессия?')
    raw_cols = [c for c in df.columns
                if c not in ('tick', 'predicted', 'input_anomaly',
                             'input_anomaly_features', 'error_anomaly',
                             'error_ratio')]
    mask = np.ones(len(df), dtype=bool) if all_rows \
        else (df['error_anomaly'].values == 1)
    if not mask.any():
        raise SystemExit('В сессии нет ни одного error_anomaly-маркера — '
                         'дообучаться не на чем (или используйте --all-rows).')
    # расширение контекстом назад + горизонт вперёд
    idx = np.flatnonzero(mask)
    lo = np.maximum(idx - seq_len, 0)
    hi = np.minimum(idx + 1, len(df))
    keep = np.zeros(len(df), dtype=bool)
    for a, b in zip(lo, hi):
        keep[a:b] = True
    out = df.loc[keep, raw_cols]
    return out


def find_new_run(notes: str, started: datetime):
    """Новый run по hypothesis == notes, созданный после старта дообучения."""
    for run in reversed(reg.list_runs()):
        m = run['manifest']
        if m is None:
            continue
        if m.get('hypothesis') == notes and \
                str(m.get('created_at', '')) >= started.isoformat(timespec='seconds'):
            return run
    return None


def main():
    force_utf8_stdio()
    args = parse_args()
    cfg = OnlineConfig()
    max_epochs = args.max_epochs or cfg.finetune_max_epochs
    patience = args.patience or cfg.finetune_patience
    min_rows = args.min_rows or cfg.finetune_min_rows
    wall_clock_min = args.wall_clock_min or cfg.finetune_wall_clock_minutes
    cooldown_min = args.cooldown_min or cfg.finetune_cooldown_minutes

    # --- 1. cooldown -------------------------------------------------------
    if not args.skip_cooldown:
        reason = check_cooldown(cooldown_min)
        if reason:
            raise SystemExit(f'⛔ {reason}')

    # --- базовая модель ------------------------------------------------------
    base_dir = reg.resolve_run(args.run_id)
    base_manifest = reg.load_manifest(base_dir)
    if base_manifest is None:
        raise SystemExit(f'У базового run нет manifest: {base_dir.name}')
    base_mae = (base_manifest.get('results', {}).get('physical', {})
                .get('overall', {}).get('mae'))
    if base_mae is None:
        raise SystemExit('В manifest базовой модели нет results.physical.overall.mae')
    base_lr = (base_manifest.get('training', {}).get('learning_rate')
               or base_manifest.get('training', {}).get('lr') or 0.0005)

    # --- 2. сегменты из сессии ------------------------------------------------
    session_path = Path(args.session)
    if session_path.is_dir():
        session_id = session_path.name
        session_csv = session_path / 'session.csv'
    else:
        session_id = session_path.parent.name
        session_csv = session_path
    if not session_csv.exists():
        raise SystemExit(f'Нет файла: {session_csv}')

    seq_len = base_manifest.get('model', {}).get('sequence_length', 120)
    ft_df = extract_segments(session_csv, seq_len, args.all_rows)
    if len(ft_df) < min_rows:
        raise SystemExit(f'⛔ Слишком мало строк для дообучения: {len(ft_df)} '
                         f'< min_rows={min_rows} (план §7). Продолжайте сбор данных '
                         f'или понизьте порог --min-rows осознанно.')
    FT_DIR.mkdir(parents=True, exist_ok=True)
    ft_csv = FT_DIR / f'{session_id}_ft.tsv'
    ft_df.to_csv(ft_csv, sep='\t', index=False)
    print(f'✓ Извлечено {len(ft_df)} строк уникальных данных → {ft_csv.name}')

    # --- 3-4. дообучение (subprocess + watchdog'и) -----------------------------
    notes = f'online-finetune from {base_dir.name} session {session_id}'
    lr = base_lr * args.lr_frac
    cmd = [
        sys.executable, 'run_training.py',
        '--data-path', str(args.data),
        '--skip-rows', str(args.skip_rows),
        '--train-segments', str(args.train_segments),
        '--subset-seed', '123',
        '--fixed-scaler',
        '--extra-train-csv', str(ft_csv),
        '--init-from', str(base_dir),
        '--max-epochs', str(max_epochs),
        '--lr', f'{lr:.6g}',
        '--seed', str(base_manifest.get('training', {}).get('seed', 42)),
        '--notes', notes,
    ]
    print(f'▶ Дообучение: LR={lr:.2g} (базовый {base_lr:.2g} × {args.lr_frac}), '
          f'max_epochs={max_epochs}, patience={patience}, '
          f'wall-clock лимит {wall_clock_min:.0f} мин')
    started = datetime.now()
    try:
        r = subprocess.run(cmd, cwd=PROJECT_ROOT,
                           timeout=wall_clock_min * 60)
    except subprocess.TimeoutExpired:
        raise SystemExit(f'⛔ Wall-clock таймаут {wall_clock_min:.0f} мин — '
                         f'процесс дообучения остановлен (план §7). '
                         f'Статус не изменён, cooldown не начат.')
    if r.returncode != 0:
        raise SystemExit(f'⛔ run_training завершился с кодом {r.returncode}')

    new_run = find_new_run(notes, started)
    if new_run is None:
        raise SystemExit('⛔ Не найден новый run в реестре — обучение не записалось?')
    new_dir, new_manifest = new_run['dir'], new_run['manifest']
    new_mae = (new_manifest.get('results', {}).get('physical', {})
               .get('overall', {}).get('mae'))

    # --- 5. валидационный гейт ------------------------------------------------
    gate_pass = args.skip_gate or (new_mae is not None and
                                   new_mae <= base_mae * (1 + args.gate_tolerance))
    gate = {
        'base_run': base_dir.name, 'base_mae': base_mae,
        'new_run': new_dir.name, 'new_mae': new_mae,
        'tolerance': args.gate_tolerance,
        'passed': bool(gate_pass),
    }
    print(f'\nГЕЙТ (holdout, физ. единицы): базовый MAE={base_mae:.4f}, '
          f'новый MAE={new_mae:.4f} → {"ПРОЙДЕН" if gate_pass else "НЕ ПРОЙДЕН"}')

    if gate_pass:
        reg.update_manifest(new_dir, {'status': 'candidate',
                                      'gate': gate,
                                      'finetune': {'session': session_id,
                                                   'rows': len(ft_df),
                                                   'ft_csv': str(ft_csv)}})
        STATE_FILE.write_text(json.dumps({
            'last_finished': datetime.now().isoformat(timespec='seconds'),
            'last_run_id': new_dir.name,
        }, ensure_ascii=False), encoding='utf-8')
        print(f'✓ Модель {new_dir.name} зарегистрирована (candidate). '
              f'Cooldown {cooldown_min:.0f} мин начат. Автосвитча нет: выберите '
              f'модель в UI вручную (ADR-6).')
    else:
        reg.update_manifest(new_dir, {'status': 'rejected-finetune', 'gate': gate})
        print(f'✗ Модель {new_dir.name} помечена rejected-finetune '
              f'(базовая остаётся рабочей). Cooldown НЕ начат.')

    # --- отчёт -----------------------------------------------------------------
    report = {
        'session': session_id, 'base_run': base_dir.name,
        'ft_rows': len(ft_df), 'ft_csv': str(ft_csv),
        'max_epochs': max_epochs, 'lr': lr,
        'gate': gate, 'finished_at': datetime.now().isoformat(timespec='seconds'),
    }
    out = Path('results/online') / f'{session_id}_finetune'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False,
                                                 indent=2), encoding='utf-8')
    print(f'\n📁 Отчёт: {out / "report.json"}')


if __name__ == '__main__':
    main()
