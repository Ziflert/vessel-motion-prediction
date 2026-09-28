"""
Сравнение нескольких моделей на ОДИНАКОВЫХ окнах данных.

Раньше модели сравнивались по inference-запускам с разными индексами и
настройками — это было некорректно. Здесь все модели прогоняются по одной и
той же сетке окон и одинаковому горизонту, метрики в физических единицах.

Использование:
  python scripts/compare_models.py --all
  python scripts/compare_models.py --runs v003 v004
  python scripts/compare_models.py --runs production --horizon 15
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from data.features import engineer_features_dataframe
from run_training import load_data
import registry as reg


def evaluate_run(run_dir, df, window_starts, horizon, device):
    """Возвращает физические метрики модели на заданных окнах."""
    model, config, feature_scaler, target_scaler, manifest = reg.load_trained_model(run_dir)
    model = model.to(device)

    seq_len = config.sequence_length
    eff_horizon = min(horizon, config.prediction_horizon)

    feats = df[config.feature_columns].values.astype(np.float32)
    targets_all = df[config.target_columns].values.astype(np.float64)

    valid_starts = [s for s in window_starts if s + seq_len + eff_horizon <= len(df)]

    xs = [feats[s:s + seq_len] for s in valid_starts]
    batch = 128
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = np.stack(xs[i:i + batch])
            xb = feature_scaler.transform(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape)
            t = torch.from_numpy(xb.astype(np.float32)).to(device)
            pred = model(t, target=None, teacher_forcing_ratio=0.0).cpu().numpy()
            pred = pred[:, :eff_horizon, :]
            pred = target_scaler.inverse_transform(pred.reshape(-1, pred.shape[-1]))
            preds.append(pred.reshape(len(xs[i:i + batch]), eff_horizon, -1))
    preds = np.concatenate(preds, axis=0)

    targets = np.stack([targets_all[s + seq_len: s + seq_len + eff_horizon]
                        for s in valid_starts])

    per_target_mae = np.mean(np.abs(preds - targets), axis=(0, 1))
    overall_mae = float(np.mean(np.abs(preds - targets)))
    overall_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return {
        'run_id': run_dir.name,
        'profile': manifest['features'].get('profile', ''),
        'train_horizon': config.prediction_horizon,
        'eval_horizon': eff_horizon,
        'mae_overall': overall_mae,
        'rmse_overall': overall_rmse,
        'per_target_mae': dict(zip(config.target_columns,
                                   [round(float(v), 4) for v in per_target_mae])),
    }


def main():
    reg._force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Compare models on identical windows')
    parser.add_argument('--runs', nargs='*', default=None,
                        help='run_id (можно префиксы) или legacy-номера')
    parser.add_argument('--all', action='store_true', help='все модели из реестра')
    parser.add_argument('--data-path', type=str, default='./data/raw/your_data.csv')
    parser.add_argument('--num-windows', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=10,
                        help='Единый горизонт сравнения (обрезается под min с горизонтом модели)')
    parser.add_argument('--rows', type=str, default=None, help='Диапазон строк start:end')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.all:
        refs = [r['run_id'] for r in reg.list_runs()]
    elif args.runs:
        refs = args.runs
    else:
        parser.error('Укажите --runs id1 id2 ... или --all')

    run_dirs = [reg.resolve_run(str(r)) for r in refs]

    df = load_data(Path(args.data_path))
    # Инженерия применяется, если ХОТЬ одна модель её использует (проверим по каждой)
    any_eng = any(reg.load_manifest(d).get('features', {}).get('feature_engineering')
                  for d in run_dirs)
    if any_eng:
        # максимальный набор: применяем полную инженерию — модели без неё просто
        # не найдут свои колонки? НЕТ: сырые углы сохраняются при частичной инженерии,
        # поэтому для честности берём настройки каждой модели отдельно ниже.
        pass

    # Диапазон и сетка окон (ОДИНАКОВАЯ для всех моделей)
    seq_max = max(reg.config_from_manifest(reg.load_manifest(d)).sequence_length for d in run_dirs)
    if args.rows:
        start, end = map(int, args.rows.split(':'))
    else:
        start, end = seq_max, len(df)
    step = max(1, (end - start) // args.num_windows)
    window_starts = list(range(start, end, step))
    print(f'Окон: {len(window_starts)} на диапазоне [{start}:{end}], единый горизонт {args.horizon}')

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    results = []
    for run_dir in run_dirs:
        manifest = reg.load_manifest(run_dir)
        eng = manifest.get('features', {}).get('feature_engineering')
        df_run = engineer_features_dataframe(df, cyclic=eng['cyclic_encoding'],
                                             relative_wave_angle=eng['relative_wave_angle']) if eng else df
        try:
            r = evaluate_run(run_dir, df_run, window_starts, args.horizon, device)
            results.append(r)
            print(f"✓ {run_dir.name}: MAE={r['mae_overall']:.4f} "
                  f"(profile: {r['profile']}, train horizon {r['train_horizon']})")
        except Exception as e:
            print(f"✗ {run_dir.name}: {e}")

    if not results:
        print('Нет результатов')
        return

    # Таблица
    print('\n' + '=' * 90)
    print(f"СРАВНЕНИЕ НА ОДИНАКОВЫХ ОКНАХ (horizon={args.horizon})")
    print('=' * 90)
    all_targets = sorted({t for r in results for t in r['per_target_mae']})
    header = f"{'Run':32s} {'MAE':>8s} {'RMSE':>8s} | " + ' '.join(
        f'{t.split("(")[0][:9]:>9s}' for t in all_targets)
    print(header)
    print('-' * len(header))
    for r in sorted(results, key=lambda x: x['mae_overall']):
        row = f"{r['run_id'][:32]:32s} {r['mae_overall']:8.4f} {r['rmse_overall']:8.4f} | "
        row += ' '.join(f"{r['per_target_mae'].get(t, float('nan')):9.4f}" for t in all_targets)
        print(row)

    # Сохранение
    out_dir = PROJECT_ROOT / 'results' / f'compare_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = []
    for r in results:
        row = {'run_id': r['run_id'], 'profile': r['profile'],
               'mae_overall': r['mae_overall'], 'rmse_overall': r['rmse_overall']}
        row.update({f'mae_{t}': v for t, v in r['per_target_mae'].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / 'comparison.csv', index=False)
    print(f'\n✓ Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
