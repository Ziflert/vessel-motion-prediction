"""
Оценка модели по РЕЖИМАМ ВОЛНЕНИЯ — главный содержательный анализ для исследования качки.

Разбивает окна данных на режимы по:
  1. высоте волны (Wave.Highest): calm / moderate / rough / severe;
  2. относительному углу встречи волны (Wave.direction - Course, круговая разность):
     following / quartering / beam / head seas.

Для каждого режима считает физические MAE/RMSE по целевым переменным.
Вывод: где модель работает, а где разваливается.

Использование:
  python scripts/evaluate_regimes.py --run-id production
  python scripts/evaluate_regimes.py --run-id v003 --rows 10624:12499   # holdout-подобный сегмент
  python scripts/evaluate_regimes.py --run-id <id> --stride 5           # прореживание окон
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

from data.features import engineer_features_dataframe, circular_diff
from run_training import load_data
import registry as reg

WAVE_HEIGHT_BINS = [
    ('calm (<0.5 м)', 0.0, 0.5),
    ('moderate (0.5-1.5 м)', 0.5, 1.5),
    ('rough (1.5-2.5 м)', 1.5, 2.5),
    ('severe (>2.5 м)', 2.5, 1e9),
]

# Границы |относительного угла встречи| в градусах
SEA_WAY_BINS = [
    ('following (<30°)', 0, 30),
    ('quartering (30-60°)', 30, 60),
    ('beam (60-120°)', 60, 120),
    ('quartering (120-150°)', 120, 150),
    ('head (>150°)', 150, 181),
]


def get_regime_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет служебные колонки режима: wave_h_bin, seaway_bin."""
    df = df.copy()
    if 'Wave.Highest(метры)' in df.columns:
        df['wave_h_bin'] = pd.cut(df['Wave.Highest(метры)'],
                                  bins=[b[1] for b in WAVE_HEIGHT_BINS] + [1e10],
                                  labels=[b[0] for b in WAVE_HEIGHT_BINS], right=False)
    else:
        df['wave_h_bin'] = 'n/a'

    # Относительный угол встречи: из готовых sin/cos, либо из сырых углов
    if 'Rel.Wave.angle(cos)' in df.columns and 'Rel.Wave.angle(sin)' in df.columns:
        rel = np.rad2deg(np.arctan2(df['Rel.Wave.angle(sin)'], df['Rel.Wave.angle(cos)']))
    elif 'Wave.direction(градусы)' in df.columns and 'Course(градусы)' in df.columns:
        rel = circular_diff(df['Wave.direction(градусы)'].values, df['Course(градусы)'].values)
    else:
        rel = None

    if rel is not None:
        df['seaway_bin'] = pd.cut(np.abs(rel), bins=[b[1] for b in SEA_WAY_BINS] + [1e9],
                                  labels=[b[0] for b in SEA_WAY_BINS], right=False)
    else:
        df['seaway_bin'] = 'n/a'
    return df


def predict_windows(model, config, feature_scaler, target_scaler,
                    df: pd.DataFrame, row_indices, device: str):
    """
    Прогоняет модель по окнам, начинающимся в указанных индексах строк.

    Returns:
        dict: target_name -> (mae, rmse, n), плюс общий mae.
        Физические единицы.
    """
    seq_len = config.sequence_length
    horizon = config.prediction_horizon

    feats = df[config.feature_columns].values.astype(np.float32)
    targets_cols = config.target_columns
    tgts_all = df[targets_cols].values.astype(np.float64)

    # Группируем индексы по режимам
    per_regime = {}

    xs = []
    meta = []
    for idx in row_indices:
        if idx + seq_len + horizon > len(df):
            continue
        xs.append(feats[idx:idx + seq_len])
        meta.append(idx)

    if not xs:
        return per_regime, 0

    # Батчим предсказания
    batch = config.batch_size if hasattr(config, 'batch_size') else 48
    preds_phys = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = np.stack(xs[i:i + batch])
            xb = feature_scaler.transform(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape)
            t = torch.from_numpy(xb.astype(np.float32)).to(device)
            pred = model(t, target=None, teacher_forcing_ratio=0.0)
            pred = pred.cpu().numpy()
            pred_phys = target_scaler.inverse_transform(pred.reshape(-1, pred.shape[-1]))
            preds_phys.append(pred_phys.reshape(pred.shape[0], horizon, -1))

    preds_phys = np.concatenate(preds_phys, axis=0)  # [N, horizon, n_targets]

    for j, idx in enumerate(meta):
        actual = tgts_all[idx + seq_len: idx + seq_len + horizon]  # [horizon, n_targets]
        err = np.abs(preds_phys[j] - actual)
        wave_bin = df['wave_h_bin'].iloc[idx]
        sea_bin = df['seaway_bin'].iloc[idx]
        for key in (f'wave:{wave_bin}', f'seaway:{sea_bin}', 'ALL'):
            d = per_regime.setdefault(key, {'sum_abs': 0.0, 'sum_sq': 0.0, 'n': 0,
                                            'per_target_abs': np.zeros(len(targets_cols))})
            d['sum_abs'] += err.sum()
            d['sum_sq'] += (err ** 2).sum()
            d['n'] += err.size
            d['per_target_abs'] += err.sum(axis=0)

    return per_regime, len(meta)


def format_report(per_regime, targets, n_windows, run_id):
    lines = [f'RUN: {run_id}', f'Windows evaluated: {n_windows}', '']
    for prefix, title in (('wave:', 'ПО ВЫСОТЕ ВОЛНЫ (Wave.Highest)'),
                          ('seaway:', 'ПО УГЛУ ВСТРЕЧИ ВОЛНЫ (относительный курс)'),
                          ('ALL', 'ОБЩЕЕ')):
        lines.append('=' * 78)
        lines.append(title)
        lines.append('=' * 78)
        sub = {k: v for k, v in per_regime.items() if k.startswith(prefix) or k == 'ALL'}
        for k in sorted(sub, key=lambda x: (x != 'ALL', x)):
            d = sub[k]
            mae = d['sum_abs'] / d['n']
            rmse = np.sqrt(d['sum_sq'] / d['n'])
            n_seq = d['n'] // len(targets)
            lines.append(f'{k.split(":", 1)[1]:28s} MAE={mae:8.4f}  RMSE={rmse:8.4f}  (n={n_seq} окон)')
            lines.append('    per-target MAE: ' +
                         ', '.join(f'{t.split("(")[0]}={d["per_target_abs"][i] / d["n"]:.4f}'
                                   for i, t in enumerate(targets)))
        lines.append('')
    return '\n'.join(lines)


def main():
    reg._force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Per-regime (sea state) evaluation')
    parser.add_argument('--run-id', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data/raw/your_data.csv')
    parser.add_argument('--rows', type=str, default=None,
                        help='Диапазон строк данных start:end (по умолчанию: test-диапазоны из manifest, иначе все)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Шаг между оцениваемыми окнами (1 = все окна, медленно)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    run_dir = reg.resolve_run(args.run_id)
    model, config, feature_scaler, target_scaler, manifest = reg.load_trained_model(run_dir)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f'Run: {run_dir.name} (profile: {manifest["features"].get("profile")})')
    print(f'Device: {device}')

    df = load_data(Path(args.data_path))
    eng = manifest.get('features', {}).get('feature_engineering')
    if eng:
        df = engineer_features_dataframe(df, cyclic=eng['cyclic_encoding'],
                                         relative_wave_angle=eng['relative_wave_angle'])
    df = get_regime_frames(df)

    # Какие строки оцениваем
    if args.rows:
        start, end = map(int, args.rows.split(':'))
        row_pool = list(range(start, min(end, len(df))))
        scope = f'rows {start}:{end}'
    elif manifest.get('data', {}).get('test_row_ranges'):
        row_pool = []
        for s, e in manifest['data']['test_row_ranges']:
            row_pool.extend(range(s, e))
        scope = 'test row ranges from manifest (честная оценка)'
    else:
        row_pool = list(range(0, len(df)))
        scope = 'ВСЕ строки (legacy-модель: test-границы неизвестны — метрики in-sample!)'
    row_pool = row_pool[::args.stride]
    print(f'Scope: {scope}, stride={args.stride}, windows={len(row_pool)}')

    per_regime, n_eval = predict_windows(model, config, feature_scaler, target_scaler,
                                         df, row_pool, device)

    report = format_report(per_regime, config.target_columns, n_eval, run_dir.name)
    print('\n' + report)

    # Сохранение
    out_dir = PROJECT_ROOT / 'results' / f'regimes/{run_dir.name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.txt').write_text(report, encoding='utf-8')

    rows = []
    for k, d in per_regime.items():
        row = {'regime': k, 'mae': d['sum_abs'] / d['n'],
               'rmse': (d['sum_sq'] / d['n']) ** 0.5, 'n_values': d['n']}
        for i, t in enumerate(config.target_columns):
            row[f'mae_{t}'] = d['per_target_abs'][i] / d['n']
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / 'regimes.csv', index=False)
    print(f'\n✓ Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
