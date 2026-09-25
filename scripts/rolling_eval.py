"""
Rolling-прогноз и спектральный анализ качки.

Зачем: одиночные прогнозы (как в run_inference) не позволяют оценить ДИНАМИКУ.
Этот скрипт скользит по сегменту данных с шагом 1 секунда, строит НЕПРЕРЫВНЫЙ
ряд прогноза на выбранном упреждении (lead time) и анализирует его:

  1. MAE/RMSE/корреляция прогноза vs факт (физические единицы);
  2. сравнение с persistence-бейзлайном (skill score);
  3. СПЕКТРАЛЬНЫЙ анализ: периодограмма (Уэлч-подобное усреднение) прогноза
     и факта — совпали ли частоты качки (резонансы);
  4. ФАЗОВЫЙ сдвиг: взаимная корреляция прогноза и факта — «опережает/запаздывает»
     ли модель качание (в секундах);
  5. амплитудный коэффициент (std_pred / std_actual).

Использование:
  python scripts/rolling_eval.py --run-id v003 --lead 5
  python scripts/rolling_eval.py --run-id production --rows 11000:12499 --lead 1
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


def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 256):
    """Уэлч-оценка спектра (без scipy): сегменты с перекрытием 50%, окно Ханна."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    step = nperseg // 2
    win = np.hanning(nperseg)
    segs = []
    for start in range(0, len(x) - nperseg + 1, step):
        segs.append(x[start:start + nperseg] * win)
    if not segs:
        return np.array([0.0]), np.array([0.0])
    segs = np.array(segs)
    fft = np.fft.rfft(segs, axis=1)
    psd = (np.abs(fft) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    norm = 1.0 / (fs * (win ** 2).sum())
    psd = psd * 2 * norm
    psd[0] /= 2
    if nperseg % 2 == 0:
        psd[-1] /= 2
    return freqs, psd


def phase_lag(pred: np.ndarray, actual: np.ndarray, fs: float = 1.0, max_lag: int = 30):
    """Фазовый сдвиг (в секундах) через взаимную корреляцию: >0 — прогноз запаздывает."""
    p = pred - pred.mean()
    a = actual - actual.mean()
    n = len(p)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag >= 0:
            c = np.corrcoef(p[lag:], a[:n - lag])[0, 1] if lag < n else 0.0
        else:
            c = np.corrcoef(p[:n + lag], a[-lag:])[0, 1] if -lag < n else 0.0
        corr.append(c if np.isfinite(c) else 0.0)
    corr = np.array(corr)
    best = lags[np.argmax(corr)]
    return float(best) / fs, float(corr.max()), lags, corr


def main():
    reg._force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Rolling forecast + spectral analysis')
    parser.add_argument('--run-id', type=str, required=True)
    parser.add_argument('--data-path', type=str, default='./data/raw/your_data.csv')
    parser.add_argument('--rows', type=str, default=None,
                        help='Диапазон строк start:end (по умолчанию — тестовые диапазоны manifest, иначе последние 2000)')
    parser.add_argument('--lead', type=int, default=1,
                        help='Упреждение в шагах (1 = прогноз на следующую секунду)')
    parser.add_argument('--target', type=str, default=None,
                        help='Целевая переменная для спектрального анализа (по умолчанию Roll, если есть)')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    run_dir = reg.resolve_run(args.run_id)
    model, config, feature_scaler, target_scaler, manifest = reg.load_trained_model(run_dir)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    target_name = args.target or next(
        (t for t in config.target_columns if t.startswith('Roll')), config.target_columns[0])
    t_idx = config.target_columns.index(target_name)
    t_in_features = target_name in config.feature_columns
    t_feat_idx = config.feature_columns.index(target_name) if t_in_features else None

    print(f'Run: {run_dir.name} | target: {target_name} | lead: {args.lead} шаг(ов)')

    df = load_data(Path(args.data_path))
    eng = manifest.get('features', {}).get('feature_engineering')
    if eng:
        df = engineer_features_dataframe(df, cyclic=eng['cyclic_encoding'],
                                         relative_wave_angle=eng['relative_wave_angle'])

    # Диапазон строк
    if args.rows:
        start, end = map(int, args.rows.split(':'))
    elif manifest.get('data', {}).get('test_row_ranges'):
        ranges = manifest['data']['test_row_ranges']
        start, end = ranges[-1]  # самый поздний тестовый сегмент
        print('Использую последний тестовый сегмент из manifest')
    else:
        start, end = max(0, len(df) - 2000), len(df)
        print('Диапазон не задан, беру последние 2000 строк (legacy: in-sample!)')
    end = min(end, len(df))

    seq_len = config.sequence_length
    horizon = config.prediction_horizon
    lead = min(args.lead, horizon)
    feats = df[config.feature_columns].values.astype(np.float32)
    targets_all = df[config.target_columns].values.astype(np.float64)

    # Непрерывный rolling-прогноз
    starts = np.arange(start, end - seq_len - lead)
    n = len(starts)
    print(f'Rolling: {n} прогнозов на сегменте [{start}:{end}]...')

    batch = 256
    pred_series = np.full(n, np.nan)
    actual_series = targets_all[starts + seq_len + lead - 1, t_idx]
    persist_series = np.full(n, np.nan)

    with torch.no_grad():
        for i in range(0, n, batch):
            idx = starts[i:i + batch]
            xb = np.stack([feats[s:s + seq_len] for s in idx])
            xb = feature_scaler.transform(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape)
            t = torch.from_numpy(xb.astype(np.float32)).to(device)
            pred = model(t, target=None, teacher_forcing_ratio=0.0).cpu().numpy()
            pred = target_scaler.inverse_transform(pred.reshape(-1, pred.shape[-1]))
            pred = pred.reshape(len(idx), horizon, -1)
            pred_series[i:i + len(idx)] = pred[:, lead - 1, t_idx]
            # persistence: последнее наблюдённое значение цели
            if t_in_features:
                persist_series[i:i + len(idx)] = feats[idx + seq_len - 1, t_feat_idx]

    valid = np.isfinite(pred_series)
    p, a, b = pred_series[valid], actual_series[valid], persist_series[valid]

    mae = np.mean(np.abs(p - a))
    rmse = np.sqrt(np.mean((p - a) ** 2))
    corr = np.corrcoef(p, a)[0, 1]
    p_mae = np.mean(np.abs(b - a))
    skill = 1 - mae / p_mae if p_mae > 0 else float('nan')

    # Спектры и фаза
    fs = 1.0
    f_a, psd_a = welch_psd(a, fs)
    f_p, psd_p = welch_psd(p, fs)
    lag_s, max_corr, lags, corr_curve = phase_lag(p, a, fs)
    amp_ratio = p.std() / a.std() if a.std() > 0 else float('nan')

    # Пиковая частота (главный период качки)
    peak_a = f_a[np.argmax(psd_a[1:]) + 1] if len(f_a) > 1 else 0
    peak_p = f_p[np.argmax(psd_p[1:]) + 1] if len(f_p) > 1 else 0

    report = [
        '=' * 70, 'ROLLING FORECAST REPORT', '=' * 70,
        f'Run: {run_dir.name}',
        f'Target: {target_name} | lead: {lead} шаг(ов) | сегмент: [{start}:{end}] ({n} точек)',
        '',
        f'MAE:  {mae:.4f}   RMSE: {rmse:.4f}   Corr: {corr:.3f}',
        f'Persistence MAE: {p_mae:.4f}   Skill score: {skill:+.3f}',
        f'Амплитудный коэффициент (std_pred/std_actual): {amp_ratio:.3f}',
        '',
        'СПЕКТР:',
        f'  Пиковая частота: факт {peak_a:.4f} Гц (период {1 / peak_a:.1f} с) | '
        f'прогноз {peak_p:.4f} Гц (период {1 / peak_p:.1f} с)',
        f'ФАЗА:',
        f'  Фазовый сдвиг: {lag_s:+.1f} с (прогноз {"запаздывает" if lag_s > 0 else "опережает" if lag_s < 0 else "совпадает"})',
        f'  Максимум взаимной корреляции: {max_corr:.3f}',
        '=' * 70,
    ]
    report_text = '\n'.join(report)
    print('\n' + report_text)

    # Сохранение
    out_dir = PROJECT_ROOT / 'results' / f'rolling_{run_dir.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.txt').write_text(report_text, encoding='utf-8')

    pd.DataFrame({
        'time_idx': starts[valid] + seq_len + lead - 1,
        'actual': a, 'prediction': p, 'persistence': b,
    }).to_csv(out_dir / 'rolling_series.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    show = min(len(a), 600)
    axes[0].plot(a[:show], 'r-', alpha=0.7, label='Actual')
    axes[0].plot(p[:show], 'b-', alpha=0.7, label=f'Prediction (lead {lead}s)')
    axes[0].plot(b[:show], 'g--', alpha=0.4, label='Persistence')
    axes[0].set_title(f'{target_name}: rolling forecast, lead={lead}s '
                      f'(MAE={mae:.3f}, skill={skill:+.2f})')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(a[-show:], 'r-', alpha=0.7, label='Actual')
    axes[1].plot(p[-show:], 'b-', alpha=0.7, label='Prediction')
    axes[1].set_title('Последний сегмент (крупно)')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].semilogy(f_a, psd_a, 'r-', alpha=0.7, label='Actual PSD')
    axes[2].semilogy(f_p, psd_p, 'b-', alpha=0.7, label='Prediction PSD')
    axes[2].set_xlabel('Частота, Гц'); axes[2].set_ylabel('PSD')
    axes[2].set_title(f'Спектры (пик факта: T={1 / peak_a:.1f} с)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'rolling_analysis.png', dpi=150)
    plt.close()
    print(f'✓ Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
