"""
N4 (IDEAS §C4): валидация MC-Dropout неопределённости production-модели.

ПОСТАНОВКА (сформулирована ДО прогона, 2026-09-29; скилы experimental-design +
hypothesis-generation; manifest — источник правды).

Исследовательский вопрос: отражает ли MC-Dropout-неопределённость production-модели
  (а) фактическую ошибку прогноза (predictive uncertainty ↔ error),
  (б) даёт ли калиброванные интервалы, пригодные как вход СППР «не верить прогнозу».

Гипотезы (candidates, до проверки):
  H-MC1 (predictive): corr(std_MC, |error|) на уровне окна > 0 на всех горизонтах
        и усиливается на больших упреждениях (где Ф4 показал падение skill).
  H-MC2 (calibration): покрытие интервала mean±1.96·std для качки (Roll/Pitch/Vertical)
        близко к номиналу 95% на test; в severe-режиме — проверка на OOD.
  Rival R1 (артефакт): std_MC отражает только стохастичность сети, не связанную
        с ошибкой → corr ≈ 0, плоская по горизонту.
  Rival R2 (потеря точности): MAE среднего MC-прогноза хуже детерминированного
        (усреднение разрушает нелинейное отображение) → MC непригоден как прогноз.
  Негативный контроль: corr(std_окна, ошибка СЛУЧАЙНОГО другого окна) (перестановки)
        должна collapse к ~0 против наблюдаемой same-window корреляции.

Дизайн:
  - Модель: production-прогон (--run-id, по умолчанию f3-roll_w4-k-9-seed42);
  - Данные: минимальный датасет, test row ranges из manifest (честная оценка);
  - Окна: stride 10 (основной анализ) + stride 140 = seq_len+horizon
    (НЕПЕРЕКРЫВАЮЩИЕСЯ окна — независимые единицы, защита от псевдорепликации:
    при stride 10 соседние окна делят 110 строк и corr-оценка завышает n);
    плюс per-segment агрегация (13 сегментов — честный блок);
  - MC-сэмплы: 30 (основной), свип {10, 30, 50} на подмножестве (устойчивость оценки);
  - Dropout модели: encoder 0.15 / temporal 0.10 / decoder 0.10 (config v2-канона).

Метрики: corr(Pearson+Spearman) std vs |error|; покрытие ±1σ/±1.96σ; ширина
интервалов по горизонту; MAE(mean_MC) vs MAE(deterministic); время окна (онлайн-физ.)

Использование:
  .venv\\Scripts\\python.exe scripts\\mc_dropout_eval.py
  .venv\\Scripts\\python.exe scripts\\mc_dropout_eval.py --run-id <id> --stride 10
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

from data.features import engineer_features_dataframe
from run_training import load_data
from scripts.evaluate_regimes import get_regime_frames, WAVE_HEIGHT_BINS
import registry as reg

HORIZON_COVER_Z = (1.0, 1.96)  # номинальные покрытия 68.3% / 95%


def _mc_predict_batched(model, feature_scaler, target_scaler, xs, mc_samples,
                        device, mc_chunk=None):
    """MC-Dropout батчами: окна реплицируются mc_samples раз, dropout независимо
    на каждый сэмпл (batchnorm в модели нет — урок predict_uncertain).

    Returns: mean [N, H, T] физ., std [N, H, T] физ.
    """
    mc_chunk = mc_chunk or mc_samples
    means, stds = [], []
    model.train()  # активирует dropout
    with torch.no_grad():
        for i in range(0, len(xs), mc_chunk):
            xb = np.stack(xs[i:i + mc_chunk])
            xb = feature_scaler.transform(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape)
            t = torch.from_numpy(xb.astype(np.float32)).to(device)
            # Реплицируем окна: [K, seq, feat] — каждый сэмпл со своей маской dropout
            t_rep = t.repeat_interleave(mc_samples, dim=0)
            pred = model(t_rep, target=None, teacher_forcing_ratio=0.0)
            pred = pred.cpu().numpy().reshape(len(xb), mc_samples, pred.shape[1], -1)
            mean_scaled = pred.mean(axis=1)
            std_scaled = pred.std(axis=1, ddof=1)
            means.append(target_scaler.inverse_transform(
                mean_scaled.reshape(-1, mean_scaled.shape[-1])).reshape(mean_scaled.shape))
            stds.append(std_scaled * target_scaler.scale_[None, None, :])
    model.eval()
    return np.concatenate(means, axis=0), np.concatenate(stds, axis=0)


def _det_predict_batched(model, feature_scaler, target_scaler, xs, device, batch=48):
    preds = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            xb = np.stack(xs[i:i + batch])
            xb = feature_scaler.transform(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape)
            t = torch.from_numpy(xb.astype(np.float32)).to(device)
            pred = model(t, target=None, teacher_forcing_ratio=0.0).cpu().numpy()
            preds.append(target_scaler.inverse_transform(
                pred.reshape(-1, pred.shape[-1])).reshape(pred.shape))
    return np.concatenate(preds, axis=0)


def _corr_with_pvalue(a, b, method='pearson'):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan, len(a)
    if method == 'pearson':
        r, p = sps.pearsonr(a, b)
    else:
        r, p = sps.spearmanr(a, b)
    return float(r), float(p), len(a)


def _negative_control(std_w, err_w, n_perm=200, seed=123):
    """corr(std_окна, ошибка СЛУЧАЙНОГО другого окна): перестановка ошибок.
    Распределение corr при H0 (нет связи) — против наблюдаемой same-window corr."""
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(err_w))
        r, _, _ = _corr_with_pvalue(std_w, err_w[perm])
        if np.isfinite(r):
            null.append(r)
    return np.asarray(null)


def main():
    reg._force_utf8_stdio()
    parser = argparse.ArgumentParser(description='N4: MC-Dropout validation')
    parser.add_argument('--run-id', type=str, default='20260929-042516-f3-roll_w4-k-9-seed42-3900',
                        help='Production-прогон (по умолчанию f3-roll_w4-k-9-seed42)')
    parser.add_argument('--data-path', type=str, default='./data/raw/your_data_minimal.csv')
    parser.add_argument('--stride', type=int, default=10,
                        help='Шаг окон основного анализа (1 = все окна; 140 = независимые)')
    parser.add_argument('--mc-samples', type=int, default=30)
    parser.add_argument('--samples-sweep', type=str, default='10,30,50',
                        help='Свип числа MC-сэмплов на подмножестве окон')
    parser.add_argument('--n-perm', type=int, default=200)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    run_dir = reg.resolve_run(args.run_id)
    model, config, feature_scaler, target_scaler, manifest = reg.load_trained_model(run_dir)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    targets = list(config.target_columns)
    seq_len, horizon = config.sequence_length, config.prediction_horizon
    n_t = len(targets)

    print(f'N4 MC-Dropout | Run: {run_dir.name} (status: {manifest.get("status")})')
    print(f'Device: {device}, mc_samples={args.mc_samples}, stride={args.stride}')

    df = load_data(Path(args.data_path))
    eng = manifest.get('features', {}).get('feature_engineering')
    if eng:
        df = engineer_features_dataframe(
            df,
            cyclic=eng.get('cyclic_encoding', True),
            relative_wave_angle=eng.get('relative_wave_angle', True),
            relative_wind_angle=eng.get('relative_wind_angle', False))
    df = get_regime_frames(df)

    # Test row ranges из manifest (честная оценка)
    row_pool = []
    for s, e in manifest['data']['test_row_ranges']:
        row_pool.extend(range(s, e))
    row_pool = sorted(set(row_pool))

    # Окна
    xs, meta = [], []
    for idx in row_pool:
        if idx + seq_len + horizon <= len(df):
            xs.append(df[config.feature_columns].values.astype(np.float32)[idx:idx + seq_len])
            meta.append(idx)
    xs_strided = xs[::args.stride]
    meta_strided = meta[::args.stride]
    n_w = len(xs_strided)
    print(f'Windows: {n_w} (из {len(xs)} возможных, stride={args.stride})')

    tgts_all = df[targets].values.astype(np.float64)
    actual = np.stack([tgts_all[i + seq_len: i + seq_len + horizon] for i in meta_strided])
    wave_bins = [df['wave_h_bin'].iloc[i] for i in meta_strided]

    # ------------------------------------------------------------------
    # 1. Детерминированный прогноз (R2: MAE(mean_MC) vs MAE(det))
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    preds_det = _det_predict_batched(model, feature_scaler, target_scaler, xs_strided, device)
    det_ms_per_window = (time.monotonic() - t0) * 1000.0 / n_w

    # ------------------------------------------------------------------
    # 2. MC-Dropout (основной прогон)
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    mean_mc, std_mc = _mc_predict_batched(model, feature_scaler, target_scaler,
                                          xs_strided, args.mc_samples, device)
    mc_ms_per_window = (time.monotonic() - t0) * 1000.0 / n_w
    print(f'Runtime: det {det_ms_per_window:.1f} мс/окно, mc {mc_ms_per_window:.1f} мс/окно')

    err_mc = np.abs(mean_mc - actual)      # [N, H, T]
    err_det = np.abs(preds_det - actual)   # [N, H, T]

    # ------------------------------------------------------------------
    # 3. H-MC1: corr(std_MC, |error|) — по горизонту и по окну
    # ------------------------------------------------------------------
    per_horizon = []
    for h in range(horizon):
        row = {'horizon_step': h + 1}
        for j, t in enumerate(targets):
            r_p, p_p, n1 = _corr_with_pvalue(std_mc[:, h, j], err_mc[:, h, j], 'pearson')
            r_s, p_s, _ = _corr_with_pvalue(std_mc[:, h, j], err_mc[:, h, j], 'spearman')
            row.update({
                f'mae_{t}': float(err_mc[:, h, j].mean()),
                f'std_mean_{t}': float(std_mc[:, h, j].mean()),
                f'corr_p_{t}': r_p, f'p_p_{t}': p_p,
                f'corr_s_{t}': r_s, f'p_s_{t}': p_s,
                f'cover1_{t}': float((err_mc[:, h, j] <= HORIZON_COVER_Z[0] * std_mc[:, h, j]).mean()),
                f'cover196_{t}': float((err_mc[:, h, j] <= HORIZON_COVER_Z[1] * std_mc[:, h, j]).mean()),
            })
        per_horizon.append(row)
    df_horizon = pd.DataFrame(per_horizon)

    # corr на уровне ОКНА (усреднение по горизонту): std окна vs |error| окна
    window_rows = []
    for i in range(n_w):
        row = {'window_idx': meta_strided[i], 'wave_bin': str(wave_bins[i])}
        for j, t in enumerate(targets):
            row[f'std_mean_{t}'] = float(std_mc[i, :, j].mean())
            row[f'err_mean_{t}'] = float(err_mc[i, :, j].mean())
            row[f'err_mean_det_{t}'] = float(err_det[i, :, j].mean())
        window_rows.append(row)
    df_windows = pd.DataFrame(window_rows)

    window_corr = {}
    for j, t in enumerate(targets):
        r_p, p_p, n1 = _corr_with_pvalue(df_windows[f'std_mean_{t}'], df_windows[f'err_mean_{t}'], 'pearson')
        r_s, p_s, _ = _corr_with_pvalue(df_windows[f'std_mean_{t}'], df_windows[f'err_mean_{t}'], 'spearman')
        window_corr[t] = {'pearson': r_p, 'p_pearson': p_p, 'spearman': r_s,
                          'p_spearman': p_s, 'n': n1}

    # Per-segment агрегация (сегменты манифеста — независимые блоки против
    # псевдорепликации; маппинг строки → сегмент по test_row_ranges, НЕ m//150:
    # несмежные 150-строчные сегменты маппингом m//150 расщеплялись бы на 2 бина)
    ranges = manifest['data']['test_row_ranges']
    range_bounds = [(s, e) for s, e in ranges]
    def _seg_of(row_idx):
        for k, (s, e) in enumerate(range_bounds):
            if s <= row_idx < e:
                return k
        return -1
    seg_ids = [_seg_of(m) for m in meta_strided]
    df_windows['segment'] = seg_ids
    seg_corr = {}
    seg_grp = df_windows.groupby('segment')
    if seg_grp.ngroups >= 3:
        for j, t in enumerate(targets):
            seg_std = seg_grp[f'std_mean_{t}'].mean()
            seg_err = seg_grp[f'err_mean_{t}'].mean()
            r_p, p_p, n1 = _corr_with_pvalue(seg_std, seg_err, 'pearson')
            seg_corr[t] = {'pearson': r_p, 'p_pearson': p_p, 'n_segments': n1}

    # Негативный контроль (перестановки) — по окну, для качки + общий
    neg_ctrl = {}
    for t in targets + ['ALL']:
        if t == 'ALL':
            std_w = df_windows[[f'std_mean_{c}' for c in targets]].mean(axis=1).values
            err_w = df_windows[[f'err_mean_{c}' for c in targets]].mean(axis=1).values
        else:
            std_w = df_windows[f'std_mean_{t}'].values
            err_w = df_windows[f'err_mean_{t}'].values
        r_obs, _, _ = _corr_with_pvalue(std_w, err_w)
        null = _negative_control(std_w, err_w, n_perm=args.n_perm)
        p_emp = float((np.abs(null) >= np.abs(r_obs)).mean()) if len(null) else np.nan
        neg_ctrl[t] = {'corr_observed': r_obs, 'null_mean': float(null.mean()) if len(null) else np.nan,
                       'null_max_abs': float(np.abs(null).max()) if len(null) else np.nan,
                       'p_empirical': p_emp}

    # ------------------------------------------------------------------
    # 4. H-MC2: покрытие по режимам (wave bins)
    # ------------------------------------------------------------------
    regime_rows = []
    for label, lo, hi in WAVE_HEIGHT_BINS:
        mask = np.array([(lo <= df['Wave.Highest(метры)'].iloc[i] < hi) for i in meta_strided]) \
            if 'Wave.Highest(метры)' in df.columns else np.ones(n_w, dtype=bool)
        if mask.sum() == 0:
            continue
        row = {'regime': label, 'n_windows': int(mask.sum())}
        for j, t in enumerate(targets):
            e, s = err_mc[mask][:, :, j], std_mc[mask][:, :, j]
            row[f'mae_{t}'] = float(e.mean())
            row[f'std_mean_{t}'] = float(s.mean())
            row[f'cover1_{t}'] = float((e <= HORIZON_COVER_Z[0] * s).mean())
            row[f'cover196_{t}'] = float((e <= HORIZON_COVER_Z[1] * s).mean())
        regime_rows.append(row)
    df_regime = pd.DataFrame(regime_rows)

    # ------------------------------------------------------------------
    # 5. Свип числа MC-сэмплов (подмножество окон: каждый 3-й)
    # ------------------------------------------------------------------
    sweep_idx = list(range(0, n_w, 3))
    xs_sub = [xs_strided[i] for i in sweep_idx]
    sweep_rows = []
    for k in [int(x) for x in args.samples_sweep.split(',')]:
        t0 = time.monotonic()
        m_k, s_k = _mc_predict_batched(model, feature_scaler, target_scaler,
                                       xs_sub, k, device)
        ms = (time.monotonic() - t0) * 1000.0 / len(xs_sub)
        # устойчивость оценки std: сравнение с 50-сэмпловым референсом позже;
        # здесь — разброс std между окнами и MAE среднего
        row = {'mc_samples': k, 'ms_per_window': ms}
        for j, t in enumerate(targets):
            row[f'std_mean_{t}'] = float(s_k[:, :, j].mean())
            row[f'std_std_{t}'] = float(s_k[:, :, j].std())
            row[f'mae_{t}'] = float(np.abs(m_k[:, :, j] - actual[sweep_idx][:, :, j]).mean())
        sweep_rows.append(row)
    df_sweep = pd.DataFrame(sweep_rows)

    # ------------------------------------------------------------------
    # Отчёт
    # ------------------------------------------------------------------
    L = [f'N4 MC-Dropout — валидация неопределённости production-модели',
         f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
         f'Run: {run_dir.name} (status: {manifest.get("status")})',
         f'Device: {device}; mc_samples={args.mc_samples}; stride={args.stride}; окон={n_w}',
         f'Постановка: гипотезы H-MC1 (corr>0), H-MC2 (покрытие≈номинал),',
         f'  rival R1 (corr≈0), R2 (MAE_mean хуже det), негативный контроль (перестановки).',
         f'Псевдорепликация: окна stride=10 перекрываются (соседние делят 110 строк) —',
         f'  честная единица уровня блока: per-segment агрегация (13 сегментов);',
         f'  полностью независимые окна доступны через --stride 140 (seq_len+horizon).',
         '',
         '=' * 78,
         'RUNTIME (онлайн-физibilidad: watchdog inference_timeout_s)',
         '=' * 78,
         f'Детерминированный: {det_ms_per_window:.1f} мс/окно',
         f'MC ({args.mc_samples} сэмплов): {mc_ms_per_window:.1f} мс/окно '
         f'(×{mc_ms_per_window / max(det_ms_per_window, 1e-9):.1f})',
         '',
         '=' * 78,
         'R2: MAE(mean_MC) vs MAE(deterministic) — усреднение не должно портить прогноз',
         '=' * 78]
    for j, t in enumerate(targets):
        L.append(f'{t.split("(")[0]:12s} MAE_det={err_det[:, :, j].mean():8.4f}  '
                 f'MAE_mc={err_mc[:, :, j].mean():8.4f}  '
                 f'Δ={err_mc[:, :, j].mean() - err_det[:, :, j].mean():+.4f}')
    L += ['',
          '=' * 78,
          'H-MC1: corr(std_MC, |error|) на уровне ОКНА (усреднение по горизонту 1-20)',
          '=' * 78]
    for t, d in window_corr.items():
        L.append(f'{t.split("(")[0]:12s} Pearson r={d["pearson"]:+.3f} (p={d["p_pearson"]:.2e})  '
                 f'Spearman ρ={d["spearman"]:+.3f} (p={d["p_spearman"]:.2e})  n={d["n"]}')
    if seg_corr:
        L += ['', f'Per-SEGMENT агрегация (сегменты манифеста — независимые блоки):']
        for t, d in seg_corr.items():
            L.append(f'{t.split("(")[0]:12s} Pearson r={d["pearson"]:+.3f} '
                     f'(p={d["p_pearson"]:.3f})  n_segments={d["n_segments"]}')
    L += ['',
          '=' * 78,
          'НЕГАТИВНЫЙ КОНТРОЛЬ: corr(std_окна, ошибка СЛУЧАЙНОГО другого окна)',
          '=' * 78]
    for t, d in neg_ctrl.items():
        L.append(f'{t.split("(")[0]:12s} r_obs={d["corr_observed"]:+.3f}  '
                 f'H0: mean={d["null_mean"]:+.3f}, max|.|={d["null_max_abs"]:.3f}, '
                 f'p_emp={d["p_empirical"]:.3f}')
    L += ['',
          '=' * 78,
          'H-MC1 по горизонту: corr(std_MC, |error|) и покрытие по шагам (1-20)',
          '=' * 78,
          'Шаг | ' + ' | '.join(f'{t.split("(")[0]}: r_p / cover196' for t in targets)]
    for h in range(horizon):
        row = df_horizon.iloc[h]
        cells = []
        for t in targets:
            cells.append(f'{row[f"corr_p_{t}"]:+.2f} / {row[f"cover196_{t}"]:.2f}')
        L.append(f'{h + 1:3d} | ' + ' | '.join(cells))
    L += ['',
          '=' * 78,
          'H-MC2: покрытие интервалов по РЕЖИМАМ ВОЛНЕНИЯ (±1σ / ±1.96σ)',
          '=' * 78]
    for _, row in df_regime.iterrows():
        L.append(f'{row["regime"]:18s} (n={row["n_windows"]} окон)')
        for t in targets:
            L.append(f'    {t.split("(")[0]:10s} MAE={row[f"mae_{t}"]:7.4f}  '
                     f'std_mean={row[f"std_mean_{t}"]:7.4f}  '
                     f'cover1σ={row[f"cover1_{t}"]:.2f}  cover1.96σ={row[f"cover196_{t}"]:.2f}')
    L += ['',
          '=' * 78,
          'Свип числа MC-сэмплов (подмножество окон): стоимость/устойчивость',
          '=' * 78]
    for _, row in df_sweep.iterrows():
        L.append(f"K={int(row['mc_samples']):3d}  {row['ms_per_window']:8.1f} мс/окно  "
                 + '  '.join(f"{t.split('(')[0]}: std_mean={row[f'std_mean_{t}']:.4f}, MAE={row[f'mae_{t}']:.4f}"
                             for t in targets))
    report = '\n'.join(L)
    print('\n' + report)

    # ------------------------------------------------------------------
    # Сохранение
    # ------------------------------------------------------------------
    out_dir = PROJECT_ROOT / 'results' / 'mc_dropout_n4'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'report.txt').write_text(report, encoding='utf-8')
    df_horizon.to_csv(out_dir / 'per_horizon.csv', index=False)
    df_regime.to_csv(out_dir / 'coverage_by_regime.csv', index=False)
    df_windows.to_csv(out_dir / 'windows_raw.csv', index=False)
    df_sweep.to_csv(out_dir / 'samples_sweep.csv', index=False)
    print(f'\n✓ Сохранено: {out_dir}')


if __name__ == '__main__':
    main()
