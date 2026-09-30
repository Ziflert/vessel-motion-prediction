"""
CLI режима A (playback): гоняет CSV через PredictionEngine как «онлайн»,
1 строка/сек, и пишет сессию в results/online/<session_id>/.

Запуск (из корня проекта):
    .venv/Scripts/python.exe -m online.playback --model <run-id|path> --csv <csv>
        [--speed 1|5|max] [--limit N] [--start-row N]

Артефакты сессии:
    session.csv     — raw-строки + tick + служебные колонки
    forecasts.csv   — прогнозы: одна строка на (tick, horizon)
    summary.txt     — реализованная ошибка «факт vs прогноз», тайминги
    events.log      — таймауты, пропуски, ошибки (план §9)
    config_snapshot.json — вся конфигурация запуска
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import registry as reg
import numpy as np
from run_inference import VesselPredictor_Inference

from config.online import OnlineConfig
from online.engine import PredictionEngine, EngineStalledError
from online.sources import TestSource, load_raw_csv, extract_fe_params


def force_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser(description='Онлайн-playback CSV через обученную модель')
    p.add_argument('--model', required=True,
                   help='run-id (или префикс, или "latest"/"production") либо путь к run-директории')
    p.add_argument('--csv', required=True, help='CSV с данными (raw-схема train)')
    p.add_argument('--speed', default='1',
                   help='множитель скорости: 1, 5, 20, 100 или "max" (без пауз)')
    p.add_argument('--limit', type=int, default=None, help='максимум строк (bound, план §7)')
    p.add_argument('--start-row', type=int, default=0, help='пропустить первые N строк CSV')
    return p.parse_args()


def resolve_model_dir(ref: str) -> Path:
    """ref — run-id/префикс/latest/production или прямой путь к директории."""
    path = Path(ref)
    if path.exists():
        return path.resolve()
    return reg.resolve_run(ref)


def main():
    force_utf8_stdio()
    args = parse_args()
    cfg = OnlineConfig()

    # --- модель ------------------------------------------------------------
    run_dir = resolve_model_dir(args.model)
    manifest = reg.load_manifest(run_dir)
    if manifest is None:
        raise SystemExit(f"У run {run_dir.name} нет manifest.json — не подходит для онлайн")
    checkpoint = run_dir / 'checkpoints' / 'best_model.pt'
    scalers = run_dir / 'checkpoints' / 'scalers.pkl'
    if not (checkpoint.exists() and scalers.exists()):
        raise SystemExit(f"В {run_dir} нет checkpoints/best_model.pt или scalers.pkl")

    predictor = VesselPredictor_Inference(str(checkpoint), str(scalers))
    fe_params = extract_fe_params(manifest)

    # --- данные ------------------------------------------------------------
    df_raw = load_raw_csv(args.csv)
    missing = [c for c in predictor.config.target_columns if c not in df_raw.columns]
    # Признаки после инженерии: их нет в raw-CSV — их наличие проверим на probe-окне ниже
    from data.features import engineer_features_dataframe
    df_probe = engineer_features_dataframe(
        df_raw.head(cfg.buffer_multiplier * max(predictor.config.sequence_length, 600)),
        cyclic=fe_params['cyclic'],
        relative_wave_angle=fe_params['relative_wave_angle'],
        relative_wind_angle=fe_params.get('relative_wind_angle', False),
    )
    missing += [c for c in predictor.config.feature_columns if c not in df_probe.columns]
    if missing:
        raise SystemExit(f"CSV не содержит колонок модели: {missing}")
    if not df_raw[predictor.config.target_columns].notna().all(axis=None):
        print("⚠ В CSV есть NaN в целевых колонках — эти tick'и будут пропущены прогнозом")

    # --- сессия ------------------------------------------------------------
    speed_str = args.speed.lower()
    speed = 0.0 if speed_str == 'max' else float(speed_str)
    if speed_str != 'max' and speed < 0:
        raise SystemExit("--speed должен быть > 0 или 'max'")

    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_playback'
    out_dir = PROJECT_ROOT / 'results' / 'online' / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    events_path = out_dir / 'events.log'
    config_snapshot = {
        'session_id': session_id,
        'mode': 'playback',
        'model_run_id': manifest.get('run_id', run_dir.name),
        'model_dir': str(run_dir),
        'feature_engineering': fe_params,
        'csv': str(Path(args.csv).resolve()),
        'speed': speed_str,
        'start_row': args.start_row,
        'limit': args.limit,
        'sequence_length': predictor.config.sequence_length,
        'prediction_horizon': predictor.config.prediction_horizon,
        'targets': predictor.config.target_columns,
        'online_config': cfg.to_dict(),
    }
    (out_dir / 'config_snapshot.json').write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding='utf-8')

    def log_event(level: str, msg: str):
        ts = datetime.now().isoformat(timespec='seconds')
        with open(events_path, 'a', encoding='utf-8') as f:
            f.write(f"{ts} | {level} | {msg}\n")

    source = TestSource(df_raw, speed=speed, limit=args.limit,
                         start_row=args.start_row, tick_seconds=cfg.tick_seconds)
    engine = PredictionEngine(predictor, fe_params, cfg)

    print(f"▶ Playback: {source.n_rows} строк, speed={speed_str or '1'}, "
          f"окно={predictor.config.sequence_length}, горизонт={predictor.config.prediction_horizon}")

    # --- tick-цикл ---------------------------------------------------------
    raw_columns = list(df_raw.columns)
    session_path = out_dir / 'session.csv'
    forecasts_path = out_dir / 'forecasts.csv'

    stats = {'warmup': 0, 'predicted': 0, 'nan_window': 0, 'timeout': 0,
             'input_anomaly_ticks': 0, 'error_anomaly_ticks': 0}
    prev_input_marker = False
    prev_error_marker = False
    inference_ms_all = []
    t_start = time.monotonic()

    try:
        with open(session_path, 'w', encoding='utf-8', newline='') as f_session, \
             open(forecasts_path, 'w', encoding='utf-8', newline='') as f_fore:
            import csv as csv_mod
            session_writer = csv_mod.writer(f_session)
            session_writer.writerow(['tick', 'predicted', 'input_anomaly', 'input_anomaly_features',
                                      'error_anomaly', 'error_ratio'] + raw_columns)
            fore_writer = csv_mod.writer(f_fore)
            fore_writer.writerow(['tick', 'horizon_step'] +
                                  [f'pred_{c}' for c in predictor.config.target_columns])

            for tick, row in source:
                result = engine.on_tick(tick, row)

                # Сверка факт/прогноз ДО записи: маркер ошибки актуален на этот tick
                try:
                    engine.observe_actual(tick, row)
                except (KeyError, TypeError, ValueError) as e:
                    log_event('ERROR', f"tick {tick}: observe_actual: {e}")

                # Запись raw-строки + статус tick'а + маркеры (план §6)
                session_writer.writerow(
                    [tick, int(result.predicted), int(result.input_anomaly),
                     result.input_anomaly_features, int(engine.last_error_marker),
                     '' if engine.last_error_ratio is None
                        else f'{engine.last_error_ratio:.3f}']
                    + [row.get(c, '') for c in raw_columns])

                if result.warmup:
                    stats['warmup'] += 1
                elif result.predicted:
                    stats['predicted'] += 1
                    inference_ms_all.append(result.inference_ms)
                    for h in range(engine.horizon):
                        fore_writer.writerow([tick, h + 1] +
                                              [round(float(v), 6) for v in result.preds[h]])
                elif result.skipped_reason == 'nan_window':
                    stats['nan_window'] += 1
                    log_event('WARN', f"tick {tick}: окно с NaN — прогноз пропущен")
                elif result.skipped_reason == 'timeout':
                    stats['timeout'] += 1
                    log_event('WARN', f"tick {tick}: таймаут inference, прогноз пропущен")

                # События маркеров: логируем только ПЕРЕКЛЮЧЕНИЯ статуса (не каждый tick)
                if result.input_anomaly and not prev_input_marker:
                    log_event('MARK',
                              f"tick {tick}: INPUT-анонималия: {result.input_anomaly_features or '?'}")
                if engine.last_error_marker and not prev_error_marker:
                    log_event('MARK', f"tick {tick}: ERROR-анонималия "
                              f"(ratio={engine.last_error_ratio:.2f})" if engine.last_error_ratio
                              else f"tick {tick}: ERROR-анонималия")
                if result.input_anomaly:
                    stats['input_anomaly_ticks'] += 1
                if engine.last_error_marker:
                    stats['error_anomaly_ticks'] += 1
                prev_input_marker = result.input_anomaly
                prev_error_marker = engine.last_error_marker

                if (tick + 1) % 60 == 0:
                    print(f"  tick {tick + 1}/{source.n_rows} "
                          f"(прогноз: {stats['predicted']})")
    except EngineStalledError as e:
        log_event('ERROR', f"PLAYBACK ОСТАНОВЛЕН: {e}")
        print(f"\n⛔ {e}")
        print("Источник остановлен по правилу плана §7 (stale_pause_threshold).")
    finally:
        wall_s = time.monotonic() - t_start

    # --- summary -----------------------------------------------------------
    err = engine.error_stats()
    summary = []
    summary.append('=' * 72)
    summary.append(f'ONLINE PLAYBACK SUMMARY — {session_id}')
    summary.append('=' * 72)
    summary.append(f'Модель:            {manifest.get("run_id", run_dir.name)} '
                   f'(status: {manifest.get("status", "n/a")})')
    summary.append(f'CSV:               {args.csv}')
    summary.append(f'Строк обработано:  {stats["warmup"] + stats["predicted"] + stats["nan_window"] + stats["timeout"]}')
    summary.append(f'  прогрев:         {stats["warmup"]}')
    summary.append(f'  прогнозов:       {stats["predicted"]}')
    summary.append(f'  пропуски NaN:    {stats["nan_window"]}')
    summary.append(f'  таймауты:        {stats["timeout"]}')
    summary.append(f'Время сессии:      {wall_s:.1f} с')
    summary.append('')
    summary.append('Маркеры аномалий (план §6):')
    summary.append(f'  вход (input_anomaly):    {stats["input_anomaly_ticks"]} tick\'ов')
    summary.append(f'  ошибка (error_anomaly):  {stats["error_anomaly_ticks"]} tick\'ов')
    if not engine.error_monitor_enabled:
        summary.append('  ⚠ error_anomaly: в manifest нет per-target MAE — монитор отключён')

    if inference_ms_all:
        ms = sorted(inference_ms_all)
        p50 = ms[len(ms) // 2]
        p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
        summary.append(f'Inference, мс:     p50={p50:.0f} p95={p95:.0f} '
                        f'max={ms[-1]:.0f} (n={len(ms)})')

    summary.append('')
    if err.get('count'):
        summary.append(f'Реализованная ошибка «факт vs прогноз» (проверено фактов: {err["count"]}):')
        summary.append('')
        summary.append(f'{"Цель":40s} MAE')
        for col, mae in err['per_target'].items():
            summary.append(f'  {col:38s} {mae:.4f}')
        summary.append('')
        summary.append('MAE по упреждению (агрегат по целям):')
        for h, mae in enumerate(err['per_horizon'], start=1):
            summary.append(f'  +{h:2d} с: {mae:.4f}')
    else:
        summary.append('Реализованная ошибка: нет данных (слишком короткая сессия)')

    summary_text = '\n'.join(summary)
    (out_dir / 'summary.txt').write_text(summary_text, encoding='utf-8')
    log_event('INFO', f"Сессия завершена: {stats}")

    print('\n' + summary_text)
    print(f"\n📁 Артефакты: {out_dir}")


if __name__ == '__main__':
    main()
