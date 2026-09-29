import torch
import pandas as pd
import numpy as np
import json
import pickle
import random
import argparse
import time
from pathlib import Path
from datetime import datetime

from config.config import Config
from data.dataset import create_dataloaders
from data.features import engineer_features_dataframe
from models.vessel_predictor import VesselPredictor
from training.trainer import Trainer
from training.baselines import evaluate_baselines
from training.metrics import skill_score
import registry as reg


# ============================================================================
# Данные
# ============================================================================

def load_data(data_path: Path) -> pd.DataFrame:
    """Загрузка данных: TSV/UTF-16 с fallback-ами."""
    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except Exception:
        try:
            df = pd.read_csv(data_path, sep='\t')
        except Exception:
            df = pd.read_csv(data_path)

    if 'time' in df.columns:
        df = df.drop('time', axis=1)
    return df


def split_segments(df: pd.DataFrame, chunk_size: int = 1000,
                   train_frac: float = 0.7, val_frac: float = 0.15,
                   min_chunk: int = 100):
    """
    Mixed Weather Split (Chunked) — БЕЗ склейки сегментов.

    Возвращает списки НЕПРЕРЫВНЫХ сегментов (train/val/test). Окна
    последовательностей нарезаются внутри сегментов (см. VesselDataset),
    поэтому:
      - нет утечки: окна не пересекают границы между сплитами;
      - нет «склеенных» окон через временные разрывы между чанками.
    """
    num_chunks = len(df) // chunk_size
    train_segments, val_segments, test_segments = [], [], []
    test_row_ranges = []  # (start, end) в исходном df — для честной оценки на test

    for i in range(num_chunks + 1):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        if start_idx >= len(df):
            break

        chunk = df.iloc[start_idx:end_idx]
        if len(chunk) < min_chunk:
            train_segments.append(chunk)
            continue

        n = len(chunk)
        n_train = int(n * train_frac)
        n_val = int(n * (train_frac + val_frac))
        train_segments.append(chunk.iloc[:n_train])
        val_segments.append(chunk.iloc[n_train:n_val])
        test_segments.append(chunk.iloc[n_val:])
        test_row_ranges.append((int(start_idx + n_val), int(end_idx)))

    return train_segments, val_segments, test_segments, test_row_ranges


def check_target_signal(train_segments, config, min_std: float = 1e-6) -> dict:
    """
    Гейт «мёртвых» участков ДО обучения (урок E-1; Ф0.5).

    Проверяет std целей train-подмножества ДО создания DataLoader'ов: канал с
    std≈0 в train → вырожденный target-скалер → коллапс модели (E-1: val MAE
    688σ, «модель сломалась» — фактически данные без сигнала).

    Поведение (уточнение §5.8):
      - предупреждает по каждому безсигнальному каналу (например Pitch std=0
        на спокойной воде — канал живой в шторме, вклад ограничен);
      - падает (ValueError) ТОЛЬКО если ВСЕ цели без сигнала — чистая стоянка.

    Returns: dict {target -> std} (попадает в manifest['data']['target_std_train']).
    """
    df_train = pd.concat(list(train_segments), ignore_index=True)
    stds, dead, missing = {}, [], []
    for t in config.target_columns:
        if t not in df_train.columns:
            missing.append(t)
            continue
        s = float(df_train[t].std())
        stds[t] = s
        if s < min_std:
            dead.append(t)
    if missing:
        raise ValueError(f'E-1 ГЕЙТ: цели отсутствуют в данных: {missing}')
    if len(dead) == len(config.target_columns):
        raise ValueError(
            'E-1 ГЕЙТ: ВСЕ цели без сигнала в train-подмножестве (чистая стоянка?) — '
            'обучение отменено. Урок E-1: исключать участки без сигнала ДО обучения.')
    if dead:
        print(f"  ⚠ E-1 ГЕЙТ: без сигнала в train ({len(dead)}/{len(config.target_columns)}): "
              + ', '.join(dead))
        print("    (НЕ чистая стоянка — остальные цели живые; вклад канала ограничен, "
              "per-target оценка по режимам обязательна)")
    return stds


# ============================================================================
# Основной пайплайн
# ============================================================================



def _force_utf8_stdio():
    """Windows-консоли часто нужен явный UTF-8 (иначе падает печать R²/эмодзи)."""
    import sys as _sys
    for stream in (_sys.stdout, _sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Vessel Motion Prediction - Training')
    parser.add_argument('--data-path', type=str, default='./data/raw/your_data.csv')
    parser.add_argument('--notes', type=str, default='',
                        help='Гипотеза/описание эксперимента (попадёт в manifest и реестр)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Переопределить число эпох (быстрые проверки пайплайна)')
    parser.add_argument('--data-limit', type=int, default=None,
                        help='Взять только первые N строк данных (быстрые проверки)')
    parser.add_argument('--skip-rows', type=int, default=None,
                        help='Отбросить первые N строк (например «мёртвую» тихую преамбулу записи)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Переопределить seed (мультисид-эксперименты)')
    parser.add_argument('--train-frac', type=float, default=None,
                        help='Доля каждого train-сегмента (ломается при малых долях — используйте --train-segments)')
    parser.add_argument('--train-segments', type=int, default=None,
                        help='Использовать K train-сегментов: первые (или случайные при --subset-seed)')
    parser.add_argument('--subset-seed', type=int, default=None,
                        help='Seed СЛУЧАЙНОГО выбора K train-сегментов (отвязывает размер от порядка сегментов)')
    parser.add_argument('--train-synthetic-csv', type=str, default=None,
                        help='Заменить train на синтетический файл (E2: чистая синтетика)')
    parser.add_argument('--train-synthetic-rows', type=int, default=None,
                        help='Сколько строк синтетики взять (для --train-synthetic-csv)')
    parser.add_argument('--extra-train-csv', type=str, default=None,
                        help='Добавить синтетический файл к реальному train (E3: аугментация)')
    parser.add_argument('--extra-train-rows', type=int, default=None,
                        help='Сколько строк синтетики взять из --extra-train-csv')
    parser.add_argument('--fixed-scaler', action='store_true',
                        help='Скалеры обучаются на ПОЛНОМ train независимо от подмножества '
                             '(controlled learning-curve: убирает влияние скалера из сравнения)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Переопределить learning_rate')
    parser.add_argument('--roll-weight', type=float, default=None,
                        help='Абсолютный вес цели Roll(градусы) в loss (иначе из профиля)')
    parser.add_argument('--huber-weight', type=float, default=None,
                        help='Вес huber-компоненты loss')
    parser.add_argument('--smooth-weight', type=float, default=None,
                        help='Вес smoothness-компоненты loss')
    parser.add_argument('--no-attention', action='store_true',
                        help='Отключить attention в декодере')
    parser.add_argument('--prediction-horizon', type=int, default=None,
                        help='Переопределить горизонт прогноза в шагах '
                             '(A3: матрица горизонтов 10/20/30; меняет окна датасета и выход декодера)')
    parser.add_argument('--profile', type=str, default=None,
                         help='Профиль обучения (motion_prediction | motion_core_prediction | '
                              'rot_prediction | speed_prediction | full_prediction); '
                              'default — профиль из config/config.py')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Путь к run-директории: инициализация весов из её '
                             'checkpoints/best_model.pt (warm start / дообучение)')
    args = parser.parse_args()

    start_time = time.time()

    # 1. Конфигурация + воспроизводимость + переопределения (для sweep-экспериментов)
    config = Config(profile=args.profile) if args.profile else Config()
    if args.seed is not None:
        config.seed = args.seed
    if args.max_epochs is not None:
        config.num_epochs = args.max_epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.roll_weight is not None and config.target_weights and 'Roll(градусы)' in config.target_weights:
        config.target_weights['Roll(градусы)'] = args.roll_weight
    if args.huber_weight is not None or args.smooth_weight is not None:
        lw = dict(config.loss_weights)
        if args.huber_weight is not None:
            lw['huber'] = args.huber_weight
        if args.smooth_weight is not None:
            lw['smoothness'] = args.smooth_weight
        config.loss_weights = lw
    if args.no_attention:
        config.use_attention = False
    if args.prediction_horizon is not None:
        config.prediction_horizon = args.prediction_horizon
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    print("=" * 70)
    print("VESSEL MOTION PREDICTION - Training (registry-based versioning)")
    print("=" * 70)

    # 2. Создание запуска: run_id + manifest.json (единый источник правды)
    # run_id включает читаемый слаг из notes (например 'minimal-e1-k2-s42')
    run_id = reg.new_run_id(slug=args.notes)
    manifest = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'status': 'running',
        'hypothesis': args.notes,
        'git_commit': reg.git_commit_hash(),
        'data': {},
        'model': {},
        'training': {},
        'features': {},
        'scalers': {},
        'results': {},
    }
    run_dir = reg.create_run(reg.MODELS_DIR, manifest)
    config.checkpoint_dir = str(run_dir / 'checkpoints')
    config.log_dir = str(run_dir / 'logs')
    reg.save_pip_freeze(run_dir)

    print(f"\n📦 Run ID: {run_id}")
    print(f"   Directory: {run_dir}")
    if manifest['git_commit']:
        print(f"   Git commit: {manifest['git_commit'][:10]}")

    device = torch.device(config.device)
    if device.type == 'cpu':
        print("\n⚠ CUDA not available, using CPU")

    # 3. Загрузка данных + инженерия признаков (ОДИНАКОВАЯ с inference)
    print("\n" + "-" * 70)
    print("Loading data...")
    data_path = Path(args.data_path)
    df = load_data(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {data_path.name}")

    if args.skip_rows is not None:
        df = df.iloc[args.skip_rows:].reset_index(drop=True)
        print(f"  ⚡ SKIP-ROWS={args.skip_rows}: осталось {len(df)} строк")

    if args.data_limit is not None:
        df = df.iloc[:args.data_limit].reset_index(drop=True)
        print(f"  ⚡ QUICK TEST: data limited to first {len(df)} rows")

    if config.feature_engineering:
        df = engineer_features_dataframe(
            df,
            cyclic=config.feature_engineering['cyclic_encoding'],
            relative_wave_angle=config.feature_engineering['relative_wave_angle'],
            relative_wind_angle=config.feature_engineering['relative_wind_angle'],
        )
        print(f"  Feature engineering applied: {config.feature_engineering}")
        print(f"  Columns after engineering: {len(df.columns)}")

    # 4. Сплит на НЕПРЕРЫВНЫЕ сегменты (без утечки между выборками)
    print("\nPerforming Mixed Weather Split (chunked segments, no window leakage)...")
    train_segments, val_segments, test_segments, test_row_ranges = split_segments(df)

    # Learning curve: подмножество train-сегментов. При --subset-seed — СЛУЧАЙНОЕ
    # (отвязывает размер выборки от порядка сегментов/штормовости), иначе первые K.
    full_train_segments = list(train_segments)
    if args.train_segments is not None:
        total_segs = len(train_segments)
        if args.subset_seed is not None:
            srng = np.random.default_rng(args.subset_seed)
            k = min(args.train_segments, total_segs)
            chosen = sorted(srng.choice(total_segs, k, replace=False).tolist())
            train_segments = [train_segments[i] for i in chosen]
            print(f"  ⚡ TRAIN random {k}/{total_segs} segs (subset-seed={args.subset_seed}), "
                  f"rows={sum(len(s) for s in train_segments)}")
        else:
            train_segments = train_segments[:args.train_segments]
            print(f"  ⚡ TRAIN-SEGMENTS={args.train_segments}: train rows = "
                  f"{sum(len(s) for s in train_segments)}")
    elif args.train_frac is not None and 0 < args.train_frac < 1:
        train_segments = [s.iloc[:max(int(len(s) * args.train_frac), 1)] for s in train_segments]
        print(f"  ⚡ TRAIN-FRAC={args.train_frac}: train rows = "
              f"{sum(len(s) for s in train_segments)}")

    # E2: полная замена train на синтетику
    extra_train_segments = None
    if args.train_synthetic_csv:
        syn = load_data(Path(args.train_synthetic_csv))
        if args.train_synthetic_rows is not None:
            syn = syn.iloc[:args.train_synthetic_rows]
        if config.feature_engineering:
            syn = engineer_features_dataframe(
                syn, cyclic=config.feature_engineering['cyclic_encoding'],
                relative_wave_angle=config.feature_engineering['relative_wave_angle'],
                relative_wind_angle=config.feature_engineering['relative_wind_angle'])
        train_segments = [syn]
        print(f"  ⚡ TRAIN = SYNTHETIC: {len(syn)} rows from {args.train_synthetic_csv}")

    # E3: аугментация — добавляем синтетику к реальному train (скалеры по реальному)
    if args.extra_train_csv:
        syn = load_data(Path(args.extra_train_csv))
        if config.feature_engineering:
            syn = engineer_features_dataframe(
                syn, cyclic=config.feature_engineering['cyclic_encoding'],
                relative_wave_angle=config.feature_engineering['relative_wave_angle'],
                relative_wind_angle=config.feature_engineering['relative_wind_angle'])
        if args.extra_train_rows is not None:
            syn = syn.iloc[:args.extra_train_rows]
        extra_train_segments = [syn]
        print(f"  ⚡ EXTRA TRAIN (synthetic): +{len(syn)} rows from {args.extra_train_csv}")

    # Гейт «мёртвых» участков (урок E-1, Ф0.5): std целей train-подмножества
    # ДО обучения — коллапс E-1 проверяется на новом пайплайне автоматически.
    target_std_train = check_target_signal(train_segments, config)
    print("  Target std (train): " + ", ".join(
        f"{k.split('(')[0]}={v:.3f}" for k, v in target_std_train.items()))

    n_train = sum(len(s) for s in train_segments)
    n_val = sum(len(s) for s in val_segments)
    n_test = sum(len(s) for s in test_segments)
    print(f"  Segments: train={len(train_segments)}, val={len(val_segments)}, test={len(test_segments)}")
    print(f"  Rows:     train={n_train}, val={n_val}, test={n_test}")

    manifest['data'] = {
        'path': str(data_path),
        'sha256': reg.file_sha256(data_path),
        'rows_total': int(len(df)),
        'rows_train': int(n_train),
        'rows_val': int(n_val),
        'rows_test': int(n_test),
        'n_segments_train': len(train_segments),
        'n_segments_val': len(val_segments),
        'n_segments_test': len(test_segments),
        'test_row_ranges': test_row_ranges,
        'train_segments_used': args.train_segments,
        'subset_seed': args.subset_seed,
        'fixed_scaler': bool(args.fixed_scaler),
        'train_frac': args.train_frac,
        'target_std_train': target_std_train,
        'synthetic_train': ({'path': args.train_synthetic_csv,
                             'sha256': reg.file_sha256(args.train_synthetic_csv),
                             'rows_used': args.train_synthetic_rows} if args.train_synthetic_csv else None),
        'extra_train': ({'path': args.extra_train_csv,
                         'sha256': reg.file_sha256(args.extra_train_csv),
                         'rows_used': args.extra_train_rows} if args.extra_train_csv else None),
        'split': {'scheme': 'chunked_segments', 'chunk_size': 1000,
                  'train_frac': 0.7, 'val_frac': 0.15, 'seed': config.seed},
    }
    manifest['features'] = {
        'profile': config.profile,
        'input': list(config.feature_columns),
        'targets': list(config.target_columns),
        'target_weights': dict(config.target_weights or {}),
        'feature_engineering': config.feature_engineering,
        'input_dim': len(config.feature_columns),
        'output_dim': len(config.target_columns),
    }
    manifest['model'] = {
        'type': config.model_type,
        'sequence_length': config.sequence_length,
        'prediction_horizon': config.prediction_horizon,
        'prediction_step': config.prediction_step,
        'encoder_hidden_dims': list(config.encoder_hidden_dims),
        'encoder_dropout': config.encoder_dropout,
        'temporal_hidden_size': config.temporal_hidden_size,
        'temporal_num_layers': config.temporal_num_layers,
        'temporal_dropout': config.temporal_dropout,
        'decoder_hidden_dim': config.decoder_hidden_dim,
        'decoder_dropout': config.decoder_dropout,
        'bidirectional': config.bidirectional,
        'use_attention': config.use_attention,
    }
    manifest['training'] = {
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'num_epochs': config.num_epochs,
        'early_stopping_patience': config.early_stopping_patience,
        'max_grad_norm': config.max_grad_norm,
        'loss_weights': {'mse': 1.0, 'huber': 0.5, 'smoothness': 0.1},
        'loss_weights': dict(config.loss_weights),
        'teacher_forcing_ratio': 0.5,
        'seed': config.seed,
        'device': str(device),
        'quick_test': bool(args.data_limit or args.max_epochs),
        'overrides': {'lr': args.lr, 'roll_weight': args.roll_weight,
                      'huber_weight': args.huber_weight,
                      'smooth_weight': args.smooth_weight,
                      'no_attention': args.no_attention},
    }
    reg.write_manifest(run_dir, manifest)

    # 5. DataLoader'ы (StandardScaler обучается ТОЛЬКО на реальном train;
    #    при --fixed-scaler — на полном train до подрезки)
    study_scalers = None
    if args.fixed_scaler:
        from data.dataset import VesselDataset as _VDS
        study_scalers = _VDS(full_train_segments, config, fit_scalers=True).scalers
        print("  ⚡ FIXED-SCALER: нормализация по ПОЛНОМУ train (controlled experiment)")

    train_loader, val_loader, test_loader, scalers = create_dataloaders(
        train_segments, val_segments, test_segments, config,
        extra_train_data=extra_train_segments,
        scalers=study_scalers,
    )

    # Параметры скалеров дублируем в manifest — inference больше не зависит от pickle
    ts = scalers['targets']
    manifest['scalers'] = {
        'type': 'StandardScaler',
        'target_mean': ts.mean_.tolist(),
        'target_std': ts.scale_.tolist(),
        'target_columns': list(config.target_columns),
        'feature_scaler_file': 'checkpoints/scalers.pkl',
    }
    reg.write_manifest(run_dir, manifest)

    # 6. Модель
    print("\n" + "-" * 70)
    print("Creating model...")
    model = VesselPredictor(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        config=config
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 6.1 Warm start (дообучение): инициализация весами базовой модели
    if args.init_from:
        init_dir = Path(args.init_from)
        init_ckpt = init_dir / 'checkpoints' / 'best_model.pt'
        if not init_ckpt.exists():
            raise FileNotFoundError(f'--init-from: нет {init_ckpt}')
        init_state = torch.load(init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(init_state['model_state_dict'])
        print(f"⚡ WARM START: веса из {init_dir.name}")

    # 7. Обучение
    print("\n" + "-" * 70)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_scaler=scalers['targets'],
    )

    training_start = time.time()
    trainer.train()
    training_time = time.time() - training_start
    total_epochs = len(trainer.train_losses)

    # 8. Тест лучшей модели (масштабированные + ФИЗИЧЕСКИЕ метрики)
    print("\n" + "-" * 70)
    print("Testing best model...")
    trainer.load_checkpoint(str(Path(config.checkpoint_dir) / "best_model.pt"))
    test_metrics = trainer.evaluate(test_loader)

    phys = test_metrics.get('phys', {})
    print("\nTest Results (scaled space):")
    print(f"  Loss: {test_metrics['loss']:.4f}  MAE: {test_metrics['mae']:.4f}  "
          f"RMSE: {test_metrics['rmse']:.4f}  R²: {test_metrics['r2']:.4f}")
    if phys:
        print("\nTest Results (PHYSICAL units):")
        print(f"  MAE:  {phys['overall']['mae']:.4f}  RMSE: {phys['overall']['rmse']:.4f}  "
              f"R²: {phys['overall']['r2']:.4f}")
        print("  Per-target MAE:")
        for name, m in phys['per_target'].items():
            print(f"    {name:40s}: {m['mae']:.4f} (R²: {m['r2']:.3f})")

    # 9. Бейзлайны (persistence, линейная экстраполяция) и skill score
    print("\n" + "-" * 70)
    print("Baselines (physical units)...")
    baselines = evaluate_baselines(test_loader, config, scalers['targets'], device=str(device))
    skill = {}
    if 'persistence' in baselines and phys:
        base_mae = baselines['persistence']['overall']['mae']
        skill = {
            'overall': skill_score(phys['overall']['mae'], base_mae),
            'per_target': {
                name: skill_score(phys['per_target'][name]['mae'],
                                  baselines['persistence']['per_target'][name]['mae'])
                for name in phys['per_target']
            },
        }
        print(f"  Persistence MAE: {base_mae:.4f}  ->  Model skill score: {skill['overall']:.3f}")
        print(f"  Linear extrapolation MAE: {baselines['linear_extrapolation']['overall']['mae']:.4f}")
        print("  Skill per target (>0 = лучше бейзлайна):")
        for name, s in skill['per_target'].items():
            print(f"    {name:40s}: {s:+.3f}")
    else:
        print(f"  Skipped: {baselines.get('error', 'unknown reason')}")

    # 10. Сохранение скалеров
    scalers_path = Path(config.checkpoint_dir) / "scalers.pkl"
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)

    # 11. Legacy training_metrics.json (совместимость с model_manager)
    metrics_data = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'training': {
            'total_epochs': total_epochs,
            'training_time_seconds': training_time,
            'best_validation_loss': float(trainer.best_val_loss),
        },
        'test_metrics': {
            'loss': float(test_metrics['loss']),
            'mae': float(test_metrics['mae']),
            'rmse': float(test_metrics['rmse']),
            'r2': float(test_metrics['r2']),
        },
        'physical': phys,
        'baselines': baselines,
        'skill_vs_persistence': skill,
        'model': {
            'total_parameters': total_params,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    }
    with open(run_dir / "training_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    # 12. Финальный manifest + строка в реестре экспериментов
    reg.update_manifest(run_dir, {
        'status': 'candidate',
        'results': {
            'total_epochs': total_epochs,
            'training_time_seconds': training_time,
            'best_val_loss': float(trainer.best_val_loss),
            'scaled_test': {
                'loss': float(test_metrics['loss']),
                'mae': float(test_metrics['mae']),
                'rmse': float(test_metrics['rmse']),
                'r2': float(test_metrics['r2']),
            },
            'physical': phys,
            'baselines': baselines,
            'skill_vs_persistence': skill,
            'model': metrics_data['model'],
        },
    })

    reg.append_registry_row(reg.MODELS_DIR, {
        'run_id': run_id,
        'created_at': manifest['created_at'],
        'status': 'candidate',
        'profile': config.profile,
        'n_targets': len(config.target_columns),
        'targets': ';'.join(config.target_columns),
        'seq_len': config.sequence_length,
        'horizon': config.prediction_horizon,
        'params': total_params,
        'best_val_loss': f"{trainer.best_val_loss:.6f}",
        'test_mae_phys': f"{phys['overall']['mae']:.4f}" if phys else '',
        'test_r2_phys': f"{phys['overall']['r2']:.4f}" if phys else '',
        'test_r2_scaled': f"{test_metrics['r2']:.4f}",
        'skill_vs_persistence': f"{skill['overall']:.3f}" if skill else '',
        'data_rows': int(len(df)),
        'notes': args.notes,
        'path': str(run_dir),
    })

    # 13. Человекочитаемая сводка
    write_summary(run_dir, run_id, config, trainer, test_metrics, phys,
                  baselines, skill, training_time, total_epochs, total_params)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\n📦 Run: {run_id}")
    print(f"⏱  Total time: {total_time / 60:.1f} minutes")
    print(f"🚀 Inference:  python run_inference.py --run-id {run_id}")
    print(f"📋 Promote:    поменяйте status на 'production' в manifest.json")
    print("=" * 70)


def write_summary(run_dir, run_id, config, trainer, test_metrics, phys,
                  baselines, skill, training_time, total_epochs, total_params):
    lines = [
        "=" * 70, "TRAINING SUMMARY", "=" * 70, "",
        f"Run ID: {run_id}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Profile: {config.profile}",
        f"Training time: {training_time:.1f} s ({training_time / 60:.1f} min)",
        f"Total epochs: {total_epochs}",
        f"Device: {trainer.device}",
        f"Total parameters: {total_params:,}", "",
        "-" * 70, "RESULTS (scaled space)", "-" * 70,
        f"Best validation loss: {trainer.best_val_loss:.6f}",
        f"Test loss: {test_metrics['loss']:.6f}  MAE: {test_metrics['mae']:.6f}  "
        f"RMSE: {test_metrics['rmse']:.6f}  R2: {test_metrics['r2']:.6f}", "",
    ]
    if phys:
        lines += ["-" * 70, "RESULTS (PHYSICAL units)", "-" * 70,
                  f"MAE:  {phys['overall']['mae']:.4f}",
                  f"RMSE: {phys['overall']['rmse']:.4f}",
                  f"R2:   {phys['overall']['r2']:.4f}", "",
                  "Per-target:"]
        for name, m in phys['per_target'].items():
            lines.append(f"  {name:40s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R2={m['r2']:.3f}")
        lines.append("")
        lines.append("MAE by lead time (per horizon step):")
        lines.append("  " + "  ".join(f"{v:.3f}" for v in phys['per_horizon_mae']))
        lines.append("")
    if 'persistence' in baselines:
        lines += ["-" * 70, "BASELINES (physical units)", "-" * 70,
                  f"Persistence MAE:          {baselines['persistence']['overall']['mae']:.4f}",
                  f"Linear extrapolation MAE: {baselines['linear_extrapolation']['overall']['mae']:.4f}",
                  f"Skill vs persistence:     {skill.get('overall', float('nan')):+.3f} "
                  "(>0 = model beats baseline)", ""]
    lines += ["=" * 70, "Training completed successfully!", "=" * 70]

    with open(Path(run_dir) / "training_summary.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n✓ Summary saved to {Path(run_dir).name}/training_summary.txt")


if __name__ == "__main__":
    main()
