import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from config.config import Config
from data.dataset import create_dataloaders
from models.vessel_predictor import VesselPredictor
from training.trainer import Trainer


def get_next_version_number(base_dir: Path) -> int:
    """Определяет следующий номер версии модели"""
    if not base_dir.exists():
        return 1

    # Ищем все папки с версиями
    version_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('v')]

    if not version_dirs:
        return 1

    # Извлекаем номера версий
    versions = []
    for d in version_dirs:
        try:
            version_num = int(d.name.split('_')[0][1:])  # v001 -> 1
            versions.append(version_num)
        except:
            continue

    return max(versions) + 1 if versions else 1


def create_version_directory(base_dir: Path, config: Config) -> tuple:
    """
    Создает директорию для новой версии модели

    Returns:
        (version_dir, version_number, timestamp)
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    version_number = get_next_version_number(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Формат: v001_20241218_143052
    version_name = f"v{version_number:03d}_{timestamp}"
    version_dir = base_dir / version_name
    version_dir.mkdir(parents=True, exist_ok=True)

    # Создаем подпапки
    (version_dir / "checkpoints").mkdir(exist_ok=True)
    (version_dir / "logs").mkdir(exist_ok=True)

    return version_dir, version_number, timestamp


def save_training_config(version_dir: Path, config: Config,
                         data_info: dict, version_number: int, timestamp: str):
    """Сохраняет конфигурацию обучения"""
    config_data = {
        'version': version_number,
        'timestamp': timestamp,
        'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Информация о данных
        'data': data_info,

        # Параметры модели
        'model': {
            'type': config.model_type,
            'sequence_length': config.sequence_length,
            'prediction_horizon': config.prediction_horizon,
            'input_dim': config.input_dim,
            'output_dim': config.output_dim,
            'encoder_hidden_dims': config.encoder_hidden_dims,
            'temporal_hidden_size': config.temporal_hidden_size,
            'temporal_num_layers': config.temporal_num_layers,
            'decoder_hidden_dim': config.decoder_hidden_dim,
            'bidirectional': config.bidirectional,
            'use_attention': config.use_attention,
        },

        # Параметры обучения
        'training': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'num_epochs': config.num_epochs,
            'early_stopping_patience': config.early_stopping_patience,
            'max_grad_norm': config.max_grad_norm,
        },

        # Признаки
        'features': {
            'input_features': config.feature_columns,
            'target_features': config.target_columns,
        }
    }

    config_file = version_dir / "training_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Configuration saved to: {config_file.name}")


def save_training_summary(version_dir: Path, trainer, test_metrics: dict,
                          training_time: float, total_epochs: int):
    """Сохраняет итоговую информацию об обучении"""

    summary_file = version_dir / "training_summary.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Version: {version_dir.name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training time: {training_time:.2f} seconds ({training_time / 60:.1f} minutes)\n")
        f.write(f"Total epochs: {total_epochs}\n")
        f.write(f"Device: {trainer.device}\n\n")

        f.write("-" * 70 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n\n")

        f.write("-" * 70 + "\n")
        f.write("TRAINING RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best validation loss: {trainer.best_val_loss:.6f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Loss: {test_metrics['loss']:.6f}\n")
        f.write(f"MAE:  {test_metrics['mae']:.6f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.6f}\n")
        f.write(f"R²:   {test_metrics['r2']:.6f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("FILES SAVED\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: checkpoints/best_model.pt\n")
        f.write(f"Scalers: checkpoints/scalers.pkl\n")
        f.write(f"Config: training_config.json\n")
        f.write(f"Summary: training_summary.txt\n")
        f.write(f"Metrics: training_metrics.json\n\n")

        f.write("=" * 70 + "\n")
        f.write("Training completed successfully!\n")
        f.write("=" * 70 + "\n")

    print(f"✓ Training summary saved to: {summary_file.name}")


def save_training_metrics(version_dir: Path, trainer, test_metrics: dict,
                          total_epochs: int, training_time: float):
    """Сохраняет метрики в JSON для программной обработки"""

    metrics_data = {
        'version': version_dir.name,
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
        'model': {
            'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        }
    }

    metrics_file = version_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print(f"✓ Training metrics saved to: {metrics_file.name}")


def main():
    import time
    start_time = time.time()

    # 1. Setup
    config = Config()

    print("=" * 70)
    print("VESSEL MOTION PREDICTION - Training with Versioning")
    print("=" * 70)

    # 2. Создаем директорию для новой версии
    base_models_dir = Path("models_archive")
    version_dir, version_number, timestamp = create_version_directory(base_models_dir, config)

    print(f"\n📦 Creating new model version:")
    print(f"   Version: v{version_number:03d}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Directory: {version_dir}")

    # Обновляем пути в конфиге
    config.checkpoint_dir = str(version_dir / "checkpoints")
    config.log_dir = str(version_dir / "logs")

    device = torch.device(config.device)
    if device.type == 'cpu':
        print("\n⚠ CUDA not available, using CPU")

    print(f"\nDevice: {device}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Prediction horizon: {config.prediction_horizon}")
    print(f"Model type: {config.model_type}")

    # 3. Data Loading
    print("\n" + "-" * 70)
    print("Loading data...")

    data_path = Path("data/raw/your_data.csv")

    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except:
        df = pd.read_csv(data_path, sep='\t')

    print(f"  Loading file: {data_path.name}")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    if 'time' in df.columns:
        df = df.drop('time', axis=1)
        print("  Dropped 'time' column")

    # Mixed Weather Split (Chunked)
    print("\nPerforming Mixed Weather Split (Chunking)...")

    chunk_size = 1000
    num_chunks = len(df) // chunk_size

    train_dfs = []
    val_dfs = []
    test_dfs = []

    for i in range(num_chunks + 1):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))

        if start_idx >= len(df):
            break

        chunk = df.iloc[start_idx:end_idx]

        if len(chunk) < 100:
            train_dfs.append(chunk)
            continue

        n = len(chunk)
        n_train = int(n * 0.7)
        n_val = int(n * 0.85)

        train_dfs.append(chunk.iloc[:n_train])
        val_dfs.append(chunk.iloc[n_train:n_val])
        test_dfs.append(chunk.iloc[n_val:])

    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    print(f"  Split Strategy: Mixed Weather (Chunked)")
    print(f"  Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Сохраняем информацию о данных
    data_info = {
        'total_rows': len(df),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'num_features': len(df.columns),
    }

    # 4. Create Datasets & Loaders
    train_loader, val_loader, test_loader, scalers = create_dataloaders(
        train_df, val_df, test_df, config
    )

    # 5. Model
    print("\n" + "-" * 70)
    print("Creating model...")

    model = VesselPredictor(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        config=config
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Сохраняем конфигурацию перед обучением
    save_training_config(version_dir, config, data_info, version_number, timestamp)

    # 6. Training
    print("\n" + "-" * 70)
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Сохраняем начальное время
    training_start = time.time()

    trainer.train()

    training_time = time.time() - training_start
    total_epochs = len(trainer.train_losses) if hasattr(trainer, 'train_losses') else config.num_epochs

    # 7. Testing
    print("\n" + "-" * 70)
    print("Testing best model...")

    # Загружаем лучшую модель
    best_model_path = f"{config.checkpoint_dir}/best_model.pt"
    trainer.load_checkpoint(best_model_path)

    test_metrics = trainer.evaluate(test_loader)

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")

    # 8. Save scalers
    import pickle
    scalers_path = f"{config.checkpoint_dir}/scalers.pkl"
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"\nScalers saved to {scalers_path}")

    # 9. Сохраняем итоговую информацию
    save_training_summary(version_dir, trainer, test_metrics, training_time, total_epochs)
    save_training_metrics(version_dir, trainer, test_metrics, total_epochs, training_time)

    # 10. Создаем README для этой версии
    readme_path = version_dir / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Version: v{version_number:03d}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nQuick Stats:\n")
        f.write(f"- Test MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"- Test RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"- Test R²: {test_metrics['r2']:.4f}\n")
        f.write(f"\nTo use this model:\n")
        f.write(f"python run_inference.py --model-version {version_number}\n")

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\n📦 Model saved to: {version_dir}")
    print(f"📊 Version: v{version_number:03d}")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    print(f"\n📁 Files created:")
    print(f"   - checkpoints/best_model.pt")
    print(f"   - checkpoints/scalers.pkl")
    print(f"   - training_config.json")
    print(f"   - training_summary.txt")
    print(f"   - training_metrics.json")
    print(f"   - README.txt")
    print(f"\n🚀 To use this model:")
    print(f"   python run_inference.py --model-version {version_number}")
    print("=" * 70)


if __name__ == "__main__":
    main()