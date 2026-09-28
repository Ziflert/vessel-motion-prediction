import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from config.config import Config
from models.vessel_predictor import VesselPredictor
from data.features import engineer_features_dataframe
import registry as reg


def list_available_models(base_dir: Path = Path("models_archive")) -> list:
    """Список доступных моделей через реестр (manifest.json + legacy fallback)."""
    models = []
    for run in reg.list_runs(base_dir):
        version_dir = run['dir']
        model_file = version_dir / "checkpoints" / "best_model.pt"
        scalers_file = version_dir / "checkpoints" / "scalers.pkl"
        if not (model_file.exists() and scalers_file.exists()):
            continue

        if run['manifest'] is not None:
            m = run['manifest']
            results = m.get('results', {})
            config_data = {
                'features': {'target_features': m.get('features', {}).get('targets', [])},
                'model': m.get('model', {}),
            }
            metrics = {
                'test_metrics': results.get('scaled_test', {}),
                'training': {'total_epochs': results.get('total_epochs')},
                'status': m.get('status'),
                'run_id': m.get('run_id'),
                'hypothesis': m.get('hypothesis', ''),
            }
        else:
            # Legacy-папки (старый формат)
            config_data = {}
            metrics = {}
            cfg_file = version_dir / "training_config.json"
            metrics_file = version_dir / "training_metrics.json"
            if cfg_file.exists():
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            if metrics_file.exists():
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)

        models.append({
            'version': version_dir.name,
            'path': version_dir,
            'model_file': model_file,
            'scalers_file': scalers_file,
            'config_data': config_data,
            'metrics': metrics,
            'manifest': run['manifest'],
        })

    return models


def select_model_interactive(models: list) -> dict:
    """Интерактивный выбор модели"""
    if not models:
        raise FileNotFoundError("No trained models found in ./models_archive/")

    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)

    for i, model in enumerate(models, 1):
        print(f"\n[{i}] {model['version']}")

        # Показываем профиль если есть
        if model['config_data']:
            features_info = model['config_data'].get('features', {})
            targets = features_info.get('target_features', [])
            print(f"    Targets: {len(targets)} variables")
            if len(targets) <= 3:
                print(f"      {', '.join(targets)}")
            else:
                print(f"      {', '.join(targets[:2])}, ... (+{len(targets) - 2} more)")

        if model['metrics']:
            test_metrics = model['metrics'].get('test_metrics', {})
            training = model['metrics'].get('training', {})

            status = model['metrics'].get('status')
            if status:
                print(f"    Status: {status}")
            hypothesis = model['metrics'].get('hypothesis')
            if hypothesis:
                print(f"    Notes: {hypothesis}")

            if 'mae' in test_metrics:
                print(f"    MAE:  {test_metrics['mae']:.4f}")
            if 'rmse' in test_metrics:
                print(f"    RMSE: {test_metrics['rmse']:.4f}")
            if 'r2' in test_metrics:
                print(f"    R²:   {test_metrics['r2']:.4f}")
            if 'total_epochs' in training:
                print(f"    Epochs: {training['total_epochs']}")
        else:
            print("    (No metrics available)")

    print("\n" + "=" * 70)

    while True:
        try:
            choice = input(f"\nSelect model [1-{len(models)}] (or press Enter for latest): ").strip()

            if choice == "":
                selected = models[-1]
                print(f"✓ Selected: {selected['version']} (latest)")
                return selected

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"✓ Selected: {selected['version']}")
                return selected
            else:
                print(f"⚠ Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("⚠ Please enter a valid number")


def select_model_by_version(models: list, version_number: int) -> dict:
    """Выбор модели по номеру версии"""
    for model in models:
        if model['version'].startswith(f"v{version_number:03d}"):
            return model
    raise FileNotFoundError(f"Model version v{version_number:03d} not found")


def _scaler_from_params(mean: list, std: list) -> StandardScaler:
    """StandardScaler из параметров mean/std (без pickle)."""
    sc = StandardScaler()
    sc.mean_ = np.asarray(mean, dtype=np.float64)
    sc.scale_ = np.asarray(std, dtype=np.float64)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = len(sc.mean_)
    return sc


class VesselPredictor_Inference:
    """Класс для inference (предсказания) обученной модели"""

    def __init__(self, checkpoint_path: str, scalers_path: str,
                 new_sequence_length: int = None, new_prediction_horizon: int = None):
        """
        Args:
            checkpoint_path: Путь к сохранённой модели (best_model.pt)
            scalers_path: Путь к scalers (scalers.pkl)
            new_sequence_length: Новая длина входной последовательности
            new_prediction_horizon: Новый горизонт предсказания
        """
        # Загружаем чекпоинт и конфигурацию
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        run_dir = Path(checkpoint_path).parent.parent
        self.manifest = reg.load_manifest(run_dir)

        if self.manifest is not None:
            # Новый путь: конфигурация из manifest.json — признаки/цели/архитектура
            # восстанавливаются из чистого JSON, а не из pickle
            self.config = reg.config_from_manifest(self.manifest)
            self.run_id = self.manifest.get('run_id', run_dir.name)
            self.status = self.manifest.get('status', 'unknown')
        else:
            # Legacy: конфигурация из pickled объекта в чекпоинте
            self.config = checkpoint['config']
            self.manifest = None
            self.run_id = run_dir.name
            self.status = 'legacy'

        # Обновляем параметры если заданы новые
        self.original_sequence_length = self.config.sequence_length
        self.original_prediction_horizon = self.config.prediction_horizon

        if new_sequence_length is not None:
            self.config.sequence_length = new_sequence_length
            print(f"⚙ Sequence length changed: {self.original_sequence_length} → {new_sequence_length}")

        if new_prediction_horizon is not None:
            self.config.prediction_horizon = new_prediction_horizon
            print(f"⚙ Prediction horizon changed: {self.original_prediction_horizon} → {new_prediction_horizon}")

        # Загружаем scalers (фичи — из pickle; цели — из manifest, если есть)
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)

        if self.manifest is not None:
            ts_params = self.manifest.get('scalers', {})
            if ts_params.get('target_mean'):
                self.scalers['targets'] = _scaler_from_params(
                    ts_params['target_mean'], ts_params['target_std'])

        self.feature_scaler = self.scalers['features']
        self.target_scaler = self.scalers['targets']

        # Создаем модель
        self.model = VesselPredictor(
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
            config=self.config
        )

        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.device = torch.device('cpu')
        self.model.to(self.device)

        # Определяем профиль модели
        self.profile = getattr(self.config, 'profile', 'unknown')

        print(f"\n✓ Model loaded successfully!")
        print(f"  Run ID: {self.run_id} (status: {self.status})")
        print(f"  Profile: {self.profile.upper()}")
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  Input features: {self.config.input_dim}")
        print(f"  Output targets: {self.config.output_dim}")

        # Показываем целевые переменные
        print(f"\n  🎯 Target variables ({len(self.config.target_columns)}):")
        for target in self.config.target_columns:
            weight = self.config.target_weights.get(target, 1.0) if self.config.target_weights else 1.0
            weight_str = f" (weight: {weight:.1f}x)" if weight != 1.0 else ""
            print(f"     • {target}{weight_str}")

        print(f"\n  ⏱ Sequence length: {self.config.sequence_length} steps")
        print(f"  ⏱ Prediction horizon: {self.config.prediction_horizon} steps")

    def predict_uncertain(self, input_data: np.ndarray, mc_samples: int = 30) -> dict:
        """
        MC-Dropout: включает dropout и усредняет N стохастических прогонов.

        Returns:
            {'mean': [horizon, n_targets], 'std': [horizon, n_targets]} —
            средний прогноз и его неопределённость (физические единицы).
        """
        if input_data.shape[0] != self.config.sequence_length:
            raise ValueError(f'Input sequence length must be {self.config.sequence_length}')

        input_scaled = self.feature_scaler.transform(input_data)
        x = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)

        self.model.train()  # активирует dropout (batchnorm в модели нет)
        samples = []
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = self.model(x, target=None, teacher_forcing_ratio=0.0)
                samples.append(pred.cpu().numpy()[0])
        self.model.eval()

        samples = np.stack(samples)  # [N, horizon, n_targets]
        mean = self.target_scaler.inverse_transform(samples.mean(axis=0))
        std = samples.std(axis=0) * self.target_scaler.scale_[None, :]
        return {'mean': mean, 'std': std}

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Делает предсказание для одной последовательности

        Args:
            input_data: numpy array формы [sequence_length, num_features]

        Returns:
            predictions: numpy array формы [prediction_horizon, num_targets]
        """
        # Проверка размерности
        if input_data.shape[0] != self.config.sequence_length:
            raise ValueError(
                f"Input sequence length must be {self.config.sequence_length}, "
                f"got {input_data.shape[0]}"
            )

        if input_data.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Input features must be {self.config.input_dim}, "
                f"got {input_data.shape[1]}"
            )

        # Нормализация входных данных
        input_scaled = self.feature_scaler.transform(input_data)

        # Преобразование в тензор
        x = torch.FloatTensor(input_scaled).unsqueeze(0).to(self.device)

        # Предсказание
        with torch.no_grad():
            predictions = self.model(x, target=None, teacher_forcing_ratio=0.0)

        # Обратное масштабирование
        predictions_np = predictions.cpu().numpy()[0]
        predictions_unscaled = self.target_scaler.inverse_transform(predictions_np)

        return predictions_unscaled

    def predict_from_dataframe(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        Делает предсказание из DataFrame

        Args:
            df: DataFrame с данными
            start_idx: Индекс начала последовательности

        Returns:
            dict с предсказаниями и реальными значениями
        """
        # Извлекаем входную последовательность
        input_seq = df[self.config.feature_columns].iloc[
            start_idx:start_idx + self.config.sequence_length
        ].values

        # Извлекаем реальные будущие значения (для сравнения)
        target_start = start_idx + self.config.sequence_length
        target_end = target_start + self.config.prediction_horizon

        if target_end <= len(df):
            actual_values = df[self.config.target_columns].iloc[
                target_start:target_end
            ].values
        else:
            actual_values = None

        # Делаем предсказание
        predictions = self.predict(input_seq)

        return {
            'predictions': predictions,
            'actual': actual_values,
            'start_idx': start_idx,
            'input_sequence': input_seq
        }


def get_profile_emoji(profile: str) -> str:
    """Возвращает эмодзи для профиля"""
    emojis = {
        'motion_prediction': '🌊',
        'rot_prediction': '↩️',
        'speed_prediction': '⚡',
        'full_prediction': '🎯',
    }
    return emojis.get(profile, '📊')


def visualize_predictions(result: dict, config: Config, time_interval: float = 1.0,
                          save_path: str = None):
    """
    Визуализация предсказаний с учётом профиля модели

    Args:
        result: Результат из predict_from_dataframe
        config: Конфигурация
        time_interval: Интервал между измерениями в секундах
        save_path: Путь для сохранения графика
    """
    predictions = result['predictions']
    actual = result['actual']
    start_idx = result['start_idx']

    num_targets = len(config.target_columns)

    # Определяем размер фигуры в зависимости от количества переменных
    fig_height = min(3.5 * num_targets, 20)  # Максимум 20 дюймов
    fig, axes = plt.subplots(num_targets, 1, figsize=(14, fig_height))

    if num_targets == 1:
        axes = [axes]

    # Временные оси (в секундах)
    history_window = min(100, config.sequence_length)
    history_time = np.arange(-history_window * time_interval, 0, time_interval)
    future_time = np.arange(0, config.prediction_horizon * time_interval, time_interval)

    # Определяем профиль для заголовка
    profile = getattr(config, 'profile', 'unknown')
    profile_emoji = get_profile_emoji(profile)

    for i, target_name in enumerate(config.target_columns):
        ax = axes[i]

        # Получаем вес переменной
        weight = config.target_weights.get(target_name, 1.0) if config.target_weights else 1.0

        # 1. Исторические данные (контекст) - последние 100 точек
        if 'input_sequence' in result:
            # Ищем эту переменную в feature_columns
            if target_name in config.feature_columns:
                feat_idx = config.feature_columns.index(target_name)
                history_data = result['input_sequence'][-history_window:, feat_idx]

                if len(history_data) > 0:
                    actual_history_time = np.arange(-len(history_data) * time_interval, 0, time_interval)
                    ax.plot(actual_history_time, history_data, 'gray', alpha=0.6,
                            linewidth=1.5, label='History', marker='.', markersize=3)

        # 2. Точка разделения
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2,
                   label='Current moment', alpha=0.7)

        # 3. Предсказания
        ax.plot(future_time, predictions[:, i], 'b-o',
                label='Prediction', linewidth=2.5, markersize=6, alpha=0.8)

        # 4. Реальные значения (если есть)
        if actual is not None:
            ax.plot(future_time, actual[:, i], 'r--s',
                    label='Actual', linewidth=2.5, markersize=6, alpha=0.8)

            # Область ошибки
            ax.fill_between(future_time, predictions[:, i], actual[:, i],
                            alpha=0.2, color='orange', label='Error')

            # Метрики
            mae = np.mean(np.abs(predictions[:, i] - actual[:, i]))
            rmse = np.sqrt(np.mean((predictions[:, i] - actual[:, i]) ** 2))
            mape = np.mean(np.abs((predictions[:, i] - actual[:, i]) / (np.abs(actual[:, i]) + 1e-8))) * 100
            correlation = np.corrcoef(predictions[:, i], actual[:, i])[0, 1]

            metrics_text = (
                f'MAE: {mae:.4f}\n'
                f'RMSE: {rmse:.4f}\n'
                f'MAPE: {mape:.2f}%\n'
                f'Corr: {correlation:.3f}\n'
                f'Weight: {weight:.1f}x'
            )

            ax.text(0.02, 0.98, metrics_text,
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(f'Time (seconds)', fontsize=11)
        ax.set_ylabel(target_name, fontsize=11)

        # Форматируем заголовок с информацией о времени
        history_minutes = config.sequence_length * time_interval / 60
        future_seconds = config.prediction_horizon * time_interval
        title = (f'{target_name} (weight: {weight:.1f}x)\n'
                 f'History: {history_minutes:.1f} min ({config.sequence_length} steps) → '
                 f'Forecast: {future_seconds:.0f} sec ({config.prediction_horizon} steps)')
        ax.set_title(title, fontsize=11, fontweight='bold')

        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Общий заголовок с профилем
    profile_name = profile.replace('_', ' ').title()
    fig.suptitle(
        f'{profile_emoji} {profile_name} - Prediction from index {start_idx}',
        fontsize=14, fontweight='bold', y=1.00
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Plot saved: {Path(save_path).name}")

    plt.close()




def _force_utf8_stdio():
    """Windows-консоли часто нужен явный UTF-8 (иначе падает печать R²/эмодзи)."""
    import sys as _sys
    for stream in (_sys.stdout, _sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def get_user_input():
    """Интерактивный ввод параметров от пользователя"""

    print("\n" + "=" * 70)
    print("CONFIGURATION SETUP")
    print("=" * 70)

    # Временной интервал
    print("\n📊 Time interval between measurements:")
    print("  Your data has 1 second interval")
    time_interval = 1.0

    # Длина истории
    print("\n📈 History length (input sequence):")
    print("  Recommendations:")
    print("    - For 1 sec interval: 300-600 steps (5-10 minutes)")
    print("    - Longer history = better context but slower inference")

    while True:
        try:
            seq_input = input("\n  Enter history length in STEPS (or press Enter for default 600): ").strip()
            if seq_input == "":
                sequence_length = 600
            else:
                sequence_length = int(seq_input)

            if sequence_length < 10:
                print("  ⚠ Too short! Minimum is 10 steps")
                continue
            if sequence_length > 3600:
                print("  ⚠ Too long! Maximum is 3600 steps (1 hour)")
                continue

            seq_minutes = sequence_length * time_interval / 60
            print(f"  ✓ History: {sequence_length} steps = {seq_minutes:.1f} minutes")
            break
        except ValueError:
            print("  ⚠ Please enter a valid number")

    # Горизонт предсказания
    print("\n🔮 Prediction horizon (forecast):")
    print("  Recommendations:")
    print("    - For 1 sec interval: 60-120 steps (1-2 minutes)")
    print("    - Longer forecast = less accurate predictions")

    while True:
        try:
            pred_input = input("\n  Enter prediction horizon in STEPS (or press Enter for default 100): ").strip()
            if pred_input == "":
                prediction_horizon = 100
            else:
                prediction_horizon = int(pred_input)

            if prediction_horizon < 5:
                print("  ⚠ Too short! Minimum is 5 steps")
                continue
            if prediction_horizon > 600:
                print("  ⚠ Too long! Maximum is 600 steps (10 minutes)")
                continue

            pred_seconds = prediction_horizon * time_interval
            pred_minutes = pred_seconds / 60
            print(f"  ✓ Forecast: {prediction_horizon} steps = {pred_seconds:.0f} seconds ({pred_minutes:.1f} min)")
            break
        except ValueError:
            print("  ⚠ Please enter a valid number")

    # Количество тестов
    print("\n🎯 Number of test predictions:")
    print("  Select different points in your dataset to test model performance")

    while True:
        try:
            num_input = input("\n  How many predictions to test? (or press Enter for default 5): ").strip()
            if num_input == "":
                num_tests = 5
            else:
                num_tests = int(num_input)

            if num_tests < 1:
                print("  ⚠ Minimum is 1 test")
                continue
            if num_tests > 20:
                print("  ⚠ Maximum is 20 tests")
                continue

            print(f"  ✓ Will perform {num_tests} test predictions")
            break
        except ValueError:
            print("  ⚠ Please enter a valid number")

    # Интервалы для тестов
    print("\n🔍 Test prediction intervals:")
    print("  1. Evenly distributed across dataset (recommended)")
    print("  2. Manual selection of specific indices")
    print("  3. Random selection")

    while True:
        choice = input("\n  Select option (1-3, or press Enter for option 1): ").strip()
        if choice == "" or choice == "1":
            test_mode = "even"
            break
        elif choice == "2":
            test_mode = "manual"
            break
        elif choice == "3":
            test_mode = "random"
            break
        else:
            print("  ⚠ Invalid choice. Please enter 1, 2, or 3")

    return {
        'time_interval': time_interval,
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'num_tests': num_tests,
        'test_mode': test_mode
    }


def generate_test_indices(df_length: int, num_tests: int, test_mode: str,
                          sequence_length: int, prediction_horizon: int) -> list:
    """Генерирует индексы для тестирования"""

    min_idx = sequence_length
    max_idx = df_length - prediction_horizon

    if min_idx >= max_idx:
        raise ValueError(f"Dataset too small! Need at least {sequence_length + prediction_horizon} rows")

    if test_mode == "even":
        step = (max_idx - min_idx) // (num_tests + 1)
        indices = [min_idx + step * (i + 1) for i in range(num_tests)]

    elif test_mode == "manual":
        indices = []
        print(f"\n  Valid range: {min_idx} to {max_idx}")
        for i in range(num_tests):
            while True:
                try:
                    idx = int(input(f"  Enter index #{i + 1}/{num_tests}: "))
                    if min_idx <= idx <= max_idx:
                        indices.append(idx)
                        break
                    else:
                        print(f"    ⚠ Index must be between {min_idx} and {max_idx}")
                except ValueError:
                    print("    ⚠ Please enter a valid number")

    else:  # random
        import random
        indices = random.sample(range(min_idx, max_idx), num_tests)
        indices.sort()

    return indices


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description='Vessel Motion Prediction - Inference')
    parser.add_argument('--model-version', type=int, help='Legacy model version number (e.g., 3 for v003)')
    parser.add_argument('--run-id', type=str, help='Run ID from experiments registry (or prefix, or "production"/"latest")')
    parser.add_argument('--model-path', type=str, help='Direct path to model directory')
    parser.add_argument('--data-path', type=str, default="./data/raw/your_data.csv",
                        help='Path to data file')
    parser.add_argument('--mc-samples', type=int, default=0,
                        help='>0: включить MC-Dropout с указанным числом прогонов (неопределённость прогноза)')
    args = parser.parse_args()

    print("=" * 70)
    print("VESSEL MOTION PREDICTION - Interactive Inference")
    print("=" * 70)

    # 1. Выбор модели
    print("\n" + "-" * 70)
    print("Selecting model...")

    if args.model_path:
        model_dir = Path(args.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
        scalers_path = model_dir / "checkpoints" / "scalers.pkl"
        selected_manifest = reg.load_manifest(model_dir)

        print(f"  Using model from: {model_dir}")

    else:
        available_models = list_available_models()

        if args.run_id:
            run_dir = reg.resolve_run(args.run_id)
            selected_model = next(m for m in available_models if m['path'] == run_dir)
        elif args.model_version:
            selected_model = select_model_by_version(available_models, args.model_version)
        else:
            selected_model = select_model_interactive(available_models)

        checkpoint_path = selected_model['model_file']
        scalers_path = selected_model['scalers_file']
        selected_manifest = selected_model.get('manifest')

        print(f"\n  Model path: {selected_model['path']}")

    # 2. Проверяем существование файлов
    data_path = Path(args.data_path)

    print("\n" + "-" * 70)
    print("Checking files...")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    print(f"  ✓ Model checkpoint: {checkpoint_path.name}")

    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    print(f"  ✓ Scalers: {scalers_path.name}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    print(f"  ✓ Data file: {data_path.name}")

    # 3. Загружаем данные
    print("\n" + "-" * 70)
    print("Loading data...")

    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except:
        try:
            df = pd.read_csv(data_path, sep='\t')
        except:
            df = pd.read_csv(data_path)

    if 'time' in df.columns:
        df = df.drop('time', axis=1)

    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Инженерия признаков — ТА ЖЕ, что при обучении (зафиксирована в manifest)
    eng = (selected_manifest or {}).get('features', {}).get('feature_engineering')
    if eng:
        df = engineer_features_dataframe(
            df,
            cyclic=eng['cyclic_encoding'],
            relative_wave_angle=eng['relative_wave_angle'],
            relative_wind_angle=eng.get('relative_wind_angle', True),
        )
        print(f"  ✓ Feature engineering applied: {eng}")
    else:
        print("  Feature engineering: not used by this model (legacy)")

    # 4. Получаем параметры от пользователя
    params = get_user_input()

    # 5. Генерируем индексы для тестирования
    test_indices = generate_test_indices(
        len(df),
        params['num_tests'],
        params['test_mode'],
        params['sequence_length'],
        params['prediction_horizon']
    )

    print(f"\n  Test indices: {test_indices}")

    # 6. Загружаем модель с новыми параметрами
    print("\n" + "-" * 70)
    print("Loading model...")

    predictor = VesselPredictor_Inference(
        checkpoint_path,
        scalers_path,
        new_sequence_length=params['sequence_length'],
        new_prediction_horizon=params['prediction_horizon']
    )
    predictor.mc_samples = args.mc_samples
    if args.mc_samples > 0:
        print(f"  ⚡ MC-Dropout включён: {args.mc_samples} прогонов (неопределённость прогноза)")

    # 7. Создаем уникальную папку для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_name = Path(checkpoint_path).parent.parent.name if "models_archive" in str(
        checkpoint_path) else "default"
    output_dir = Path(f"results/inference_{model_version_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  ✓ Results will be saved to: {output_dir}")

    # Сохраняем параметры запуска
    params_file = output_dir / "parameters.txt"
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("INFERENCE PARAMETERS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model version: {model_version_name}\n")
        f.write(f"Run ID: {getattr(predictor, 'run_id', 'n/a')} "
                f"(status: {getattr(predictor, 'status', 'n/a')})\n")
        if getattr(predictor, 'manifest', None):
            f.write(f"Hypothesis: {predictor.manifest.get('hypothesis', '')}\n")
        f.write(f"Model profile: {predictor.profile}\n")
        f.write(f"Model path: {checkpoint_path}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Time interval: {params['time_interval']} seconds\n")
        f.write(f"History length: {params['sequence_length']} steps "
                f"({params['sequence_length'] * params['time_interval'] / 60:.1f} minutes)\n")
        f.write(f"Prediction horizon: {params['prediction_horizon']} steps "
                f"({params['prediction_horizon'] * params['time_interval']:.0f} seconds)\n")
        f.write(f"Number of tests: {params['num_tests']}\n")
        f.write(f"Test mode: {params['test_mode']}\n")
        f.write(f"Test indices: {test_indices}\n")
        f.write(f"\nTarget variables ({len(predictor.config.target_columns)}):\n")
        for target in predictor.config.target_columns:
            weight = predictor.config.target_weights.get(target, 1.0) if predictor.config.target_weights else 1.0
            f.write(f"  • {target} (weight: {weight:.1f}x)\n")

    # 8. Делаем предсказания
    print("\n" + "-" * 70)
    print("Making predictions...")

    all_results = []

    for idx in test_indices:
        print(f"\n  [{len(all_results) + 1}/{len(test_indices)}] Predicting from index {idx}...")

        result = predictor.predict_from_dataframe(df, start_idx=idx)
        all_results.append(result)

        if result['actual'] is not None:
            predictions = result['predictions']
            actual = result['actual']

            mae = np.mean(np.abs(predictions - actual))
            rmse = np.sqrt(np.mean((predictions - actual) ** 2))

            print(f"    MAE:  {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")

            # Визуализация
            visualize_predictions(
                result,
                predictor.config,
                time_interval=params['time_interval'],
                save_path=output_dir / f"prediction_{idx}.png"
            )

    # 9. Сохраняем результаты в CSV
    print("\n" + "-" * 70)
    print("Saving results...")

    results_data = []

    for i, result in enumerate(all_results):
        for t in range(predictor.config.prediction_horizon):
            row = {
                'sequence_idx': test_indices[i],
                'time_step': t,
                'time_seconds': t * params['time_interval'],
            }

            # Предсказания
            for j, col in enumerate(predictor.config.target_columns):
                row[f'pred_{col}'] = result['predictions'][t, j]

            # Реальные значения
            if result['actual'] is not None:
                for j, col in enumerate(predictor.config.target_columns):
                    row[f'actual_{col}'] = result['actual'][t, j]
                    row[f'error_{col}'] = result['predictions'][t, j] - result['actual'][t, j]

            # Неопределённость (MC-Dropout)
            if 'pred_std' in result:
                for j, col in enumerate(predictor.config.target_columns):
                    row[f'std_{col}'] = result['pred_std'][t, j]

            results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_csv_path = output_dir / "predictions.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"  ✓ Results saved to {results_csv_path.name}")

    # 10. Сводная статистика
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)

    if all_results[0]['actual'] is not None:
        all_maes = []
        all_rmses = []
        per_target_stats = {col: [] for col in predictor.config.target_columns}

        for result in all_results:
            mae = np.mean(np.abs(result['predictions'] - result['actual']))
            rmse = np.sqrt(np.mean((result['predictions'] - result['actual']) ** 2))
            all_maes.append(mae)
            all_rmses.append(rmse)

            # Per-target statistics
            for i, col in enumerate(predictor.config.target_columns):
                target_mae = np.mean(np.abs(result['predictions'][:, i] - result['actual'][:, i]))
                per_target_stats[col].append(target_mae)

        print(f"\n📊 Overall Performance:")
        print(f"  Average MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
        print(f"  Average RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}")
        print(f"  Best MAE:     {np.min(all_maes):.4f} (at index {test_indices[np.argmin(all_maes)]})")
        print(f"  Worst MAE:    {np.max(all_maes):.4f} (at index {test_indices[np.argmax(all_maes)]})")

        print(f"\n📈 Per-Target Performance:")
        for col in predictor.config.target_columns:
            target_mae_values = per_target_stats[col]
            weight = predictor.config.target_weights.get(col, 1.0) if predictor.config.target_weights else 1.0
            print(f"  {col:40s}: MAE = {np.mean(target_mae_values):.4f} ± {np.std(target_mae_values):.4f} "
                  f"(weight: {weight:.1f}x)")

        # Сохраняем статистику
        stats_file = output_dir / "summary_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model Profile: {predictor.profile}\n\n")
            f.write(f"Overall Performance:\n")
            f.write(f"  Average MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}\n")
            f.write(f"  Average RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}\n")
            f.write(f"  Best MAE:     {np.min(all_maes):.4f} (at index {test_indices[np.argmin(all_maes)]})\n")
            f.write(f"  Worst MAE:    {np.max(all_maes):.4f} (at index {test_indices[np.argmax(all_maes)]})\n\n")
            f.write("Per-Target Statistics:\n")
            for col in predictor.config.target_columns:
                target_mae_values = per_target_stats[col]
                weight = predictor.config.target_weights.get(col, 1.0) if predictor.config.target_weights else 1.0
                f.write(f"  {col:40s}: MAE = {np.mean(target_mae_values):.4f} ± {np.std(target_mae_values):.4f} "
                        f"(weight: {weight:.1f}x)\n")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETED!")
    print("=" * 70)
    print(f"\n📁 All results saved to: {output_dir.absolute()}")
    print(f"   - Prediction plots: prediction_*.png")
    print(f"   - CSV data: predictions.csv")
    print(f"   - Parameters: parameters.txt")
    print(f"   - Statistics: summary_statistics.txt")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()