import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path
from datetime import datetime
from config.config import Config
from models.vessel_predictor import VesselPredictor


def list_available_models(base_dir: Path = Path("models_archive")) -> list:
    """Возвращает список доступных версий моделей"""
    if not base_dir.exists():
        return []

    models = []
    for version_dir in sorted(base_dir.iterdir()):
        if not version_dir.is_dir() or not version_dir.name.startswith('v'):
            continue

        # Проверяем наличие необходимых файлов
        model_file = version_dir / "checkpoints" / "best_model.pt"
        scalers_file = version_dir / "checkpoints" / "scalers.pkl"
        metrics_file = version_dir / "training_metrics.json"

        if not (model_file.exists() and scalers_file.exists()):
            continue

        # Загружаем метрики если есть
        metrics = {}
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

        models.append({
            'version': version_dir.name,
            'path': version_dir,
            'model_file': model_file,
            'scalers_file': scalers_file,
            'metrics': metrics
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

        if model['metrics']:
            test_metrics = model['metrics'].get('test_metrics', {})
            training = model['metrics'].get('training', {})

            print(f"    MAE:  {test_metrics.get('mae', 'N/A'):.4f}" if 'mae' in test_metrics else "    MAE:  N/A")
            print(f"    RMSE: {test_metrics.get('rmse', 'N/A'):.4f}" if 'rmse' in test_metrics else "    RMSE: N/A")
            print(f"    R²:   {test_metrics.get('r2', 'N/A'):.4f}" if 'r2' in test_metrics else "    R²:   N/A")
            print(f"    Epochs: {training.get('total_epochs', 'N/A')}")
        else:
            print("    (No metrics available)")

    print("\n" + "=" * 70)

    while True:
        try:
            choice = input(f"\nSelect model [1-{len(models)}] (or press Enter for latest): ").strip()

            if choice == "":
                selected = models[-1]  # Последняя (самая новая)
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


class VesselPredictor_Inference:
    """Класс для inference (предсказания) обученной модели"""

    def __init__(self, checkpoint_path: str, scalers_path: str,
                 new_sequence_length: int = None, new_prediction_horizon: int = None):
        """
        Args:
            checkpoint_path: Путь к сохраненной модели (best_model.pt)
            scalers_path: Путь к scalers (scalers.pkl)
            new_sequence_length: Новая длина входной последовательности (если None, используется из модели)
            new_prediction_horizon: Новый горизонт предсказания (если None, используется из модели)
        """
        # Загружаем конфигурацию и модель
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.config = checkpoint['config']

        # Обновляем параметры если заданы новые
        self.original_sequence_length = self.config.sequence_length
        self.original_prediction_horizon = self.config.prediction_horizon

        if new_sequence_length is not None:
            self.config.sequence_length = new_sequence_length
            print(f"⚙ Sequence length changed: {self.original_sequence_length} → {new_sequence_length}")

        if new_prediction_horizon is not None:
            self.config.prediction_horizon = new_prediction_horizon
            print(f"⚙ Prediction horizon changed: {self.original_prediction_horizon} → {new_prediction_horizon}")

        # Загружаем scalers
        with open(scalers_path, 'rb') as f:
            self.scalers = pickle.load(f)

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

        print(f"✓ Model loaded successfully!")
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"  Input features: {self.config.input_dim}")
        print(f"  Output targets: {self.config.output_dim}")
        print(f"  Sequence length: {self.config.sequence_length} steps")
        print(f"  Prediction horizon: {self.config.prediction_horizon} steps")

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


def visualize_predictions(result: dict, config: Config, time_interval: float = 1.0,
                          save_path: str = None):
    """
    Визуализация предсказаний с учетом временного интервала

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

    fig, axes = plt.subplots(num_targets, 1, figsize=(14, 3.5 * num_targets))

    if num_targets == 1:
        axes = [axes]

    # Временные оси (в секундах)
    history_window = min(100, config.sequence_length)  # Показываем последние 100 точек истории
    history_time = np.arange(-history_window * time_interval, 0, time_interval)
    future_time = np.arange(0, config.prediction_horizon * time_interval, time_interval)

    for i, target_name in enumerate(config.target_columns):
        ax = axes[i]

        # 1. Исторические данные (контекст) - последние 100 точек
        if 'input_sequence' in result:
            history_data = result['input_sequence'][-history_window:, i] if result['input_sequence'].shape[
                                                                                1] > i else None
            if history_data is not None and len(history_data) > 0:
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
                f'Corr: {correlation:.3f}'
            )

            ax.text(0.02, 0.98, metrics_text,
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(f'Time (seconds)', fontsize=11)
        ax.set_ylabel(target_name, fontsize=11)

        # Форматируем заголовок с информацией о времени
        history_minutes = config.sequence_length * time_interval / 60
        future_seconds = config.prediction_horizon * time_interval
        title = (f'{target_name}\n'
                 f'History: {history_minutes:.1f} min ({config.sequence_length} steps) → '
                 f'Forecast: {future_seconds:.0f} sec ({config.prediction_horizon} steps)')
        ax.set_title(title, fontsize=12, fontweight='bold')

        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Общий заголовок
    fig.suptitle(
        f'Vessel Motion Prediction - Starting from index {start_idx}',
        fontsize=14, fontweight='bold', y=1.00
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Plot saved: {Path(save_path).name}")

    plt.close()


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
    print("  Original model was trained with: 60 steps (1 minute)")
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
    print("  Original model was trained with: 10 steps (10 seconds)")
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
    print("\n📍 Test prediction intervals:")
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
        # Равномерное распределение
        step = (max_idx - min_idx) // (num_tests + 1)
        indices = [min_idx + step * (i + 1) for i in range(num_tests)]

    elif test_mode == "manual":
        # Ручной ввод
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
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Vessel Motion Prediction - Inference')
    parser.add_argument('--model-version', type=int, help='Model version number (e.g., 1 for v001)')
    parser.add_argument('--model-path', type=str, help='Direct path to model directory')
    parser.add_argument('--data-path', type=str, default="./data/raw/your_data.csv",
                        help='Path to data file (default: ./data/raw/your_data.csv)')
    args = parser.parse_args()

    print("=" * 70)
    print("VESSEL MOTION PREDICTION - Interactive Inference")
    print("=" * 70)

    # 1. Выбор модели
    print("\n" + "-" * 70)
    print("Selecting model...")

    if args.model_path:
        # Прямой путь к модели
        model_dir = Path(args.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        checkpoint_path = model_dir / "checkpoints" / "best_model.pt"
        scalers_path = model_dir / "checkpoints" / "scalers.pkl"

        print(f"  Using model from: {model_dir}")

    else:
        # Выбор из архива моделей
        available_models = list_available_models()

        if args.model_version:
            # Выбор по номеру версии
            selected_model = select_model_by_version(available_models, args.model_version)
        else:
            # Интерактивный выбор
            selected_model = select_model_interactive(available_models)

        checkpoint_path = selected_model['model_file']
        scalers_path = selected_model['scalers_file']

        print(f"\n  Model path: {selected_model['path']}")

    # 2. Определяем путь к данным (ИСПРАВЛЕНО: определяем ПЕРЕД использованием)
    data_path = Path(args.data_path)

    # 3. Проверяем существование всех файлов
    print("\n" + "-" * 70)
    print("Checking files...")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    print(f"  ✓ Model checkpoint: {checkpoint_path}")

    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    print(f"  ✓ Scalers: {scalers_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    print(f"  ✓ Data file: {data_path}")

    # 4. Загружаем данные
    print("\n" + "-" * 70)
    print("Loading data...")

    try:
        df = pd.read_csv(data_path, sep='\t', encoding='utf-16')
    except:
        try:
            df = pd.read_csv(data_path, sep='\t')
        except:
            df = pd.read_csv(data_path)

    # Сохраняем time для визуализации (если есть)
    time_column = None
    if 'time' in df.columns:
        time_column = df['time'].copy()
        df = df.drop('time', axis=1)

    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # 5. Получаем параметры от пользователя
    params = get_user_input()

    # 6. Генерируем индексы для тестирования
    test_indices = generate_test_indices(
        len(df),
        params['num_tests'],
        params['test_mode'],
        params['sequence_length'],
        params['prediction_horizon']
    )

    print(f"\n  Test indices: {test_indices}")

    # 7. Загружаем модель с новыми параметрами
    print("\n" + "-" * 70)
    print("Loading model...")

    predictor = VesselPredictor_Inference(
        checkpoint_path,
        scalers_path,
        new_sequence_length=params['sequence_length'],
        new_prediction_horizon=params['prediction_horizon']
    )

    # 8. Создаем уникальную папку для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version_name = Path(checkpoint_path).parent.parent.name if "models_archive" in str(
        checkpoint_path) else "default"
    output_dir = Path(f"./results/inference_{model_version_name}_{timestamp}")
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
        f.write(f"Model path: {checkpoint_path}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Time interval: {params['time_interval']} seconds\n")
        f.write(
            f"History length: {params['sequence_length']} steps ({params['sequence_length'] * params['time_interval'] / 60:.1f} minutes)\n")
        f.write(
            f"Prediction horizon: {params['prediction_horizon']} steps ({params['prediction_horizon'] * params['time_interval']:.0f} seconds)\n")
        f.write(f"Number of tests: {params['num_tests']}\n")
        f.write(f"Test mode: {params['test_mode']}\n")
        f.write(f"Test indices: {test_indices}\n")
        f.write(f"\nOriginal model training parameters:\n")
        f.write(f"  Sequence length: {predictor.original_sequence_length}\n")
        f.write(f"  Prediction horizon: {predictor.original_prediction_horizon}\n")

    # 9. Делаем предсказания
    print("\n" + "-" * 70)
    print("Making predictions...")

    all_results = []

    for idx in test_indices:
        print(f"\n  [{len(all_results) + 1}/{len(test_indices)}] Predicting from index {idx}...")

        result = predictor.predict_from_dataframe(df, start_idx=idx)
        all_results.append(result)

        # Выводим метрики
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

    # 10. Сохраняем результаты в CSV
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

            results_data.append(row)

    results_df = pd.DataFrame(results_data)
    results_csv_path = output_dir / "predictions.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"  ✓ Results saved to {results_csv_path}")

    # 11. Сводная статистика
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)

    if all_results[0]['actual'] is not None:
        all_maes = []
        all_rmses = []

        for result in all_results:
            mae = np.mean(np.abs(result['predictions'] - result['actual']))
            rmse = np.sqrt(np.mean((result['predictions'] - result['actual']) ** 2))
            all_maes.append(mae)
            all_rmses.append(rmse)

        print(f"\nOverall Performance:")
        print(f"  Average MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")
        print(f"  Average RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}")
        print(f"  Best MAE:     {np.min(all_maes):.4f} (at index {test_indices[np.argmin(all_maes)]})")
        print(f"  Worst MAE:    {np.max(all_maes):.4f} (at index {test_indices[np.argmax(all_maes)]})")

        # Сохраняем статистику
        stats_file = output_dir / "summary_statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Average MAE:  {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}\n")
            f.write(f"Average RMSE: {np.mean(all_rmses):.4f} ± {np.std(all_rmses):.4f}\n")
            f.write(f"Best MAE:     {np.min(all_maes):.4f} (at index {test_indices[np.argmin(all_maes)]})\n")
            f.write(f"Worst MAE:    {np.max(all_maes):.4f} (at index {test_indices[np.argmax(all_maes)]})\n\n")
            f.write("Per-target statistics:\n")
            for i, col in enumerate(predictor.config.target_columns):
                target_maes = [np.mean(np.abs(r['predictions'][:, i] - r['actual'][:, i]))
                               for r in all_results if r['actual'] is not None]
                f.write(f"  {col}: MAE = {np.mean(target_maes):.4f} ± {np.std(target_maes):.4f}\n")

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