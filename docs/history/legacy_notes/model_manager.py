"""
Model Manager - Утилита для управления версиями моделей
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from tabulate import tabulate


class ModelManager:
    """Класс для управления версиями моделей"""

    def __init__(self, base_dir: Path = Path("models_archive")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self, detailed: bool = False):
        """Список всех моделей (manifest.json для новых запусков, legacy-файлы для старых)"""
        models = []

        for version_dir in sorted(self.base_dir.iterdir()):
            if not version_dir.is_dir() or version_dir.name.startswith('.'):
                continue

            manifest_file = version_dir / "manifest.json"
            metrics_file = version_dir / "training_metrics.json"
            config_file = version_dir / "training_config.json"

            model_info = {
                'version': version_dir.name,
                'path': str(version_dir),
                'status': '',
            }

            if manifest_file.exists():
                # Новый формат запусков
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                results = manifest.get('results', {})
                scaled = results.get('scaled_test', {})
                model_info.update({
                    'mae': scaled.get('mae', 'N/A'),
                    'rmse': scaled.get('rmse', 'N/A'),
                    'r2': scaled.get('r2', 'N/A'),
                    'epochs': results.get('total_epochs', 'N/A'),
                    'time': results.get('training_time_seconds', 'N/A'),
                    'params': results.get('model', {}).get('total_parameters', 'N/A'),
                    'seq_len': manifest.get('model', {}).get('sequence_length', 'N/A'),
                    'pred_horizon': manifest.get('model', {}).get('prediction_horizon', 'N/A'),
                    'datetime': manifest.get('created_at', 'N/A'),
                    'status': manifest.get('status', ''),
                })
            else:
                # Legacy-папки
                if metrics_file.exists():
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        model_info.update({
                            'mae': metrics.get('test_metrics', {}).get('mae', 'N/A'),
                            'rmse': metrics.get('test_metrics', {}).get('rmse', 'N/A'),
                            'r2': metrics.get('test_metrics', {}).get('r2', 'N/A'),
                            'epochs': metrics.get('training', {}).get('total_epochs', 'N/A'),
                            'time': metrics.get('training', {}).get('training_time_seconds', 'N/A'),
                            'params': metrics.get('model', {}).get('total_parameters', 'N/A'),
                        })

                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        model_info.update({
                            'seq_len': config.get('model', {}).get('sequence_length', 'N/A'),
                            'pred_horizon': config.get('model', {}).get('prediction_horizon', 'N/A'),
                            'datetime': config.get('datetime', 'N/A'),
                        })

            models.append(model_info)

        return models

    def print_models_table(self):
        """Выводит таблицу всех моделей"""
        models = self.list_models()

        if not models:
            print("No models found in ./models_archive/")
            return

        # Подготовка данных для таблицы
        table_data = []
        for m in models:
            row = [
                m['version'],
                f"{m.get('mae', 'N/A'):.4f}" if isinstance(m.get('mae'), float) else 'N/A',
                f"{m.get('rmse', 'N/A'):.4f}" if isinstance(m.get('rmse'), float) else 'N/A',
                f"{m.get('r2', 'N/A'):.4f}" if isinstance(m.get('r2'), float) else 'N/A',
                m.get('epochs', 'N/A'),
                f"{m.get('seq_len', 'N/A')}/{m.get('pred_horizon', 'N/A')}",
                m.get('datetime', 'N/A'),
                m.get('status', ''),
            ]
            table_data.append(row)

        headers = ['Version', 'MAE', 'RMSE', 'R²', 'Epochs', 'Seq/Pred', 'Date', 'Status']

        print("\n" + "=" * 110)
        print("AVAILABLE MODELS")
        print("=" * 110)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print(f"\nTotal models: {len(models)}")
        print(f"Storage location: {self.base_dir.absolute()}")

    def get_best_model(self, metric: str = 'mae'):
        """Возвращает лучшую модель по указанной метрике"""
        models = self.list_models()

        if not models:
            return None

        valid_models = [m for m in models if isinstance(m.get(metric), float)]

        if not valid_models:
            return None

        if metric in ['mae', 'rmse']:
            best = min(valid_models, key=lambda x: x[metric])
        else:  # r2
            best = max(valid_models, key=lambda x: x[metric])

        return best

    def compare_models(self, version1: str, version2: str):
        """Сравнивает две версии моделей"""
        models = {m['version']: m for m in self.list_models()}

        if version1 not in models:
            print(f"Model {version1} not found")
            return

        if version2 not in models:
            print(f"Model {version2} not found")
            return

        m1 = models[version1]
        m2 = models[version2]

        print("\n" + "=" * 70)
        print(f"COMPARISON: {version1} vs {version2}")
        print("=" * 70)

        metrics = ['mae', 'rmse', 'r2', 'epochs', 'params']

        for metric in metrics:
            v1 = m1.get(metric, 'N/A')
            v2 = m2.get(metric, 'N/A')

            if isinstance(v1, float) and isinstance(v2, float):
                diff = v2 - v1
                pct = (diff / v1) * 100 if v1 != 0 else 0

                if metric in ['mae', 'rmse']:
                    indicator = "✓" if diff < 0 else "✗"
                elif metric == 'r2':
                    indicator = "✓" if diff > 0 else "✗"
                else:
                    indicator = ""

                print(f"{metric.upper():10} {v1:10.4f} → {v2:10.4f}  ({diff:+.4f}, {pct:+.1f}%) {indicator}")
            else:
                print(f"{metric.upper():10} {str(v1):>10} → {str(v2):>10}")

    def delete_model(self, version: str, confirm: bool = True):
        """Удаляет модель"""
        version_dir = self.base_dir / version

        if not version_dir.exists():
            print(f"Model {version} not found")
            return False

        if confirm:
            response = input(f"Are you sure you want to delete {version}? (yes/no): ")
            if response.lower() != 'yes':
                print("Deletion cancelled")
                return False

        shutil.rmtree(version_dir)
        print(f"✓ Model {version} deleted")
        return True

    def export_model(self, version: str, output_path: Path):
        """Экспортирует модель в отдельную папку"""
        version_dir = self.base_dir / version

        if not version_dir.exists():
            print(f"Model {version} not found")
            return False

        output_path = Path(output_path)

        if output_path.exists():
            response = input(f"{output_path} already exists. Overwrite? (yes/no): ")
            if response.lower() != 'yes':
                print("Export cancelled")
                return False
            shutil.rmtree(output_path)

        shutil.copytree(version_dir, output_path)
        print(f"✓ Model {version} exported to {output_path}")
        return True

    def get_model_info(self, version: str):
        """Выводит детальную информацию о модели"""
        version_dir = self.base_dir / version

        if not version_dir.exists():
            print(f"Model {version} not found")
            return

        print("\n" + "=" * 70)
        print(f"MODEL INFO: {version}")
        print("=" * 70)

        # Читаем все доступные файлы
        files_info = {
            'config': version_dir / "training_config.json",
            'metrics': version_dir / "training_metrics.json",
            'summary': version_dir / "training_summary.txt",
        }

        # Конфигурация
        if files_info['config'].exists():
            with open(files_info['config'], 'r') as f:
                config = json.load(f)

            print("\nCONFIGURATION:")
            print(f"  Date: {config.get('datetime', 'N/A')}")
            print(f"  Model type: {config.get('model', {}).get('type', 'N/A')}")
            print(f"  Sequence length: {config.get('model', {}).get('sequence_length', 'N/A')}")
            print(f"  Prediction horizon: {config.get('model', {}).get('prediction_horizon', 'N/A')}")
            print(f"  Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}")

        # Метрики
        if files_info['metrics'].exists():
            with open(files_info['metrics'], 'r') as f:
                metrics = json.load(f)

            print("\nPERFORMANCE:")
            test_metrics = metrics.get('test_metrics', {})
            print(f"  MAE:  {test_metrics.get('mae', 'N/A'):.6f}")
            print(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.6f}")
            print(f"  R²:   {test_metrics.get('r2', 'N/A'):.6f}")

            training = metrics.get('training', {})
            print(f"\n  Training epochs: {training.get('total_epochs', 'N/A')}")
            print(f"  Training time: {training.get('training_time_seconds', 0):.1f}s")

            model_info = metrics.get('model', {})
            print(f"  Total parameters: {model_info.get('total_parameters', 'N/A'):,}")

        # Файлы
        print(f"\nFILES:")
        print(f"  Location: {version_dir}")
        print(f"  Model: {(version_dir / 'checkpoints' / 'best_model.pt').exists()}")
        print(f"  Scalers: {(version_dir / 'checkpoints' / 'scalers.pkl').exists()}")
        print(f"  Config: {files_info['config'].exists()}")
        print(f"  Metrics: {files_info['metrics'].exists()}")
        print(f"  Summary: {files_info['summary'].exists()}")




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
    import argparse

    parser = argparse.ArgumentParser(description='Model Manager - Manage model versions')
    parser.add_argument('command', choices=['list', 'info', 'compare', 'best', 'delete', 'export'],
                        help='Command to execute')
    parser.add_argument('--version', '-v', help='Model version (e.g., v001_20241218_143052)')
    parser.add_argument('--version2', help='Second model version for comparison')
    parser.add_argument('--metric', default='mae', choices=['mae', 'rmse', 'r2'],
                        help='Metric for best model selection')
    parser.add_argument('--output', help='Output path for export')

    args = parser.parse_args()

    manager = ModelManager()

    if args.command == 'list':
        manager.print_models_table()

    elif args.command == 'info':
        if not args.version:
            print("Error: --version required")
            return
        manager.get_model_info(args.version)

    elif args.command == 'compare':
        if not args.version or not args.version2:
            print("Error: --version and --version2 required")
            return
        manager.compare_models(args.version, args.version2)

    elif args.command == 'best':
        best = manager.get_best_model(args.metric)
        if best:
            print(f"\nBest model by {args.metric.upper()}: {best['version']}")
            manager.get_model_info(best['version'])
        else:
            print("No models found")

    elif args.command == 'delete':
        if not args.version:
            print("Error: --version required")
            return
        manager.delete_model(args.version)

    elif args.command == 'export':
        if not args.version or not args.output:
            print("Error: --version and --output required")
            return
        manager.export_model(args.version, Path(args.output))


if __name__ == "__main__":
    main()