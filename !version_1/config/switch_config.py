"""
Утилита для быстрого переключения между режимами обучения
"""

import sys
from pathlib import Path

CONFIGS = {
    'original': {
        'sequence_length': 60,
        'prediction_horizon': 10,
        'batch_size': 32,
        'description': 'Original configuration (как было)',
        'speed': '⚡⚡⚡⚡⚡',
        'quality': '⭐⭐⭐',
        'time': '~30 min',
    },
    'fast': {
        'sequence_length': 120,
        'prediction_horizon': 30,
        'batch_size': 64,
        'description': 'Fast mode - для экспериментов (рекомендуется)',
        'speed': '⚡⚡⚡⚡',
        'quality': '⭐⭐⭐⭐',
        'time': '~1.5 hours',
    },
    'balanced': {
        'sequence_length': 300,
        'prediction_horizon': 60,
        'batch_size': 64,
        'description': 'Balanced mode - для production',
        'speed': '⚡⚡⚡',
        'quality': '⭐⭐⭐⭐⭐',
        'time': '~5 hours',
    },
    'quality': {
        'sequence_length': 600,
        'prediction_horizon': 100,
        'batch_size': 64,
        'description': 'Quality mode - максимальное качество',
        'speed': '⚡',
        'quality': '⭐⭐⭐⭐⭐⭐',
        'time': '~15 hours',
    },
}


def show_current_config():
    """Показать текущую конфигурацию"""
    config_file = Path("config/config.py")

    if not config_file.exists():
        print("❌ config/config.py not found")
        return

    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Извлекаем текущие значения
    import re

    seq_match = re.search(r'sequence_length:\s*int\s*=\s*(\d+)', content)
    pred_match = re.search(r'prediction_horizon:\s*int\s*=\s*(\d+)', content)
    batch_match = re.search(r'batch_size:\s*int\s*=\s*(\d+)', content)

    if seq_match and pred_match and batch_match:
        seq_len = int(seq_match.group(1))
        pred_hor = int(pred_match.group(1))
        batch = int(batch_match.group(1))

        print("\n" + "=" * 70)
        print("CURRENT CONFIGURATION")
        print("=" * 70)
        print(f"  Sequence length:     {seq_len}")
        print(f"  Prediction horizon:  {pred_hor}")
        print(f"  Batch size:          {batch}")
        print(f"  Complexity:          {seq_len * pred_hor:,}")

        # Определяем режим
        for mode, cfg in CONFIGS.items():
            if (cfg['sequence_length'] == seq_len and
                    cfg['prediction_horizon'] == pred_hor):
                print(f"\n  Mode: {mode.upper()}")
                print(f"  {cfg['description']}")
                print(f"  Speed: {cfg['speed']}")
                print(f"  Quality: {cfg['quality']}")
                print(f"  Est. time: {cfg['time']}")
                break
        else:
            print(f"\n  Mode: CUSTOM")

        print("=" * 70)


def switch_config(mode: str):
    """Переключить конфигурацию"""
    if mode not in CONFIGS:
        print(f"❌ Unknown mode: {mode}")
        print(f"Available modes: {', '.join(CONFIGS.keys())}")
        return False

    config_file = Path("config/config.py")

    if not config_file.exists():
        print("❌ config/config.py not found")
        return False

    # Читаем файл
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Получаем новые значения
    cfg = CONFIGS[mode]

    # Заменяем значения
    import re

    content = re.sub(
        r'sequence_length:\s*int\s*=\s*\d+',
        f'sequence_length: int = {cfg["sequence_length"]}',
        content
    )

    content = re.sub(
        r'prediction_horizon:\s*int\s*=\s*\d+',
        f'prediction_horizon: int = {cfg["prediction_horizon"]}',
        content
    )

    content = re.sub(
        r'batch_size:\s*int\s*=\s*\d+',
        f'batch_size: int = {cfg["batch_size"]}',
        content
    )

    # Сохраняем
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n" + "=" * 70)
    print(f"✅ SWITCHED TO: {mode.upper()}")
    print("=" * 70)
    print(f"  {cfg['description']}")
    print(f"\n  Settings:")
    print(f"    Sequence length:     {cfg['sequence_length']}")
    print(f"    Prediction horizon:  {cfg['prediction_horizon']}")
    print(f"    Batch size:          {cfg['batch_size']}")
    print(f"    Complexity:          {cfg['sequence_length'] * cfg['prediction_horizon']:,}")
    print(f"\n  Performance:")
    print(f"    Speed:   {cfg['speed']}")
    print(f"    Quality: {cfg['quality']}")
    print(f"    Est. training time: {cfg['time']}")
    print("=" * 70)
    print("\n✅ Ready to train! Run: python run_training.py")

    return True


def show_all_modes():
    """Показать все доступные режимы"""
    print("\n" + "=" * 70)
    print("AVAILABLE MODES")
    print("=" * 70)

    for mode, cfg in CONFIGS.items():
        complexity = cfg['sequence_length'] * cfg['prediction_horizon']
        print(f"\n[{mode.upper()}]")
        print(f"  {cfg['description']}")
        print(
            f"  Seq: {cfg['sequence_length']:3d} | Pred: {cfg['prediction_horizon']:3d} | Batch: {cfg['batch_size']:3d} | Complexity: {complexity:6,}")
        print(f"  Speed: {cfg['speed']} | Quality: {cfg['quality']} | Time: {cfg['time']}")

    print("\n" + "=" * 70)
    print("\nUsage: python switch_config.py <mode>")
    print("Example: python switch_config.py fast")


def main():
    if len(sys.argv) == 1:
        # Без аргументов - показываем текущую конфигурацию
        show_current_config()
        print("\n💡 Tip: To switch mode, run:")
        print("   python switch_config.py <mode>")
        print("\n   Available modes: original, fast, balanced, quality")
        return

    command = sys.argv[1].lower()

    if command in ['list', 'show', 'modes']:
        show_all_modes()
    elif command in ['current', 'status']:
        show_current_config()
    elif command in CONFIGS:
        switch_config(command)
    else:
        print(f"❌ Unknown command or mode: {command}")
        print("\nAvailable commands:")
        print("  python switch_config.py                  # Show current config")
        print("  python switch_config.py list             # Show all modes")
        print("  python switch_config.py <mode>           # Switch to mode")
        print(f"\nAvailable modes: {', '.join(CONFIGS.keys())}")


if __name__ == "__main__":
    main()