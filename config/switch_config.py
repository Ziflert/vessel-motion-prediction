"""
Утилита для быстрого переключения между режимами обучения
Поддерживает:
1. Профили предсказания (motion, rot, speed, full)
2. Режимы скорости (fast, balanced, quality)
"""

import sys
from pathlib import Path
import re


# ============================================================================
# ПРОФИЛИ ПРЕДСКАЗАНИЯ
# ============================================================================
PROFILES = {
    'motion': {
        'name': 'motion_prediction',
        'description': 'Предсказание качки судна (Pitch, Roll, Vertical + velocities)',
        'targets': 6,
        'emoji': '🌊',
    },
    'core': {
        'name': 'motion_core_prediction',
        'description': 'Только позиционные цели качки (Pitch, Roll, Vertical) — без зашумлённых производных',
        'targets': 3,
        'emoji': '🎯',
    },
    'rot': {
        'name': 'rot_prediction',
        'description': 'Предсказание поворота (ROT на основе руля и погоды)',
        'targets': 1,
        'emoji': '↩️',
    },
    'speed': {
        'name': 'speed_prediction',
        'description': 'Предсказание скорости (SOG на основе условий)',
        'targets': 1,
        'emoji': '⚡',
    },
    'full': {
        'name': 'full_prediction',
        'description': 'Всё вместе (качка + ROT + скорость)',
        'targets': 8,
        'emoji': '🎯',
    },
}

# ============================================================================
# РЕЖИМЫ СКОРОСТИ
# ============================================================================
SPEED_MODES = {
    'fast': {
        'sequence_length': 120,
        'prediction_horizon': 20,
        'batch_size': 48,
        'description': 'Fast mode - для экспериментов',
        'speed': '⚡⚡⚡⚡',
        'quality': '⭐⭐⭐',
        'time': '~1 hour',
    },
    'balanced': {
        'sequence_length': 180,
        'prediction_horizon': 30,
        'batch_size': 32,
        'description': 'Balanced mode - рекомендуется для production',
        'speed': '⚡⚡⚡',
        'quality': '⭐⭐⭐⭐',
        'time': '~2-3 hours',
    },
    'quality': {
        'sequence_length': 300,
        'prediction_horizon': 60,
        'batch_size': 32,
        'description': 'Quality mode - максимальное качество',
        'speed': '⚡⚡',
        'quality': '⭐⭐⭐⭐⭐',
        'time': '~6-8 hours',
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

    # Извлекаем значения
    profile_match = re.search(r'profile:\s*str\s*=\s*["\']([^"\']+)["\']', content)
    seq_match = re.search(r'sequence_length:\s*int\s*=\s*(\d+)', content)
    pred_match = re.search(r'prediction_horizon:\s*int\s*=\s*(\d+)', content)
    batch_match = re.search(r'batch_size:\s*int\s*=\s*(\d+)', content)

    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)

    if profile_match:
        profile_name = profile_match.group(1)
        print(f"\n🎯 PREDICTION PROFILE: {profile_name}")

        # Находим соответствующий профиль
        for key, prof in PROFILES.items():
            if prof['name'] == profile_name:
                print(f"   {prof['emoji']} {prof['description']}")
                print(f"   Targets: {prof['targets']} variables")
                break

    if seq_match and pred_match and batch_match:
        seq_len = int(seq_match.group(1))
        pred_hor = int(pred_match.group(1))
        batch = int(batch_match.group(1))

        print(f"\n⚙️  MODEL PARAMETERS:")
        print(f"   Sequence length:     {seq_len}")
        print(f"   Prediction horizon:  {pred_hor}")
        print(f"   Batch size:          {batch}")
        print(f"   Complexity:          {seq_len * pred_hor:,}")

        # Определяем режим скорости
        for mode, cfg in SPEED_MODES.items():
            if (cfg['sequence_length'] == seq_len and
                cfg['prediction_horizon'] == pred_hor):
                print(f"\n   Speed Mode: {mode.upper()}")
                print(f"   {cfg['description']}")
                print(f"   Speed: {cfg['speed']}")
                print(f"   Quality: {cfg['quality']}")
                print(f"   Est. time: {cfg['time']}")
                break
        else:
            print(f"\n   Speed Mode: CUSTOM")

    print("="*70)


def switch_profile(profile_key: str):
    """Переключить профиль предсказания"""
    if profile_key not in PROFILES:
        print(f"❌ Unknown profile: {profile_key}")
        print(f"Available profiles: {', '.join(PROFILES.keys())}")
        return False

    config_file = Path("config/config.py")
    if not config_file.exists():
        print("❌ config/config.py not found")
        return False

    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    profile = PROFILES[profile_key]

    # Заменяем профиль
    content = re.sub(
        r'profile:\s*str\s*=\s*["\'][^"\']+["\']',
        f'profile: str = "{profile["name"]}"',
        content
    )

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n" + "="*70)
    print(f"✅ SWITCHED PROFILE TO: {profile_key.upper()}")
    print("="*70)
    print(f"{profile['emoji']} {profile['description']}")
    print(f"   Targets: {profile['targets']} variables")
    print("="*70)
    print("\n✅ Ready to train! Run: python run_training.py")

    return True


def switch_speed_mode(mode: str):
    """Переключить режим скорости"""
    if mode not in SPEED_MODES:
        print(f"❌ Unknown mode: {mode}")
        print(f"Available modes: {', '.join(SPEED_MODES.keys())}")
        return False

    config_file = Path("config/config.py")
    if not config_file.exists():
        print("❌ config/config.py not found")
        return False

    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cfg = SPEED_MODES[mode]

    # Заменяем значения
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

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print("\n" + "="*70)
    print(f"✅ SWITCHED SPEED MODE TO: {mode.upper()}")
    print("="*70)
    print(f"   {cfg['description']}")
    print(f"\n   Settings:")
    print(f"     Sequence length:     {cfg['sequence_length']}")
    print(f"     Prediction horizon:  {cfg['prediction_horizon']}")
    print(f"     Batch size:          {cfg['batch_size']}")
    print(f"     Complexity:          {cfg['sequence_length'] * cfg['prediction_horizon']:,}")
    print(f"\n   Performance:")
    print(f"     Speed:   {cfg['speed']}")
    print(f"     Quality: {cfg['quality']}")
    print(f"     Est. time: {cfg['time']}")
    print("="*70)
    print("\n✅ Ready to train! Run: python run_training.py")

    return True


def show_all_profiles():
    """Показать все профили"""
    print("\n" + "="*70)
    print("AVAILABLE PREDICTION PROFILES")
    print("="*70)

    for key, prof in PROFILES.items():
        print(f"\n[{key.upper()}] {prof['emoji']}")
        print(f"  {prof['description']}")
        print(f"  Targets: {prof['targets']} variables")

    print("\n" + "="*70)
    print("Usage: python switch_config.py profile <name>")
    print("Example: python switch_config.py profile full")


def show_all_speed_modes():
    """Показать все режимы скорости"""
    print("\n" + "="*70)
    print("AVAILABLE SPEED MODES")
    print("="*70)

    for mode, cfg in SPEED_MODES.items():
        complexity = cfg['sequence_length'] * cfg['prediction_horizon']
        print(f"\n[{mode.upper()}]")
        print(f"  {cfg['description']}")
        print(f"  Seq: {cfg['sequence_length']:3d} | Pred: {cfg['prediction_horizon']:3d} | "
              f"Batch: {cfg['batch_size']:3d} | Complexity: {complexity:6,}")
        print(f"  Speed: {cfg['speed']} | Quality: {cfg['quality']} | Time: {cfg['time']}")

    print("\n" + "="*70)
    print("Usage: python switch_config.py speed <mode>")
    print("Example: python switch_config.py speed balanced")




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
    if len(sys.argv) == 1:
        # Без аргументов - показываем текущую конфигурацию
        show_current_config()
        print("\n💡 Quick commands:")
        print("   python switch_config.py profile <name>  # motion, rot, speed, full")
        print("   python switch_config.py speed <mode>    # fast, balanced, quality")
        print("   python switch_config.py list            # Show all options")
        return

    command = sys.argv[1].lower()

    if command in ['list', 'show', 'help']:
        show_all_profiles()
        show_all_speed_modes()

    elif command == 'current':
        show_current_config()

    elif command == 'profile':
        if len(sys.argv) < 3:
            show_all_profiles()
        else:
            switch_profile(sys.argv[2].lower())

    elif command == 'speed':
        if len(sys.argv) < 3:
            show_all_speed_modes()
        else:
            switch_speed_mode(sys.argv[2].lower())

    else:
        print(f"❌ Unknown command: {command}")
        print("\nAvailable commands:")
        print("  python switch_config.py                    # Show current config")
        print("  python switch_config.py profile <name>     # Switch prediction profile")
        print("  python switch_config.py speed <mode>       # Switch speed mode")
        print("  python switch_config.py list               # Show all options")
        print("\nExamples:")
        print("  python switch_config.py profile full       # Full prediction mode")
        print("  python switch_config.py speed balanced     # Balanced speed mode")


if __name__ == "__main__":
    main()