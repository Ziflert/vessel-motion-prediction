from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Config:
    """Оптимизированная конфигурация с максимальной производительностью"""

    # =========================================================================
    # ВСЕ ДОСТУПНЫЕ КОЛОНКИ ИЗ CSV
    # =========================================================================
    # Все колонки из вашего файла - закомментируйте ненужные

    all_available_columns: List[str] = field(default_factory=lambda: [
        # === ДВИЖЕНИЕ СУДНА (основное) ===
        'Pitch(градусы)',  # ✓ Используется
        'Roll(градусы)',  # ✓ Используется
        'Vertical(Метр)',  # ✓ Используется
        'Velocity.Pitching(°/мин)',  # ✓ Используется
        'Velocity.Rolling(°/мин)',  # ✓ Используется
        'Velocity.Vertical(узлы)',  # ✓ Используется
        'Velocity.Yawing(°/мин)',  # ✓ Используется

        # === ВОЛНЫ И МОРЕ ===
        'Swell(метры)',  # ✓ Используется
        'Wave.Highest(метры)',  # ✓ Используется
        'Wave.current(метры)',  # ✗ Не используется (дубликат?)
        'Wave.direction(градусы)',  # ✓ Используется
        'Swell.direction(градусы)',  # ✗ Не используется (дубликат Wave.direction)
        'Wave.speed(узлы)',  # ✓ Используется

        # === ВЕТЕР ===
        'Wind.direction(градусы)',  # ✓ Используется

        # === ТЕЧЕНИЕ ===
        'Current.direction(градусы)',  # ✗ Не используется
        'Current.speed(узлы)',  # ✓ Используется

        # === УПРАВЛЕНИЕ ===
        'Rudder Order(градусы)',  # ✓ Используется
        'Rudder State(градусы)',  # ✓ Используется
        'RPM(Обороты в минуту)',  # ✓ Используется

        # === НАВИГАЦИЯ ===
        'Long(градусы)',  # ✗ Не используется (медленно меняется)
        'Lat(градусы)',  # ✗ Не используется (медленно меняется)
        'STW(узлы)',  # ✗ Не используется (дубликат SOG?)
        'SOG(узлы)',  # ✓ Используется
        'ROT(°/мин)',  # ✗ Не используется (редко нужно)
        'Course(градусы)',  # ✗ Не используется (медленно меняется)

        # === МОМЕНТЫ И СИЛЫ ===
        'Moment Yawing(тс*м)',  # ✗ Не используется (вторичный параметр)
        'Moment Rolling(тс*м)',  # ✗ Не используется (вторичный параметр)
        'Moment Pitching(тс*м)',  # ✗ Не используется (вторичный параметр)
        'Force Vertical(тс)',  # ✗ Не используется
        'Force Summary(тс)',  # ✗ Не используется
        'Force Longitudinal(тс)',  # ✗ Не используется
        'Force Lateral(тс)',  # ✗ Не используется

        # === ВЕТРОВЫЕ МОМЕНТЫ И СИЛЫ ===
        'Wind.Moment Yawing(тс*м)',  # ✗ Не используется
        'Wind.Moment Rolling(тс*м)',  # ✗ Не используется
        'Wind.Moment Pitching(тс*м)',  # ✗ Не используется
        'Wind.Force Vertical(тс)',  # ✗ Не используется
        'Wind.Force Summary(тс)',  # ✗ Не используется
        'Wind.Force Longitudinal(тс)',  # ✗ Не используется
        'Wind.Force Lateral(тс)',  # ✗ Не используется
    ])

    # =========================================================================
    # ВХОДНЫЕ ПРИЗНАКИ (только важные)
    # =========================================================================
    feature_columns: List[str] = field(default_factory=lambda: [
        # Движение (основа для предсказания)
        'Pitch(градусы)',
        'Roll(градусы)',
        'Vertical(Метр)',
        'Velocity.Pitching(°/мин)',
        'Velocity.Rolling(°/мин)',
        'Velocity.Vertical(узлы)',
        'Velocity.Yawing(°/мин)',

        # Среда (причины качки)
        'Swell(метры)',
        'Wave.Highest(метры)',
        'Wave.direction(градусы)',
        'Wind.direction(градусы)',
        'Wave.speed(узлы)',
        'Current.speed(узлы)',

        # Управление (влияние на судно)
        'Rudder Order(градусы)',
        'Rudder State(градусы)',
        'RPM(Обороты в минуту)',
        'SOG(узлы)',
    ])

    # =========================================================================
    # ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ (что предсказываем)
    # =========================================================================
    target_columns: List[str] = field(default_factory=lambda: [
        'Pitch(градусы)',  # Дифферент
        'Roll(градусы)',  # Крен (САМОЕ ВАЖНОЕ!)
        'Vertical(Метр)',  # Вертикальная качка
        'Velocity.Pitching(°/мин)',  # Скорость изменения
        'Velocity.Rolling(°/мин)',  # Скорость изменения
    ])

    # =========================================================================
    # ПАРАМЕТРЫ МОДЕЛИ - ОПТИМИЗИРОВАНЫ ДЛЯ СКОРОСТИ
    # =========================================================================

    # БЫСТРЫЙ РЕЖИМ (по умолчанию)
    sequence_length: int = 120  # 2 минуты
    prediction_horizon: int = 10  # 15 секунд

    prediction_step: int = 1
    model_type: str = "lstm"

    # Архитектура (легкая и быстрая)
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [64])
    encoder_dropout: float = 0.1

    temporal_hidden_size: int = 64
    temporal_num_layers: int = 1
    temporal_dropout: float = 0.0
    bidirectional: bool = False  # False быстрее

    # Attention
    num_attention_heads: int = 4
    transformer_ff_dim: int = 128
    transformer_num_layers: int = 1

    decoder_hidden_dim: int = 64
    decoder_dropout: float = 0.1
    use_attention: bool = True

    # =========================================================================
    # ОПТИМИЗАЦИЯ СКОРОСТИ
    # =========================================================================

    batch_size: int = 64  # Увеличен для скорости

    # CUDA оптимизации
    pin_memory: bool = True  # ✓ Ускорение передачи на GPU
    num_workers: int = 0  # ✓ КРИТИЧНО: 0 на Windows! (2+ работает плохо)

    # Обучение
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 20
    max_grad_norm: float = 1.0

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # КРИТИЧНО: автоопределение устройства
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Оптимизации для CUDA
    cudnn_benchmark: bool = True  # ✓ Автоматическая оптимизация

    seed: int = 42

    @property
    def input_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def output_dim(self) -> int:
        return len(self.target_columns)

    @property
    def prediction_steps(self) -> int:
        return self.prediction_horizon // self.prediction_step

    def get_speed_estimate(self) -> str:
        """Оценка скорости обучения"""
        complexity = self.sequence_length * self.prediction_horizon

        if complexity < 3000:
            return "VERY FAST ⚡⚡⚡⚡⚡"
        elif complexity < 10000:
            return "FAST ⚡⚡⚡⚡"
        elif complexity < 30000:
            return "MEDIUM ⚡⚡⚡"
        else:
            return "SLOW ⚡"

    def __post_init__(self):
        """Валидация и оптимизация"""
        complexity = self.sequence_length * self.prediction_horizon
        speed = self.get_speed_estimate()

        print(f"\n⚡ Configuration:")
        print(f"   Device: {self.device.upper()}")

        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Benchmark: {self.cudnn_benchmark}")

        print(f"   Sequence length: {self.sequence_length} steps")
        print(f"   Prediction horizon: {self.prediction_horizon} steps")
        print(f"   Complexity: {complexity:,}")
        print(f"   Speed: {speed}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Pin memory: {self.pin_memory}")
        print(f"   Num workers: {self.num_workers}")

        # Включаем оптимизации CUDA
        if self.device == 'cuda' and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print(f"   ✓ cuDNN benchmark enabled")

        # Предупреждение на CPU
        if self.device == 'cpu':
            print(f"\n   ⚠️  WARNING: Running on CPU! Training will be SLOW!")
            print(f"   ⚠️  Expected: ~5 batch/s (currently observed)")
            print(f"   ⚠️  With GPU: ~40 batch/s (10x faster!)")
            print(f"\n   💡 To use GPU:")
            print(f"      1. Check CUDA availability: torch.cuda.is_available()")
            print(f"      2. Reinstall PyTorch with CUDA support")