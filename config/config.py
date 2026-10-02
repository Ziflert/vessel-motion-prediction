from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class Config:
    """
    Гибкая конфигурация с поддержкой разных режимов обучения

    Режимы (profiles):
    - 'motion_prediction': Предсказание качки (Pitch, Roll, Vertical + velocities)
    - 'motion_core_prediction': Только ПОЗИЦИОННЫЕ цели качки (Pitch, Roll, Vertical)
    - 'rot_prediction': Предсказание поворота судна (ROT на основе руля и погоды)
    - 'speed_prediction': Предсказание скорости (SOG на основе условий)
    - 'full_prediction': Всё вместе (качка + ROT + скорость)
    - 'custom': Ручная настройка через target_columns
    """

    # =========================================================================
    # РЕЖИМ ОБУЧЕНИЯ
    # =========================================================================
    profile: str = "full_prediction"  # motion_prediction | motion_core_prediction | rot_prediction | speed_prediction | full_prediction | custom

    # =========================================================================
    # ИНЖЕНЕРИЯ ПРИЗНАКОВ (см. data/features.py)
    # =========================================================================
    # Циклические углы (Wave/Wind/Current direction, Course) кодируются парой sin/cos
    cyclic_encoding: bool = True
    # Добавляется относительный угол встречи волны (Wave.direction - Course) как sin/cos
    relative_wave_angle: bool = True
    # Добавляется относительный угол ветра (Wind.direction - Course) как sin/cos
    # (КУСОВОЙ УГОЛ: под каким углом ветер наваливается на корпус, а не абсолютное направление)
    relative_wind_angle: bool = True
    # Печатать баннер конфигурации при создании (отключается при программной загрузке)
    verbose: bool = True

    # =========================================================================
    # ВСЕ ДОСТУПНЫЕ КОЛОНКИ ИЗ CSV
    # =========================================================================
    all_available_columns: List[str] = field(default_factory=lambda: [
        # === ДВИЖЕНИЕ СУДНА ===
        'Pitch(градусы)',
        'Roll(градусы)',
        'Vertical(Метр)',
        'Velocity.Pitching(°/мин)',
        'Velocity.Rolling(°/мин)',
        'Velocity.Vertical(узлы)',
        'Velocity.Yawing(°/мин)',

        # === ВОЛНЫ И МОРЕ ===
        'Swell(метры)',
        'Wave.Highest(метры)',
        'Wave.current(метры)',
        'Wave.direction(градусы)',
        'Swell.direction(градусы)',
        'Wave.speed(узлы)',

        # === ВЕТЕР ===
        'Wind.direction(градусы)',

        # === ТЕЧЕНИЕ ===
        'Current.direction(градусы)',
        'Current.speed(узлы)',

        # === УПРАВЛЕНИЕ ===
        'Rudder Order(градусы)',
        'Rudder State(градусы)',
        'RPM(Обороты в минуту)',

        # === НАВИГАЦИЯ ===
        'Long(градусы)',
        'Lat(градусы)',
        'STW(узлы)',
        'SOG(узлы)',
        'ROT(°/мин)',
        'Course(градусы)',

        # === МОМЕНТЫ И СИЛЫ ===
        'Moment Yawing(тс*м)',
        'Moment Rolling(тс*м)',
        'Moment Pitching(тс*м)',
        'Force Vertical(тс)',
        'Force Summary(тс)',
        'Force Longitudinal(тс)',
        'Force Lateral(тс)',

        # === ВЕТРОВЫЕ МОМЕНТЫ И СИЛЫ ===
        'Wind.Moment Yawing(тс*м)',
        'Wind.Moment Rolling(тс*м)',
        'Wind.Moment Pitching(тс*м)',
        'Wind.Force Vertical(тс)',
        'Wind.Force Summary(тс)',
        'Wind.Force Longitudinal(тс)',
        'Wind.Force Lateral(тс)',
    ])

    # =========================================================================
    # ПРОФИЛИ ПРИЗНАКОВ И ЦЕЛЕЙ
    # =========================================================================

    # Эти будут автоматически заполнены на основе profile в __post_init__
    feature_columns: Optional[List[str]] = None
    target_columns: Optional[List[str]] = None

    # Веса для разных целевых переменных (для функции потерь)
    target_weights: Optional[Dict[str, float]] = None

    # =========================================================================
    # ВРЕМЕННЫЕ ПАРАМЕТРЫ
    # =========================================================================
    sequence_length: int = 120  # История (шагов)
    prediction_horizon: int = 20  # ЧИСЛО ВЫХОДОВ модели (BUG-LSTM-03: каждый выход
                                  # отстоит от предыдущего на prediction_step шагов
                                  # по времени; при step=1 — это и есть горизонт в шагах)
    prediction_step: int = 1     # Шаг по времени между выходными шагами

    # =========================================================================
    # АРХИТЕКТУРА МОДЕЛИ
    # =========================================================================
    # model_type — декларативная метка (попадает в manifest['model']['type']);
    # BUG-LSTM-10: трансформерные ветки (num_attention_heads, transformer_ff_dim,
    # transformer_num_layers) удалены — текущая Seq2Seq ветка (LSTM + attention)
    # их не использовала, изменение не меняло сеть и вводило в заблуждение свипы.
    model_type: str = "lstm"

    encoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 96])
    encoder_dropout: float = 0.15   # dropout между полносвязными слоями feature extractor

    temporal_hidden_size: int = 128
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.1   # dropout МЕЖДУ слоями LSTM (BUG-LSTM-02: реально используется)
    bidirectional: bool = False

    decoder_hidden_dim: int = 96
    decoder_dropout: float = 0.1
    use_attention: bool = True

    # =========================================================================
    # ОБУЧЕНИЕ
    # =========================================================================
    loss_weights: Dict[str, float] = field(default_factory=lambda:
        {'mse': 1.0, 'huber': 0.5, 'smoothness': 0.1})
    batch_size: int = 48
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    num_epochs: int = 150
    early_stopping_patience: int = 25
    max_grad_norm: float = 1.0
    teacher_forcing_ratio: float = 0.5  # Вероятность подачи реального target на шаг декодера

    # =========================================================================
    # ОПТИМИЗАЦИИ
    # =========================================================================
    pin_memory: bool = True
    num_workers: int = 0  # 0 для Windows

    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    cudnn_benchmark: bool = True
    seed: int = 42

    # =========================================================================
    # ОПРЕДЕЛЕНИЯ ПРОФИЛЕЙ
    # =========================================================================

    @staticmethod
    def get_profile_config(profile_name: str) -> tuple:
        """
        Возвращает (features, targets, weights) для заданного профиля
        """

        # Общие признаки внешних условий
        WEATHER_FEATURES = [
            'Swell(метры)',
            'Wave.Highest(метры)',
            'Wave.direction(градусы)',
            'Wave.speed(узлы)',
            'Wind.direction(градусы)',
            'Current.direction(градусы)',
            'Current.speed(узлы)',
        ]

        # Признаки управления
        CONTROL_FEATURES = [
            'Rudder Order(градусы)',
            'Rudder State(градусы)',
            'RPM(Обороты в минуту)',
        ]

        # Текущее состояние качки
        MOTION_STATE = [
            'Pitch(градусы)',
            'Roll(градусы)',
            'Vertical(Метр)',
            'Velocity.Pitching(°/мин)',
            'Velocity.Rolling(°/мин)',
            'Velocity.Vertical(узлы)',
        ]

        # Навигационное состояние
        NAV_STATE = [
            'SOG(узлы)',
            'ROT(°/мин)',
            'Course(градусы)',
        ]

        profiles = {
            'motion_prediction': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES + MOTION_STATE + ['SOG(узлы)'],
                'targets': [
                    'Pitch(градусы)',
                    'Roll(градусы)',
                    'Vertical(Метр)',
                    'Velocity.Pitching(°/мин)',
                    'Velocity.Rolling(°/мин)',
                    'Velocity.Vertical(узлы)',
                ],
                'weights': {
                    'Pitch(градусы)': 1.0,
                    'Roll(градусы)': 2.0,  # Крен важнее!
                    'Vertical(Метр)': 1.5,
                    'Velocity.Pitching(°/мин)': 0.8,
                    'Velocity.Rolling(°/мин)': 1.2,
                    'Velocity.Vertical(узлы)': 0.8,
                }
            },

            # Только позиционные цели качки — без зашумлённых производных.
            # Velocity.* — дифференциалы датчика, они доминируют в ошибке.
            # Скорости при необходимости можно получить дифференцированием позиций.
            'motion_core_prediction': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES + MOTION_STATE + ['SOG(узлы)'],
                'targets': [
                    'Pitch(градусы)',
                    'Roll(градусы)',
                    'Vertical(Метр)',
                ],
                'weights': {
                    'Pitch(градусы)': 1.0,
                    'Roll(градусы)': 2.0,   # Крен — самое важное для безопасности
                    'Vertical(Метр)': 1.5,
                }
            },

            'rot_prediction': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES + ['ROT(°/мин)', 'Course(градусы)', 'SOG(узлы)'] +
                            ['Roll(градусы)', 'Pitch(градусы)'],
                'targets': ['ROT(°/мин)'],
                'weights': {'ROT(°/мин)': 1.0}
            },

            # Профиль для реальных записей Transas w-5/6/7 (2026-10-01): без Pitch —
            # канал дифферента в этих записях вырожден (std≈0.002°, см.
            # results/real_w5w7_eda/report.txt; проверка канала — при следующем
            # визите в МТЦ). Убирает Pitch из features И targets.
            'transas_core': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES
                            + [c for c in MOTION_STATE if c != 'Pitch(градусы)'
                               and c != 'Velocity.Pitching(°/мин)']
                            + ['SOG(узлы)', 'ROT(°/мин)'],
                'targets': [
                    'Roll(градусы)',
                    'Vertical(Метр)',
                    'Velocity.Rolling(°/мин)',
                    'Velocity.Vertical(узлы)',
                ],
                'weights': {
                    'Roll(градусы)': 2.0,
                    'Vertical(Метр)': 1.5,
                    'Velocity.Rolling(°/мин)': 1.2,
                    'Velocity.Vertical(узлы)': 0.8,
                }
            },

            'speed_prediction': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES + ['SOG(узлы)'] + MOTION_STATE,
                'targets': ['SOG(узлы)'],
                'weights': {'SOG(узлы)': 1.0}
            },

            'full_prediction': {
                'features': WEATHER_FEATURES + CONTROL_FEATURES + MOTION_STATE + NAV_STATE,
                'targets': [
                    # Качка
                    'Pitch(градусы)',
                    'Roll(градусы)',
                    'Vertical(Метр)',
                    'Velocity.Pitching(°/мин)',
                    'Velocity.Rolling(°/мин)',
                    'Velocity.Vertical(узлы)',
                    # Навигация
                    'ROT(°/мин)',
                    'SOG(узлы)',
                ],
                'weights': {
                    # Качка
                    'Pitch(градусы)': 1.0,
                    'Roll(градусы)': 2.0,  # Самое важное
                    'Vertical(Метр)': 1.5,
                    'Velocity.Pitching(°/мин)': 0.8,
                    'Velocity.Rolling(°/мин)': 1.2,
                    'Velocity.Vertical(узлы)': 0.8,
                    # Навигация
                    'ROT(°/мин)': 1.5,
                    'SOG(узлы)': 1.0,
                }
            },

            # НЕОБХОДИМЫЙ МИНИМУМ (вопрос заказчика 2026-09-28): минимум параметров,
            # без координат/дубликатов/вычисленных эффектов. Направления ветра/волнения
            # в нейросеть НЕ подаются как абсолютные — считаются КУСОВЫЕ УГЛЫ
            # (Wave/Wind.direction − Course, sin/cos; data/features.py).
            # Датасет: data/raw/your_data_minimal.csv (17 колонок — scripts/make_minimal_dataset.py).
            'minimal_prediction': {
                'features': MOTION_STATE + ['SOG(узлы)', 'ROT(°/мин)', 'Course(градусы)']
                            + CONTROL_FEATURES
                            + ['Wave.Highest(метры)', 'Wave.speed(узлы)', 'Wind.Force Summary(тс)']
                            + ['Wave.direction(градусы)', 'Wind.direction(градусы)'],
                'targets': [
                    # Качка
                    'Pitch(градусы)',
                    'Roll(градусы)',
                    'Vertical(Метр)',
                    'Velocity.Pitching(°/мин)',
                    'Velocity.Rolling(°/мин)',
                    'Velocity.Vertical(узлы)',
                    # Навигация
                    'ROT(°/мин)',
                    'SOG(узлы)',
                ],
                'weights': {
                    # Качка
                    'Pitch(градусы)': 1.0,
                    'Roll(градусы)': 2.0,  # Самое важное
                    'Vertical(Метр)': 1.5,
                    'Velocity.Pitching(°/мин)': 0.8,
                    'Velocity.Rolling(°/мин)': 1.2,
                    'Velocity.Vertical(узлы)': 0.8,
                    # Навигация
                    'ROT(°/мин)': 1.5,
                    'SOG(узлы)': 1.0,
                }
            },
        }

        if profile_name not in profiles:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(profiles.keys())}")

        profile_data = profiles[profile_name]
        return profile_data['features'], profile_data['targets'], profile_data['weights']

    @property
    def input_dim(self) -> int:
        return len(self.feature_columns) if self.feature_columns else 0

    @property
    def output_dim(self) -> int:
        return len(self.target_columns) if self.target_columns else 0

    @property
    def prediction_steps(self) -> int:
        """Число выходных шагов модели (= prediction_horizon; BUG-LSTM-03:
        horizon — число выходов, а не физическое время; физический интервал
        между выходами задаёт prediction_step)."""
        return self.prediction_horizon

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
        """Инициализация на основе выбранного профиля"""

        # Если профиль не custom, загружаем предустановленную конфигурацию
        if self.profile != 'custom':
            features, targets, weights = self.get_profile_config(self.profile)
            self.feature_columns = features
            self.target_columns = targets
            self.target_weights = weights
        else:
            # Для custom профиля используем то, что задано вручную
            if self.feature_columns is None or self.target_columns is None:
                raise ValueError("For 'custom' profile, you must specify feature_columns and target_columns")
            if self.target_weights is None:
                # Равные веса по умолчанию
                self.target_weights = {col: 1.0 for col in self.target_columns}

        # Инженерия признаков: sin/cos для циклических углов + относительный
        # угол встречи волны. Список feature_columns становится ИТОГОВЫМ
        # (именно эти колонки должны быть в данных при обучении и inference).
        self.feature_engineering = None
        if self.cyclic_encoding or self.relative_wave_angle:
            from data.features import engineer_feature_columns
            self.feature_columns = engineer_feature_columns(
                self.feature_columns,
                cyclic=self.cyclic_encoding,
                relative_wave_angle=self.relative_wave_angle,
                relative_wind_angle=self.relative_wind_angle,
            )
            self.feature_engineering = {
                'cyclic_encoding': self.cyclic_encoding,
                'relative_wave_angle': self.relative_wave_angle,
                'relative_wind_angle': self.relative_wind_angle,
            }

        if not self.verbose:
            return

        # Валидация
        complexity = self.sequence_length * self.prediction_horizon
        speed = self.get_speed_estimate()

        print(f"\n{'=' * 70}")
        print(f"🎯 PREDICTION PROFILE: {self.profile.upper()}")
        print(f"{'=' * 70}")

        print(f"\n📊 TARGETS ({len(self.target_columns)} variables):")
        for target in self.target_columns:
            weight = self.target_weights.get(target, 1.0)
            print(f"   • {target:40s} (weight: {weight:.1f})")

        print(f"\n🌤️  INPUT FEATURES ({len(self.feature_columns)} variables):")
        if self.feature_engineering:
            print(f"   Feature engineering: cyclic sin/cos = {self.feature_engineering['cyclic_encoding']}, "
                  f"relative wave angle = {self.feature_engineering['relative_wave_angle']}, "
                  f"relative wind angle = {self.feature_engineering['relative_wind_angle']}")
        # Группируем признаки
        weather = [f for f in self.feature_columns if any(x in f for x in ['Swell', 'Wave', 'Wind', 'Current'])]
        control = [f for f in self.feature_columns if any(x in f for x in ['Rudder', 'RPM'])]
        state = [f for f in self.feature_columns if f not in weather and f not in control]

        if weather:
            print(f"   Weather ({len(weather)}): {', '.join(weather[:3])}...")
        if control:
            print(f"   Control ({len(control)}): {', '.join(control)}")
        if state:
            print(f"   State ({len(state)}): {', '.join(state[:3])}...")

        print(f"\n⚙️  MODEL CONFIGURATION:")
        print(f"   Device: {self.device.upper()}")

        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA: {torch.version.cuda}")

        print(f"   Sequence: {self.sequence_length} steps")
        print(f"   Prediction: {self.prediction_horizon} steps")
        print(f"   Complexity: {complexity:,}")
        print(f"   Speed: {speed}")
        print(f"   Batch size: {self.batch_size}")

        # CUDA оптимизации
        if self.device == 'cuda' and self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print(f"   ✓ cuDNN benchmark enabled")

        if self.device == 'cpu':
            print(f"\n   ⚠️  WARNING: Running on CPU! Training will be SLOW!")

        print(f"{'=' * 70}\n")