import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List


class VesselDataset(Dataset):
    """Оптимизированный датасет для максимальной скорости

    ВАЖНО (фикс утечки): параметр data может быть списком DataFrame —
    непрерывных временных сегментов (например, чанков сплита). Окна
    последовательностей нарезаются СТРОГО внутри одного сегмента и никогда
    не пересекают временные разрывы между ними.
    """

    def __init__(self, data, config, scalers=None, fit_scalers=False):
        """
        data: DataFrame ИЛИ список непрерывных сегментов-DataFrame
        config: Конфигурация
        scalers: Словарь с обученными скалерами
        fit_scalers: Если True, обучает скалеры
        """
        self.config = config
        self.sequence_length = config.sequence_length
        self.prediction_horizon = config.prediction_horizon
        self.prediction_step = config.prediction_step

        segments = list(data) if isinstance(data, (list, tuple)) else [data]
        segments = [s for s in segments if len(s) > 0]

        # 1. Проверка колонок (по первому сегменту)
        missing_features = [c for c in config.feature_columns if c not in segments[0].columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        missing_targets = [c for c in config.target_columns if c not in segments[0].columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        # 2. Извлекаем данные (ОПТИМИЗАЦИЯ: сразу в float32)
        self.feature_data = np.vstack(
            [s[config.feature_columns].values for s in segments]
        ).astype(np.float32)
        self.target_data = np.vstack(
            [s[config.target_columns].values for s in segments]
        ).astype(np.float32)

        # Идентификатор сегмента для каждой строки: окно валидно только если
        # оно целиком лежит в одном сегменте
        segment_ids = np.concatenate(
            [np.full(len(s), i, dtype=np.int32) for i, s in enumerate(segments)]
        )

        # 3. Масштабирование
        self.scalers = {}

        if fit_scalers:
            # Обучаем скалеры (только для Train)
            self.feature_scaler = StandardScaler()
            self.feature_data = self.feature_scaler.fit_transform(self.feature_data).astype(np.float32)
            self.scalers['features'] = self.feature_scaler

            self.target_scaler = StandardScaler()
            self.target_data = self.target_scaler.fit_transform(self.target_data).astype(np.float32)
            self.scalers['targets'] = self.target_scaler
        else:
            if scalers is None:
                raise ValueError("Scalers must be provided if fit_scalers=False")

            self.feature_scaler = scalers['features']
            self.feature_data = self.feature_scaler.transform(self.feature_data).astype(np.float32)

            self.target_scaler = scalers['targets']
            self.target_data = self.target_scaler.transform(self.target_data).astype(np.float32)
            self.scalers = scalers

        # 4. ОПТИМИЗАЦИЯ: Предвычисляем валидные индексы
        total_len = len(self.feature_data)
        required_len = self.sequence_length + self.prediction_horizon

        # Окно [start, start + required_len) валидно, если первый и последний
        # элементы лежат в одном сегменте (границы сегментов монотонны)
        n_windows = total_len - required_len + 1
        if n_windows > 0:
            same_segment = segment_ids[:n_windows] == segment_ids[required_len - 1:]
            self.valid_indices = np.nonzero(same_segment)[0].astype(np.int32)
        else:
            self.valid_indices = np.array([], dtype=np.int32)

        self.n_segments = len(segments)
        self.n_dropped_windows = n_windows - len(self.valid_indices)

        # ОПТИМИЗАЦИЯ: Конвертируем данные в torch tensors заранее (опционально)
        # Это ускоряет __getitem__ но использует больше памяти
        # Раскомментируйте если есть достаточно RAM:
        # self.feature_data = torch.from_numpy(self.feature_data)
        # self.target_data = torch.from_numpy(self.target_data)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """ОПТИМИЗИРОВАННЫЙ метод получения данных"""
        start_idx = self.valid_indices[idx]

        # Входная последовательность
        feat_end = start_idx + self.sequence_length
        x = self.feature_data[start_idx:feat_end]

        # Целевая последовательность
        target_start = feat_end
        target_end = target_start + self.prediction_horizon
        y = self.target_data[target_start:target_end:self.prediction_step]

        # ОПТИМИЗАЦИЯ: Используем from_numpy вместо FloatTensor для ускорения
        return torch.from_numpy(x), torch.from_numpy(y)


def create_dataloaders(train_data, val_data, test_data, config, extra_train_data=None,
                       scalers=None):
    """
    Создает ОПТИМИЗИРОВАННЫЕ DataLoader'ы с максимальной производительностью.

    Каждый из *_data аргументов — DataFrame ИЛИ список непрерывных сегментов.
    extra_train_data — ДОПОЛНИТЕЛЬНЫЕ обучающие сегменты (например синтетика):
    скалеры по ним НЕ обучаются (только по train_data), окна нарезаются внутри
    своих сегментов.
    """

    # 1. Train Dataset (скалеры — только по основным train-сегментам,
    #    либо переданы готовые — режим fixed-scaler для controlled-экспериментов)
    print(f"\nCreating Train Dataset:")
    if scalers is not None:
        train_dataset = VesselDataset(train_data, config, scalers=scalers, fit_scalers=False)
        print("  (fixed scalers from full train)")
    else:
        train_dataset = VesselDataset(train_data, config, fit_scalers=True)
    scalers = train_dataset.scalers

    print(f"  Features: {train_dataset.feature_data.shape[1]}")
    print(f"  Targets: {train_dataset.target_data.shape[1]}")
    print(f"  Segments: {train_dataset.n_segments}, "
          f"windows dropped at segment borders: {train_dataset.n_dropped_windows}")
    print(f"  Valid sequences: {len(train_dataset)}")

    loader_datasets = [train_dataset]
    if extra_train_data is not None:
        extra_dataset = VesselDataset(extra_train_data, config, scalers=scalers, fit_scalers=False)
        print(f"  + Extra train (synthetic): segments={extra_dataset.n_segments}, "
              f"windows={len(extra_dataset)}")
        loader_datasets.append(extra_dataset)

    # 2. Validation Dataset
    print(f"\nCreating Validation Dataset:")
    val_dataset = VesselDataset(val_data, config, scalers=scalers, fit_scalers=False)
    print(f"  Segments: {val_dataset.n_segments}")
    print(f"  Valid sequences: {len(val_dataset)}")

    # 3. Test Dataset
    print(f"\nCreating Test Dataset:")
    test_dataset = VesselDataset(test_data, config, scalers=scalers, fit_scalers=False)
    print(f"  Segments: {test_dataset.n_segments}")
    print(f"  Valid sequences: {len(test_dataset)}")

    # 4. ОПТИМИЗИРОВАННЫЕ DataLoader'ы
    print(f"\nCreating Optimized DataLoaders:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Num workers: {config.num_workers}")
    print(f"  Pin memory: {config.pin_memory}")

    # Оптимальные настройки для производительности
    dataloader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory and (config.device == 'cuda'),
        'persistent_workers': config.num_workers > 0,  # Только если workers > 0
    }

    # Дополнительные оптимизации для CUDA (только если workers > 0)
    if config.device == 'cuda' and config.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2

    if len(loader_datasets) > 1:
        from torch.utils.data import ConcatDataset
        train_src = ConcatDataset(loader_datasets)
    else:
        train_src = loader_datasets[0]

    train_loader = DataLoader(
        train_src,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Проверка размерностей
    x_sample, y_sample = next(iter(train_loader))
    print(f"\n  Input shape: {x_sample.shape}")
    print(f"  Target shape: {y_sample.shape}")
    print(f"  Input dtype: {x_sample.dtype}")
    print(f"  Device: {x_sample.device}")

    return train_loader, val_loader, test_loader, scalers