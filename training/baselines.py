"""
Бейзлайны для оценки качества прогноза качки.

Зачем: для гармонических колебаний (качка) наивный бейзлайн «повторить
последнее наблюдение» (persistence) часто очень силён. Без сравнения с ним
R² модели неинтерпретируем. Skill score = 1 - MAE_модели / MAE_бейзлайна:
> 0 — модель полезнее бейзлайна.

Все бейзлайны строятся по ТОЙ ЖЕ входной истории (последние шаги окна признаков)
и оцениваются в ФИЗИЧЕСКИХ единицах через full_physical_report.

Предполагается, что все целевые переменные входят в список признаков
(это верно для всех профилей проекта: текущее состояние качки подаётся на вход).
"""

import numpy as np
import torch
from typing import Dict, Optional

from .metrics import full_physical_report


def _target_feature_indices(config) -> Optional[list]:
    """Индексы целевых переменных в списке признаков (или None, если не все там)."""
    indices = []
    for t in config.target_columns:
        if t in config.feature_columns:
            indices.append(config.feature_columns.index(t))
        else:
            return None
    return indices


def evaluate_baselines(loader, config, target_scaler, device: str = 'cpu') -> Dict:
    """
    Оценивает бейзлайны на переданном loader'е (обычно test).

    Returns:
        {
          'persistence': <full_physical_report>,
          'linear_extrapolation': <full_physical_report>,
          'linear_window': K   # по скольким последним шагам считался наклон
        }
        или {'error': ...} если бейзлайны неприменимы.
    """
    feat_idx = _target_feature_indices(config)
    if feat_idx is None:
        return {'error': 'Not all target columns present in feature columns; '
                         'persistence baseline unavailable'}

    K = 10  # окно для линейной экстраполяции

    persistence_scaled, linear_scaled, targets_scaled = [], [], []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Текущее (последнее наблюдённое) значение целей, масштабированное
        last_values = batch_x[:, -1, feat_idx]                       # [B, n_targets]

        # Persistence: повторяем последнее значение на весь горизонт
        persistence_scaled.append(last_values.cpu().numpy())

        # Линейная экстраполяция по последним K шагам (в масштабированном
        # пространстве — допустимо, т.к. масштабирование линейное)
        hist = batch_x[:, -K:, feat_idx]                             # [B, K, T]
        slope = (hist[:, -1, :] - hist[:, 0, :]) / (K - 1)           # [B, T]
        # BUG-LSTM-09: используем ФАКТИЧЕСКОЕ число выходных шагов target
        # (batch_y.shape[1]), а не config.prediction_horizon — при ненулевом
        # prediction_step возможны несовместимые массивы
        horizon = batch_y.shape[1]
        steps = torch.arange(1, horizon + 1, device=device, dtype=torch.float32)
        pred_lin = hist[:, -1, :].unsqueeze(1) + slope.unsqueeze(1) * steps.view(1, -1, 1)
        linear_scaled.append(pred_lin.cpu().numpy())

        targets_scaled.append(batch_y.cpu().numpy())

    persistence_scaled = np.concatenate(persistence_scaled, axis=0)
    # Persistence повторяется по фактическому числу шагов горизонта target
    persistence_scaled = np.repeat(
        persistence_scaled[:, np.newaxis, :], horizon, axis=1
    )
    linear_scaled = np.concatenate(linear_scaled, axis=0)
    targets_scaled = np.concatenate(targets_scaled, axis=0)

    return {
        'persistence': full_physical_report(
            persistence_scaled, targets_scaled, target_scaler, config.target_columns
        ),
        'linear_extrapolation': full_physical_report(
            linear_scaled, targets_scaled, target_scaler, config.target_columns
        ),
        'linear_window': K,
    }
