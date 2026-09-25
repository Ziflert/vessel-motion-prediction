import torch
import numpy as np
from typing import Dict


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Вычисление метрик"""

    # Flatten
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # MAE
    mae = np.mean(np.abs(pred_flat - target_flat))

    # RMSE
    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))

    # R²
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2)
    }