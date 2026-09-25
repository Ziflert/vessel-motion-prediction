import numpy as np
from typing import Dict, List


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Вычисление метрик по flatten — работает в любом пространстве
    (обычно в стандартизованном). Для физических единиц см. full_physical_report."""

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(np.mean((pred_flat - target_flat) ** 2))
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}


def _r2(pred_flat: np.ndarray, target_flat: np.ndarray) -> float:
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))


def inverse_scale_sequences(arr: np.ndarray, scaler) -> np.ndarray:
    """Обратное преобразование массива [batch, horizon, dim] через StandardScaler."""
    n, h, d = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, d)).reshape(n, h, d)


def full_physical_report(predictions_scaled: np.ndarray, targets_scaled: np.ndarray,
                         target_scaler, target_names: List[str]) -> Dict:
    """
    Полный отчёт в ФИЗИЧЕСКИХ единицах измерения:
      - overall: MAE/RMSE/R² по всем переменным и шагам;
      - per_target: MAE/RMSE/R² по каждой переменной (усреднено по горизонту);
      - per_horizon_mae: MAE по каждому шагу упреждения (усреднено по переменным) —
        ключевая кривая для качки: «как далеко вперёд модель работает»;
      - per_horizon_per_target_mae: {цель: [MAE на шаге 1, ..., MAE на шаге H]}.

    Вход: масштабированные (z-score) массивы [batch, horizon, dim].
    """
    preds = inverse_scale_sequences(predictions_scaled, target_scaler)
    tgts = inverse_scale_sequences(targets_scaled, target_scaler)

    report: Dict = {
        'overall': compute_metrics(preds, tgts),
        'per_target': {},
        'per_horizon_mae': [],
        'per_horizon_per_target_mae': {},
    }

    n_targets = len(target_names)
    for i, name in enumerate(target_names):
        p, t = preds[:, :, i], tgts[:, :, i]
        report['per_target'][name] = {
            'mae': float(np.mean(np.abs(p - t))),
            'rmse': float(np.sqrt(np.mean((p - t) ** 2))),
            'r2': _r2(t.flatten(), p.flatten()),
        }
        # MAE по шагам упреждения для этой переменной
        report['per_horizon_per_target_mae'][name] = [
            float(np.mean(np.abs(p[:, h] - t[:, h]))) for h in range(p.shape[1])
        ]

    horizon = preds.shape[1]
    report['per_horizon_mae'] = [
        float(np.mean(np.abs(preds[:, h, :] - tgts[:, h, :]))) for h in range(horizon)
    ]

    return report


def skill_score(model_mae: float, baseline_mae: float) -> float:
    """Skill score: 1 - MAE_модели / MAE_бейзлайна. > 0 — модель лучше бейзлайна."""
    if baseline_mae is None or baseline_mae == 0:
        return float('nan')
    return float(1.0 - model_mae / baseline_mae)
