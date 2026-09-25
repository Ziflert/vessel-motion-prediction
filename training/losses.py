import torch
import torch.nn as nn
from typing import Dict, Optional


class VesselLoss(nn.Module):
    """
    Функция потерь с поддержкой взвешивания по целевым переменным

    Поддерживает:
    - Веса для каждой целевой переменной (например, Roll важнее чем Pitch)
    - MSE + Huber Loss для робастности
    - Temporal smoothness (штраф за резкие скачки)
    - Per-variable metrics для анализа
    """

    def __init__(
            self,
            target_weights: Optional[Dict[str, float]] = None,
            target_names: Optional[list] = None,
            loss_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            target_weights: Веса для каждой целевой переменной {'Roll(градусы)': 2.0, ...}
            target_names: Список имён целевых переменных (для маппинга индексов)
            loss_weights: Веса для компонент loss {'mse': 1.0, 'huber': 0.5, 'smoothness': 0.1}
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='none')  # Per-element loss
        self.huber = nn.SmoothL1Loss(reduction='none')

        # Веса для целевых переменных
        self.target_weights = target_weights
        self.target_names = target_names

        # Веса для компонент loss
        if loss_weights is None:
            loss_weights = {
                'mse': 1.0,
                'huber': 0.5,
                'smoothness': 0.1
            }
        self.loss_weights = loss_weights

        # Создаем тензор весов для целевых переменных
        if target_weights is not None and target_names is not None:
            self.register_buffer('target_weight_tensor',
                                 self._create_weight_tensor(target_weights, target_names))
        else:
            self.target_weight_tensor = None

    def _create_weight_tensor(self, target_weights: Dict[str, float],
                              target_names: list) -> torch.Tensor:
        """
        Создаёт тензор весов [output_dim] для быстрого применения
        """
        weights = []
        for name in target_names:
            weight = target_weights.get(name, 1.0)  # Default weight = 1.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: (batch, pred_steps, output_dim)
            targets: (batch, pred_steps, output_dim)

        Returns:
            dict с ключами:
                'total_loss': итоговый loss
                'mse_loss': MSE компонента
                'huber_loss': Huber компонента
                'smoothness_loss': сглаживание
                'per_variable_loss': loss для каждой переменной (если доступны имена)
        """

        # =====================================================================
        # 1. MSE LOSS (per-element)
        # =====================================================================
        mse_loss_elements = self.mse(predictions, targets)  # (batch, pred_steps, output_dim)

        # Применяем веса для целевых переменных
        if self.target_weight_tensor is not None:
            # Переносим веса на то же устройство
            weights = self.target_weight_tensor.to(predictions.device)
            # Расширяем веса до (1, 1, output_dim) для broadcasting
            weights = weights.view(1, 1, -1)
            # Взвешиваем
            mse_loss_elements = mse_loss_elements * weights

        # Усредняем
        mse_loss = mse_loss_elements.mean()

        # =====================================================================
        # 2. HUBER LOSS (для робастности к выбросам)
        # =====================================================================
        huber_loss_elements = self.huber(predictions, targets)

        if self.target_weight_tensor is not None:
            weights = self.target_weight_tensor.to(predictions.device).view(1, 1, -1)
            huber_loss_elements = huber_loss_elements * weights

        huber_loss = huber_loss_elements.mean()

        # =====================================================================
        # 3. TEMPORAL SMOOTHNESS (штраф за резкие скачки)
        # =====================================================================
        if predictions.size(1) > 1:
            # Разница между соседними временными шагами
            pred_diff = predictions[:, 1:] - predictions[:, :-1]
            target_diff = targets[:, 1:] - targets[:, :-1]

            # L2 норма разницы
            smoothness = torch.mean((pred_diff - target_diff) ** 2)
        else:
            smoothness = torch.tensor(0.0, device=predictions.device)

        # =====================================================================
        # 4. КОМБИНИРУЕМ LOSS
        # =====================================================================
        total_loss = (
                self.loss_weights['mse'] * mse_loss +
                self.loss_weights['huber'] * huber_loss +
                self.loss_weights['smoothness'] * smoothness
        )

        # =====================================================================
        # 5. PER-VARIABLE METRICS (для анализа)
        # =====================================================================
        per_variable_loss = {}

        if self.target_names is not None:
            # Вычисляем MSE для каждой переменной отдельно
            for i, name in enumerate(self.target_names):
                var_loss = mse_loss_elements[:, :, i].mean()
                per_variable_loss[name] = var_loss.item()

        # =====================================================================
        # ВОЗВРАЩАЕМ ВСЁ
        # =====================================================================
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'huber_loss': huber_loss,
            'smoothness_loss': smoothness,
            'per_variable_loss': per_variable_loss  # Может быть пустым
        }