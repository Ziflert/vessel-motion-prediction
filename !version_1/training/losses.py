import torch
import torch.nn as nn


class VesselLoss(nn.Module):
    """Функция потерь"""

    def __init__(self, weights=None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()

        # Веса для разных компонент loss (если не заданы, используем дефолтные)
        if weights is None:
            weights = {'mse': 1.0, 'direction': 0.0, 'physics': 0.0}
        self.weights = weights

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor
    ) -> dict:
        """
        Args:
            predictions: (batch, pred_steps, output_dim)
            targets: (batch, pred_steps, output_dim)
        Returns:
            dict с ключами 'total_loss', 'mse_loss', 'huber_loss', 'smoothness_loss'
        """
        # Основной loss
        mse_loss = self.mse(predictions, targets)

        # Huber loss для робастности
        huber_loss = self.huber(predictions, targets)

        # Temporal smoothness
        if predictions.size(1) > 1:
            smoothness = torch.mean(
                (predictions[:, 1:] - predictions[:, :-1]) ** 2
            )
        else:
            smoothness = torch.tensor(0.0, device=predictions.device)

        total_loss = mse_loss + 0.5 * huber_loss + 0.1 * smoothness

        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'huber_loss': huber_loss,
            'smoothness_loss': smoothness
        }