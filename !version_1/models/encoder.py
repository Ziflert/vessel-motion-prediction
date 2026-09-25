import torch
import torch.nn as nn
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int],
            temporal_hidden_size: int,
            temporal_num_layers: int,
            dropout: float = 0.1,
            bidirectional: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.temporal_hidden_size = temporal_hidden_size
        self.temporal_num_layers = temporal_num_layers
        self.bidirectional = bidirectional

        # 1. Предварительная обработка (Linear layers)
        # Если hidden_dims пустой, сразу переходим к LSTM
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        # 2. LSTM
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=temporal_hidden_size,
            num_layers=temporal_num_layers,
            dropout=dropout if temporal_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            outputs: [batch_size, seq_len, hidden_size * directions]
            hidden: (h_n, c_n)
        """
        # Прогон через полносвязные слои (если есть)
        x = self.feature_extractor(x)

        # Прогон через LSTM
        outputs, (h_n, c_n) = self.lstm(x)

        return outputs, (h_n, c_n)