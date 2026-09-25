import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class LSTMBlock(nn.Module):
    """LSTM блок"""

    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            num_layers: int,
            dropout: float = 0.2,
            bidirectional: bool = True
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        output, hidden = self.lstm(x)
        output = self.layer_norm(output)
        return output, hidden


class TransformerBlock(nn.Module):
    """Transformer блок"""

    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        return self.transformer(x)


class HybridBlock(nn.Module):
    """Гибридный блок: LSTM + Transformer"""

    def __init__(
            self,
            input_dim: int,
            hidden_size: int,
            num_lstm_layers: int,
            num_attention_heads: int,
            num_transformer_layers: int,
            ff_dim: int,
            dropout: float = 0.2
    ):
        super().__init__()

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_size * 2

        # Transformer
        self.pos_encoder = PositionalEncoding(lstm_output_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_output_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.output_dim = lstm_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = self.pos_encoder(lstm_out)
        lstm_out = lstm_out.transpose(0, 1)

        return self.transformer(lstm_out)