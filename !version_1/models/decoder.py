import torch
import torch.nn as nn
from version_1.models.attention import Attention


class Decoder(nn.Module):
    def __init__(
            self,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            use_attention: bool,
            enc_hidden_dim: int
    ):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Входная размерность LSTM декодера:
        # Если есть внимание: input_dim + context_vector (размер энкодера)
        # Если нет: input_dim
        lstm_input_dim = output_dim + (enc_hidden_dim if use_attention else 0)

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        if use_attention:
            self.attention = Attention(enc_hidden_dim, hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            hidden: tuple,
            encoder_outputs: torch.Tensor
    ):
        """
        Один шаг декодера.

        Args:
            x: [batch_size, output_dim] (вход на текущем шаге)
            hidden: tuple (h, c) - состояния LSTM
                h: [num_layers, batch_size, hidden_dim]
                c: [num_layers, batch_size, hidden_dim]
            encoder_outputs: [batch_size, seq_len, enc_hidden_dim]

        Returns:
            prediction: [batch_size, output_dim]
            hidden: tuple (h, c) - обновленные состояния
            attention_weights: [batch_size, seq_len] или None
        """
        # x: [batch_size, output_dim] -> [batch_size, 1, output_dim]
        x = x.unsqueeze(1)

        attention_weights = None

        if self.use_attention:
            # Получаем context vector из внимания
            # hidden[0] это h_n: [num_layers, batch, hidden_dim]
            # Берем последний слой: hidden[0][-1] -> [batch, hidden_dim]
            context = self.attention(hidden[0][-1], encoder_outputs)

            # context: [batch_size, enc_hidden_dim] -> [batch_size, 1, enc_hidden_dim]
            context = context.unsqueeze(1)

            # Соединяем вход и контекст
            lstm_input = torch.cat((x, context), dim=2)
        else:
            lstm_input = x

        # Шаг LSTM
        # output: [batch_size, 1, hidden_dim]
        # hidden: tuple ([num_layers, batch, hidden_dim], [num_layers, batch, hidden_dim])
        output, hidden = self.lstm(lstm_input, hidden)

        # Предсказание: [batch_size, 1, hidden_dim] -> [batch_size, output_dim]
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, attention_weights