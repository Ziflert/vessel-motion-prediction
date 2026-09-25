import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, num_heads: int = 1):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        # Простой механизм внимания (Dot Product / Luong)
        # Мы проецируем энкодер и декодер в одно пространство
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Args:
            hidden: [batch_size, dec_hidden_dim] - текущее скрытое состояние декодера
            encoder_outputs: [batch_size, seq_len, enc_hidden_dim]
        Returns:
            context: [batch_size, enc_hidden_dim] - взвешенная сумма выходов энкодера
            weights: [batch_size, seq_len] - веса внимания (для интерпретации)
        """
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        # Повторяем hidden state для каждого шага времени энкодера
        # hidden: [batch_size, dec_hidden_dim] -> [batch_size, seq_len, dec_hidden_dim]
        hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Конкатенируем hidden декодера и outputs энкодера
        # energy: [batch_size, seq_len, dec_hidden_dim]
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)

        energy = torch.tanh(self.attn(combined))

        # attention: [batch_size, seq_len]
        attention = self.v(energy).squeeze(2)

        # weights: [batch_size, seq_len] (сумма = 1)
        weights = F.softmax(attention, dim=1)

        # context: [batch_size, 1, seq_len] @ [batch_size, seq_len, enc_hidden_dim]
        #        = [batch_size, 1, enc_hidden_dim]
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)

        return context.squeeze(1), weights