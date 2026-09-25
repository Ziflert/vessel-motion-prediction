import torch
import torch.nn as nn
import random
from typing import Optional

from version_1.models.encoder import Encoder
from version_1.models.decoder import Decoder


class VesselPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = config.device

        # ---------------------------------------------------------------------
        # 1. ENCODER
        # ---------------------------------------------------------------------
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dims=config.encoder_hidden_dims,
            temporal_hidden_size=config.temporal_hidden_size,
            temporal_num_layers=config.temporal_num_layers,
            dropout=config.encoder_dropout,
            bidirectional=config.bidirectional
        )

        # Определяем размер скрытого состояния энкодера
        enc_hidden_size = config.temporal_hidden_size * (2 if config.bidirectional else 1)

        # ---------------------------------------------------------------------
        # 2. DECODER
        # ---------------------------------------------------------------------
        self.decoder = Decoder(
            output_dim=output_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_layers=config.temporal_num_layers,  # Используем то же количество слоев
            dropout=config.decoder_dropout,
            use_attention=config.use_attention,
            enc_hidden_dim=enc_hidden_size
        )

        # ---------------------------------------------------------------------
        # 3. BRIDGE LAYER (для согласования размерностей encoder -> decoder)
        # ---------------------------------------------------------------------
        # Encoder hidden: [num_layers * directions, batch, temporal_hidden_size]
        # Decoder hidden: [num_layers, batch, decoder_hidden_dim]

        self.bridge_h = nn.Linear(enc_hidden_size, config.decoder_hidden_dim)
        self.bridge_c = nn.Linear(enc_hidden_size, config.decoder_hidden_dim)

    def _bridge_hidden(self, encoder_hidden):
        """
        Преобразует скрытое состояние энкодера в формат для декодера

        Args:
            encoder_hidden: tuple (h_n, c_n)
                h_n: [num_layers * directions, batch, temporal_hidden_size]
                c_n: [num_layers * directions, batch, temporal_hidden_size]

        Returns:
            decoder_hidden: tuple (h_n, c_n)
                h_n: [num_layers, batch, decoder_hidden_dim]
                c_n: [num_layers, batch, decoder_hidden_dim]
        """
        h_n, c_n = encoder_hidden

        if self.config.bidirectional:
            # Объединяем forward и backward направления
            # [num_layers * 2, batch, hidden] -> [num_layers, batch, hidden * 2]
            num_layers = h_n.shape[0] // 2
            batch_size = h_n.shape[1]

            # Reshape и concat
            h_forward = h_n[0::2]  # [num_layers, batch, hidden]
            h_backward = h_n[1::2]  # [num_layers, batch, hidden]
            h_combined = torch.cat([h_forward, h_backward], dim=2)  # [num_layers, batch, hidden*2]

            c_forward = c_n[0::2]
            c_backward = c_n[1::2]
            c_combined = torch.cat([c_forward, c_backward], dim=2)

            h_n = h_combined
            c_n = c_combined

        # Проецируем в decoder размерность
        # [num_layers, batch, enc_hidden] -> [num_layers, batch, dec_hidden]
        h_n = self.bridge_h(h_n)
        c_n = self.bridge_c(c_n)

        return (h_n, c_n)

    def forward(
            self,
            x: torch.Tensor,
            target: Optional[torch.Tensor] = None,
            teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass модели Seq2Seq.

        Args:
            x: Входные данные [batch_size, seq_len, input_dim]
            target: Реальные будущие значения [batch_size, pred_horizon, output_dim]
            teacher_forcing_ratio: Вероятность использования реальных данных

        Returns:
            outputs: [batch_size, pred_horizon, output_dim]
        """
        batch_size = x.shape[0]
        max_len = self.config.prediction_horizon

        # КРИТИЧНО: Определяем device из входных данных
        device = x.device

        # 1. Прогоняем данные через Энкодер
        encoder_outputs, encoder_hidden = self.encoder(x)

        # 2. Преобразуем hidden состояния для декодера
        decoder_hidden = self._bridge_hidden(encoder_hidden)

        # 3. Подготовка выходного тензора (на том же device!)
        outputs = torch.zeros(batch_size, max_len, self.output_dim, device=device)

        # 4. Первый вход декодера - нули (на том же device!)
        decoder_input = torch.zeros(batch_size, self.output_dim, device=device)

        # 5. Цикл предсказания (шаг за шагом)
        for t in range(max_len):
            # Шаг декодера
            decoder_output, decoder_hidden, _ = self.decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

            # Сохраняем предсказание
            outputs[:, t, :] = decoder_output

            # Teacher Forcing: решаем, какой вход подать на следующий шаг
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force and target is not None:
                decoder_input = target[:, t, :]
            else:
                decoder_input = decoder_output

        return outputs