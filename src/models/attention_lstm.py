"""
Module containing LSTM model with attention mechanism.
"""

import torch
import torch.nn as nn


class HybridLSTMAttn(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int = 30,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dense_hidden: int = 100,
        forecast_horizon: int = 30,
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        """
        Initialize LSTM model with attention mechanism.
        
        Args:
            input_dim: Number of input features
            seq_len: Length of input sequence
            lstm_hidden: Number of LSTM hidden units
            lstm_layers: Number of LSTM layers
            dense_hidden: Number of dense layer hidden units
            forecast_horizon: Number of time steps to forecast
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Добавляем слой для преобразования входных данных в последовательность
        self.input_projection = nn.Linear(input_dim, lstm_hidden)
        
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(lstm_hidden)
        attn_output_dim = lstm_hidden * seq_len
        
        combined_dim = (
            lstm_hidden * seq_len +  # LSTM output
            attn_output_dim         # Attention output
        )
        
        self.final_net = nn.Sequential(
            nn.Linear(combined_dim, dense_hidden),
            nn.LayerNorm(dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, dense_hidden // 2),
            nn.LayerNorm(dense_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden // 2, forecast_horizon)
        )
        
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            attention_mask: Attention mask tensor
            
        Returns:
            Predictions tensor of shape (batch_size, forecast_horizon)
        """
        batch_size = x.size(0)
        
        # Если входные данные двумерные, добавляем размерность последовательности
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            x = x.repeat(1, self.seq_len, 1)  # (batch_size, seq_len, input_dim)
        
        # Проецируем входные данные в пространство LSTM
        x = self.input_projection(x)  # (batch_size, seq_len, lstm_hidden)
        
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden)
        
        key_padding_mask = (x == -1).all(dim=-1)
        
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=key_padding_mask
        )  # (batch_size, seq_len, lstm_hidden)
        attn_out = self.attention_norm(lstm_out + attn_out)
        attn_out = attn_out.reshape(batch_size, -1)
            
        lstm_out_flat = lstm_out.reshape(batch_size, -1)
        combined = torch.cat([
            lstm_out_flat,   # LSTM output
            attn_out,        # Attention output
        ], dim=1)
        
        return self.final_net(combined)  # (batch_size, forecast_horizon)