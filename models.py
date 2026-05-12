import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class MarketInfoAugmenter(nn.Module):
    """Lightweight self-attention over timesteps for market interpolation."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        interpolated = torch.matmul(attn_weights, v)
        return x + interpolated


class ConditionalMarketMixer(nn.Module):
    """Injects pooled market context into each timestep representation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        market_state = x.mean(dim=1, keepdim=True)
        mixed = torch.cat([x, market_state.expand_as(x)], dim=-1)
        mixed = self.proj(mixed)
        batch, steps, hidden = mixed.shape
        mixed = mixed.reshape(batch * steps, hidden)
        mixed = self.bn(mixed)
        mixed = self.act(mixed)
        return mixed.reshape(batch, steps, hidden)


class CryptoMixerMLP(nn.Module):
    """
    Lightweight MLP architecture inspired by CryptoMixer for DEX activity prediction.

    Ablations can be toggled by disabling the temporal stream, user stream, or market mixer.
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 96,
        seq_len: int = 10,
        num_users: int = 100,
        dropout: float = 0.15,
        use_temporal_stream: bool = True,
        use_user_stream: bool = True,
        use_market_mixer: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal_stream = use_temporal_stream
        self.use_user_stream = use_user_stream
        self.use_market_mixer = use_market_mixer

        if not (use_temporal_stream or use_user_stream):
            raise ValueError("At least one stream must remain enabled.")

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.market_augmenter = MarketInfoAugmenter(hidden_dim, dropout=dropout)
        self.market_mixer = ConditionalMarketMixer(hidden_dim)

        self.temporal_positional = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        self.temporal_proj = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_bn = nn.BatchNorm1d(hidden_dim)
        self.temporal_gate = nn.Linear(hidden_dim, 1)

        self.user_embedding = nn.Embedding(num_users, hidden_dim)
        self.user_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.user_bn = nn.BatchNorm1d(hidden_dim)

        fusion_width = hidden_dim * (use_temporal_stream + use_user_stream)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_width, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _batch_norm_sequence(self, bn_layer: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        batch, steps, hidden = x.shape
        x = x.reshape(batch * steps, hidden)
        x = bn_layer(x)
        return x.reshape(batch, steps, hidden)

    def forward(self, x: torch.Tensor, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if user_ids is None:
            user_ids = x[:, -1, -1].round().long().clamp_min(0)

        hidden = self.input_proj(x)
        hidden = self.market_augmenter(hidden)

        if self.use_market_mixer:
            hidden = self.market_mixer(hidden)

        fused_parts = []

        if self.use_temporal_stream:
            temporal_hidden = hidden + self.temporal_positional[:, : hidden.size(1), :]
            temporal_hidden = self.temporal_proj(temporal_hidden)
            temporal_hidden = self._batch_norm_sequence(self.temporal_bn, temporal_hidden)
            temporal_hidden = F.gelu(temporal_hidden)
            temporal_scores = self.temporal_gate(temporal_hidden).squeeze(-1)
            temporal_weights = torch.softmax(temporal_scores, dim=1)
            temporal_repr = torch.sum(temporal_hidden * temporal_weights.unsqueeze(-1), dim=1)
            fused_parts.append(temporal_repr)

        if self.use_user_stream:
            user_vec = self.user_embedding(user_ids)
            user_summary = hidden.mean(dim=1)
            adaptive_gate = torch.sigmoid(
                torch.sum(user_summary * user_vec, dim=-1, keepdim=True) / math.sqrt(user_summary.size(-1))
            )
            user_hidden = torch.cat(
                [adaptive_gate * user_summary, (1.0 - adaptive_gate) * user_vec], dim=-1
            )
            user_hidden = self.user_proj(user_hidden)
            user_hidden = self.user_bn(user_hidden)
            user_repr = F.gelu(user_hidden)
            fused_parts.append(user_repr)

        if len(fused_parts) == 1:
            fused = fused_parts[0]
        else:
            fused = torch.cat(fused_parts, dim=-1)

        fused = self.fusion(fused)
        return self.output_head(fused).squeeze(-1)


class BaselineLSTM(nn.Module):
    """Two-layer LSTM baseline."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        del user_ids
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class BaselineGRU(nn.Module):
    """Two-layer GRU baseline."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, user_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        del user_ids
        output, _ = self.gru(x)
        return self.head(output[:, -1, :]).squeeze(-1)
