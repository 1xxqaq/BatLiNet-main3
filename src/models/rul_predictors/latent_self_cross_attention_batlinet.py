"""Shared intra-cell self-attention before BatLiNet cross-cell attention."""

import torch
import torch.nn as nn

from src.builders import MODELS

from .latent_cross_attention_batlinet import (
    ConvTokenEncoder,
    LatentCrossAttentionBatLiNetRULPredictor,
)


class TokenSelfAttentionBlock(nn.Module):
    """Pre-normalized attention over all observed local tokens of one cell."""

    def __init__(self, channels, num_heads, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * mlp_ratio, channels),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class SelfAttentiveConvTokenEncoder(nn.Module):
    """Keep the original local grid and refine it within each battery."""

    def __init__(self, local_encoder: ConvTokenEncoder, num_layers: int,
                 num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.local_encoder = local_encoder
        self.num_tokens = local_encoder.num_tokens
        self.token_channels = local_encoder.token_channels
        self.blocks = nn.ModuleList([
            TokenSelfAttentionBlock(
                self.token_channels, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        tokens = self.local_encoder(feature)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


@MODELS.register()
class LatentSelfCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Variant A: replace only the shared encoder, inherit both branches."""

    def __init__(self, *args, self_attention_layers: int = 1,
                 self_attention_heads: int = None,
                 self_attention_mlp_ratio: int = 2,
                 self_attention_dropout: float = 0.1, **kwargs):
        if not isinstance(self_attention_layers, int) or self_attention_layers < 0:
            raise ValueError('self_attention_layers must be a non-negative integer.')
        if (not isinstance(self_attention_mlp_ratio, int)
                or self_attention_mlp_ratio < 1):
            raise ValueError('self_attention_mlp_ratio must be a positive integer.')
        if not 0 <= self_attention_dropout <= 1:
            raise ValueError('self_attention_dropout must be between 0 and 1.')
        super().__init__(*args, **kwargs)
        heads = (kwargs.get('attention_heads', 4)
                 if self_attention_heads is None else self_attention_heads)
        if not isinstance(heads, int) or heads < 1 or self.channels % heads:
            raise ValueError(
                'self_attention_heads must be positive and divide attention_channels.')
        # With zero layers, preserve the base encoder and checkpoint keys exactly.
        if self_attention_layers:
            self.cell_encoder = SelfAttentiveConvTokenEncoder(
                self.cell_encoder, self_attention_layers, heads,
                self_attention_mlp_ratio, self_attention_dropout)
