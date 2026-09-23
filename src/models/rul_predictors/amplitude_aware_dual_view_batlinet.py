"""Dual-view BatLiNet with a magnitude-preserving path in the shared stem."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.builders import MODELS

from .dual_view_latent_cross_attention_batlinet import (
    DualViewCellEncoder,
    DualViewLatentCrossAttentionBatLiNetRULPredictor,
)


class AmplitudeAwareStem(nn.Module):
    """Add bounded raw-amplitude evidence beside the normalized curve stem.

    GroupNorm in the original stem nearly removes common positive input scaling.
    The parallel path keeps signed magnitude information without changing the
    cycle/capacity token geometry or the target-support interaction.
    """

    def __init__(self,
                 normalized_stem: nn.Module,
                 in_channels: int,
                 channels: int,
                 capacity_bins: int,
                 gate_init: float):
        super().__init__()
        self.normalized_stem = normalized_stem
        self.capacity_bins = capacity_bins
        self.amplitude_projection = nn.Conv1d(
            in_channels, channels, kernel_size=1, bias=False)
        self.gate_logit = nn.Parameter(
            torch.full((1, channels, 1), float(gate_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_stem(x)
        # Compress extreme values while retaining their sign and magnitude.
        compressed = x.sign() * torch.log1p(x.abs())
        amplitude = self.amplitude_projection(compressed)
        amplitude = F.adaptive_avg_pool1d(
            amplitude, self.capacity_bins)
        return normalized + torch.sigmoid(self.gate_logit) * amplitude


class AmplitudeAwareDualViewCellEncoder(DualViewCellEncoder):
    """Keep the existing dual-view encoder, changing only its local stem."""

    def __init__(self, *args, amplitude_gate_init: float = -2.0, **kwargs):
        in_channels = kwargs.get('in_channels', args[0] if args else None)
        if in_channels is None:
            raise ValueError('in_channels is required.')
        super().__init__(*args, **kwargs)
        self.local_stem = AmplitudeAwareStem(
            self.local_stem,
            in_channels=in_channels,
            channels=self.channels,
            capacity_bins=self.capacity_bins,
            gate_init=amplitude_gate_init,
        )


@MODELS.register()
class AmplitudeAwareDualViewLatentCrossAttentionBatLiNetRULPredictor(
        DualViewLatentCrossAttentionBatLiNetRULPredictor):
    """Test whether retaining curve amplitude helps dual-view BatLiNet."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 attention_channels: int = 64,
                 attention_heads: int = 4,
                 attention_mlp_ratio: int = 2,
                 encoder_dropout: float = 0.1,
                 capacity_bins: int = 32,
                 stem_channels: int = 32,
                 cycle_encoder_layers: int = 1,
                 amplitude_gate_init: float = -2.0,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_height=input_height,
            input_width=input_width,
            attention_channels=attention_channels,
            attention_heads=attention_heads,
            attention_mlp_ratio=attention_mlp_ratio,
            encoder_dropout=encoder_dropout,
            capacity_bins=capacity_bins,
            stem_channels=stem_channels,
            cycle_encoder_layers=cycle_encoder_layers,
            **kwargs,
        )
        self.cell_encoder = AmplitudeAwareDualViewCellEncoder(
            in_channels=in_channels,
            channels=attention_channels,
            input_height=input_height,
            input_width=input_width,
            capacity_bins=capacity_bins,
            stem_channels=stem_channels,
            attention_heads=attention_heads,
            cycle_encoder_layers=cycle_encoder_layers,
            mlp_ratio=attention_mlp_ratio,
            dropout=encoder_dropout,
            amplitude_gate_init=amplitude_gate_init,
        )
