"""Parallel cycle/capacity residuals on the existing convolution feature grid."""
import torch
from torch import nn

from src.builders import MODELS
from .cycle_mixer_latent_cross_attention_batlinet import CycleAxisResidual
from .latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class CapacityAxisResidual(nn.Module):
    """Share one capacity-position MLP across batches, channels and cycle rows."""

    def __init__(self, positions, hidden=16):
        super().__init__()
        if positions < 1 or hidden < 1:
            raise ValueError('Capacity and hidden dimensions must be positive.')
        self.positions = positions
        self.net = nn.Sequential(
            nn.Linear(positions, hidden), nn.GELU(), nn.Linear(hidden, positions))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def correction(self, grid):
        if grid.ndim != 4 or grid.shape[-1] != self.positions:
            raise ValueError(f'Expected a [B, D, H, {self.positions}] feature grid.')
        return self.net(grid)

    def forward(self, grid):
        return grid + self.correction(grid)


class DualAxisConvTokenEncoder(nn.Module):
    """Apply G + R_cycle(G) + R_capacity(G) before the original final pool.

    Both residuals read G. Neither axis collapses the other axis, changes token
    order, or adds dropout. Keeping the existing cycle-mixer state keys permits
    direct functional comparisons with completed cycle-only checkpoints.
    """
    injection_index = 8

    def __init__(self, base, input_height, input_width, cycle_hidden=16,
                 capacity_hidden=16, enable_cycle=True, enable_capacity=True):
        super().__init__()
        if not isinstance(base.net[self.injection_index], nn.AvgPool2d):
            raise ValueError('The original encoder final-pooling layout changed.')
        self.base = base
        self.num_tokens = base.num_tokens
        self.token_channels = base.token_channels
        self.enable_cycle = enable_cycle
        self.enable_capacity = enable_capacity
        if enable_cycle:
            # Match the existing cycle-only initialization exactly.
            with torch.random.fork_rng(devices=[]):
                self.cycle_mixer = CycleAxisResidual(input_height // 2, cycle_hidden)
        if enable_capacity:
            # This separate fork also preserves subsequent sampling/dropout RNG.
            with torch.random.fork_rng(devices=[]):
                self.capacity_mixer = CapacityAxisResidual(
                    input_width // 4 // 4, capacity_hidden)

    def mix_grid(self, grid):
        mixed = self.cycle_mixer(grid) if self.enable_cycle else grid
        if self.enable_capacity:
            mixed = mixed + self.capacity_mixer.correction(grid)
        return mixed

    def forward(self, x):
        for index, layer in enumerate(self.base.net):
            if index == self.injection_index:
                x = self.mix_grid(x)
            x = layer(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        if tokens.size(1) != self.base.position.size(1):
            raise ValueError('The dual-axis encoder changed the original token grid.')
        return tokens + self.base.position


@MODELS.register()
class DualAxisLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Preserve the original predictor and replace only the shared encoder."""

    def __init__(self, in_channels, channels, input_height, input_width,
                 cycle_mixer_hidden=16, capacity_mixer_hidden=16,
                 enable_cycle=True, enable_capacity=True, **kwargs):
        super().__init__(in_channels=in_channels, channels=channels,
                         input_height=input_height, input_width=input_width, **kwargs)
        self.cell_encoder = DualAxisConvTokenEncoder(
            self.cell_encoder, input_height, input_width,
            cycle_mixer_hidden, capacity_mixer_hidden, enable_cycle, enable_capacity)
