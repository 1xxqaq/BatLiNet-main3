"""Preserve the original local grid and add a cycle-axis residual correction."""
import torch
from torch import nn

from src.builders import MODELS
from .latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)


class CycleAxisResidual(nn.Module):
    """Use the same cycle-axis network at every capacity position and channel.

    The MLP maps H -> hidden -> H, rather than collapsing H to one summary.
    The convolution control uses two local length-three convolutions. Its
    hidden width is chosen to approximately match the MLP parameter count.
    Neither variant mixes capacity positions or introduces another dropout.
    """

    def __init__(self, cycles, hidden=16, kind='mlp', conv_kernel=3):
        super().__init__()
        if cycles < 1 or hidden < 1:
            raise ValueError('Cycle and hidden dimensions must be positive.')
        if kind not in ('mlp', 'conv'):
            raise ValueError(f'Unknown cycle mixer: {kind}')
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError('The convolution control needs a positive odd kernel.')
        self.cycles, self.kind = cycles, kind
        self.mlp_parameters = 2 * cycles * hidden + hidden + cycles
        if kind == 'mlp':
            self.net = nn.Sequential(
                nn.Linear(cycles, hidden), nn.GELU(), nn.Linear(hidden, cycles))
        else:
            control_hidden = max(1, (self.mlp_parameters - 1) // (2 * conv_kernel + 1))
            self.net = nn.Sequential(
                nn.Conv1d(1, control_hidden, conv_kernel, padding=conv_kernel // 2),
                nn.GELU(),
                nn.Conv1d(control_hidden, 1, conv_kernel, padding=conv_kernel // 2))
        # At initialization the entire encoder is exactly the original encoder.
        # Earlier mixer weights receive gradients after the output layer moves.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, grid):
        if grid.ndim != 4 or grid.shape[2] != self.cycles:
            raise ValueError(f'Expected a [B, D, {self.cycles}, M] feature grid.')
        sequence = grid.movedim(2, -1)  # [B, D, M, H], with fixed cycle ordering.
        if self.kind == 'mlp':
            correction = self.net(sequence)
        else:
            correction = self.net(sequence.reshape(-1, 1, self.cycles))
            correction = correction.reshape_as(sequence)
        return grid + correction.movedim(-1, 2)


class CycleEnhancedConvTokenEncoder(nn.Module):
    """Insert a correction immediately before the original final pooling."""
    injection_index = 8

    def __init__(self, base, input_height, hidden=16, kind='mlp', conv_kernel=3):
        super().__init__()
        if not isinstance(base.net[self.injection_index], nn.AvgPool2d):
            raise ValueError('The original encoder final-pooling layout changed.')
        self.base = base
        self.num_tokens = base.num_tokens
        self.token_channels = base.token_channels
        # Isolate auxiliary initialization from the original training RNG.
        # Same-seed original weights, batch permutations, reference draws and
        # dropout draws start identically in the baseline and both variants.
        with torch.random.fork_rng(devices=[]):
            self.cycle_mixer = CycleAxisResidual(
                input_height // 2, hidden, kind, conv_kernel)

    def forward(self, x):
        for index, layer in enumerate(self.base.net):
            if index == self.injection_index:
                x = self.cycle_mixer(x)
            x = layer(x)
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        if tokens.size(1) != self.base.position.size(1):
            raise ValueError('The enhanced encoder changed the original token grid.')
        return tokens + self.base.position


@MODELS.register()
class CycleMixerLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Original predictor, with only its shared convolution encoder enhanced."""

    def __init__(self, in_channels, channels, input_height, input_width,
                 cycle_mixer_hidden=16, cycle_mixer_type='mlp',
                 cycle_conv_kernel=3, **kwargs):
        super().__init__(in_channels=in_channels, channels=channels,
                         input_height=input_height, input_width=input_width, **kwargs)
        self.cell_encoder = CycleEnhancedConvTokenEncoder(
            self.cell_encoder, input_height, cycle_mixer_hidden,
            cycle_mixer_type, cycle_conv_kernel)
