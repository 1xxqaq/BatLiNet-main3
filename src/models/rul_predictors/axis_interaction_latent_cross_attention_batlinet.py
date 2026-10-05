"""Exchange axial summaries, then return a correction to the local grid."""
import torch
from torch import nn

from src.builders import MODELS
from .dual_axis_latent_cross_attention_batlinet import DualAxisConvTokenEncoder
from .latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class AxisGridFusion(nn.Module):
    """Keep local grids; only the interaction path pools the opposite axis.

    Cross and within modes have identical parameters and initialization. In
    within mode each query reads its own axis, providing an active-parameter
    control for exchanging information across the two views. Neither mode adds
    stochastic dropout. A zero final projection starts at the cycle backbone.
    """

    def __init__(self, channels, cycles, positions, heads=4, hidden=32,
                 interaction_mode='cross'):
        super().__init__()
        if interaction_mode not in ('cross', 'within'):
            raise ValueError('interaction_mode must be cross or within.')
        if min(channels, cycles, positions, heads, hidden) < 1 or channels % heads:
            raise ValueError('Positive dimensions and channels divisible by heads are required.')
        self.interaction_mode = interaction_mode
        self.cycle_position = nn.Parameter(torch.empty(1, cycles, channels))
        self.capacity_position = nn.Parameter(torch.empty(1, positions, channels))
        nn.init.trunc_normal_(self.cycle_position, std=.02)
        nn.init.trunc_normal_(self.capacity_position, std=.02)
        self.cycle_norm = nn.LayerNorm(channels)
        self.capacity_norm = nn.LayerNorm(channels)
        self.cycle_attention = nn.MultiheadAttention(
            channels, heads, dropout=0., batch_first=True)
        self.capacity_attention = nn.MultiheadAttention(
            channels, heads, dropout=0., batch_first=True)
        self.local_fusion = nn.Sequential(
            nn.LayerNorm(channels * 4),
            nn.Linear(channels * 4, hidden), nn.GELU(),
            nn.Linear(hidden, channels))
        nn.init.zeros_(self.local_fusion[-1].weight)
        nn.init.zeros_(self.local_fusion[-1].bias)

    def axis_contexts(self, cycle_grid, capacity_grid):
        if (cycle_grid.shape != capacity_grid.shape or cycle_grid.ndim != 4
                or cycle_grid.shape[1:] != (
                    self.cycle_position.shape[-1], self.cycle_position.shape[1],
                    self.capacity_position.shape[1])):
            raise ValueError('Both views must keep the configured [B, D, H, W] grid.')
        # Summaries are additional context, never replacements for local grids.
        cycle = cycle_grid.mean(dim=3).transpose(1, 2) + self.cycle_position
        capacity = capacity_grid.mean(dim=2).transpose(1, 2) + self.capacity_position
        cycle_norm, capacity_norm = self.cycle_norm(cycle), self.capacity_norm(capacity)
        cycle_kv, capacity_kv = ((capacity_norm, cycle_norm)
                               if self.interaction_mode == 'cross'
                               else (cycle_norm, capacity_norm))
        cycle_update, _ = self.cycle_attention(cycle_norm, cycle_kv, cycle_kv,
                                             need_weights=False)
        capacity_update, _ = self.capacity_attention(capacity_norm, capacity_kv,
                                                   capacity_kv, need_weights=False)
        return cycle + cycle_update, capacity + capacity_update

    def forward(self, cycle_grid, capacity_grid):
        cycle_context, capacity_context = self.axis_contexts(cycle_grid, capacity_grid)
        local_cycle, local_capacity = cycle_grid.permute(0, 2, 3, 1), capacity_grid.permute(0, 2, 3, 1)
        cycle_context = cycle_context.unsqueeze(2).expand_as(local_cycle)
        capacity_context = capacity_context.unsqueeze(1).expand_as(local_capacity)
        local = torch.cat((local_cycle, local_capacity, cycle_context, capacity_context), dim=-1)
        return self.local_fusion(local).permute(0, 3, 1, 2).contiguous()


class AxisInteractionConvTokenEncoder(DualAxisConvTokenEncoder):
    def __init__(self, base, input_height, input_width, cycle_hidden=16,
                 capacity_hidden=16, heads=4, fusion_hidden=32,
                 interaction_mode='cross'):
        super().__init__(base, input_height, input_width, cycle_hidden, capacity_hidden)
        # Keep original weights, cycle/capacity weights and later reference and
        # dropout sampling at the same RNG state as the existing encoders.
        with torch.random.fork_rng(devices=[]):
            self.axis_fusion = AxisGridFusion(
                base.token_channels, input_height // 2, input_width // 4 // 4,
                heads, fusion_hidden, interaction_mode)

    def mix_grid(self, grid):
        cycle = self.cycle_mixer(grid)
        capacity = self.capacity_mixer(grid)
        # This is intentionally an asymmetric exploratory model. Cross-attention
        # is bidirectional, but the final skip connection keeps the cycle view.
        return cycle + self.axis_fusion(cycle, capacity)


@MODELS.register()
class AxisInteractionLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Keep original preprocessing, local tokens, support matching and heads."""

    def __init__(self, in_channels, channels, input_height, input_width,
                 cycle_mixer_hidden=16, capacity_mixer_hidden=16,
                 axis_attention_heads=4, axis_fusion_hidden=32,
                 interaction_mode='cross', **kwargs):
        super().__init__(in_channels=in_channels, channels=channels,
                         input_height=input_height, input_width=input_width, **kwargs)
        self.cell_encoder = AxisInteractionConvTokenEncoder(
            self.cell_encoder, input_height, input_width,
            cycle_mixer_hidden, capacity_mixer_hidden,
            axis_attention_heads, axis_fusion_hidden, interaction_mode)
