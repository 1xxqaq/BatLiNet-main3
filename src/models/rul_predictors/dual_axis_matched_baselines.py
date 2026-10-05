"""Reuse the existing preprocessing, support aggregation and prediction heads."""
from torch import nn

from .dual_axis_latent_cross_attention_batlinet import (
    DualAxisLatentCrossAttentionBatLiNetRULPredictor,
)
from .matched_baselines import MatchedBaseline


class DualAxisMatchedBaseline(MatchedBaseline):
    def __init__(self, architecture, cycles=20, width=1000,
                 cycle_mixer_hidden=16, capacity_mixer_hidden=16):
        if architecture not in ('latent_capacity_mixer', 'latent_dual_axis_mixer'):
            raise ValueError(f'Unknown dual-axis experiment: {architecture}')
        nn.Module.__init__(self)
        self.architecture = architecture
        self.alpha = .5
        self.base = DualAxisLatentCrossAttentionBatLiNetRULPredictor(
            in_channels=6, channels=32, input_height=cycles, input_width=width,
            filter_cycles=False, train_support_size=2, test_support_size=32,
            cycle_mixer_hidden=cycle_mixer_hidden,
            capacity_mixer_hidden=capacity_mixer_hidden,
            enable_cycle=architecture == 'latent_dual_axis_mixer')
