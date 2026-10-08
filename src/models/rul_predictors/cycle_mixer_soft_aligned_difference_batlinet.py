"""Cycle-axis correction before the existing shared-value soft difference."""
from src.builders import MODELS
from .soft_aligned_difference_batlinet import SoftAlignedDifferenceBatLiNetRULPredictor
from .cycle_mixer_latent_cross_attention_batlinet import CycleEnhancedConvTokenEncoder


@MODELS.register()
class CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor(SoftAlignedDifferenceBatLiNetRULPredictor):
    """Change only the shared encoder; keep soft matching, heads and losses."""
    def __init__(self, in_channels, channels, input_height, input_width,
                 cycle_mixer_hidden=16, **kwargs):
        super().__init__(in_channels=in_channels, channels=channels,
                         input_height=input_height, input_width=input_width, **kwargs)
        self.cell_encoder = CycleEnhancedConvTokenEncoder(
            self.cell_encoder, input_height, hidden=cycle_mixer_hidden, kind='mlp')
