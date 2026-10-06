"""Controlled raw/self-cycle-difference inputs for the unchanged cycle backbone."""
import torch

from src.builders import MODELS
from .cycle_mixer_latent_cross_attention_batlinet import (
    CycleMixerLatentCrossAttentionBatLiNetRULPredictor,
)


@MODELS.register()
class CycleDifferenceBatLiNetRULPredictor(
        CycleMixerLatentCrossAttentionBatLiNetRULPredictor):
    """Clean raw curves once, then optionally subtract each cell's first cycle.

    Target and reference cells use the same independent transformation. Zero
    cycles stay zero. Prepared reference tensors must not be prepared twice.
    No new parameters or RNG draws are introduced by either input mode.
    """

    def __init__(self, *args, input_mode='raw', difference_base=0, **kwargs):
        if input_mode not in ('raw', 'self_difference'):
            raise ValueError('输入模式必须为 raw 或 self_difference。')
        height = kwargs.get('input_height')
        if not isinstance(difference_base, int) or difference_base < 0:
            raise ValueError('差分基准必须为非负循环索引。')
        if height is not None and difference_base >= height:
            raise ValueError('差分基准超过输入循环数。')
        super().__init__(*args, **kwargs)
        self.input_mode = input_mode
        self.difference_base = difference_base

    @torch.no_grad()
    def _prepare_feature(self, feature):
        prepared = super()._prepare_feature(feature)
        if self.input_mode == 'raw':
            return prepared
        if self.difference_base >= prepared.shape[2]:
            raise ValueError('差分基准超过实际输入循环数。')
        valid = prepared.abs().amax(dim=(1, 3)).ne(0)
        if not valid[:, self.difference_base].all():
            raise ValueError('部分电池的差分基准循环无效；不自动改换基准。')
        baseline = prepared[:, :, self.difference_base:self.difference_base + 1]
        difference = prepared - baseline
        # In particular, MIX-20 index 10 must not become -baseline.
        return difference.masked_fill(~valid[:, None, :, None], 0.).contiguous()
