"""Adapters only: original baseline source files and architectures stay intact."""
from types import SimpleNamespace

import torch
from torch import nn

from .batlinet import BatLiNetRULPredictor
from .latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class MatchedBaseline(nn.Module):
    def __init__(self, architecture, cycles=20, width=1000):
        super().__init__()
        self.architecture = architecture
        self.alpha = .5
        options = dict(in_channels=6, channels=32, input_height=cycles,
                       input_width=width, filter_cycles=False,
                       train_support_size=2, test_support_size=32)
        if architecture == 'batlinet':
            self.base = BatLiNetRULPredictor(diff_base=0, **options)
        elif architecture == 'latent_cross_attention':
            self.base = LatentCrossAttentionBatLiNetRULPredictor(**options)
        else:
            raise ValueError(f'Unknown baseline: {architecture}')

    def forward(self, target, references, reference_labels):
        x, r = target['raw'], references['raw']
        b, s, c, h, w = r.shape
        # Use the original input cleaning and original prediction components.
        # Copies isolate their in-place preprocessing from the cached inputs.
        with torch.no_grad():
            if self.architecture == 'batlinet':
                dataset = self.base.build_cycle_diff_dataset(SimpleNamespace(
                    feature=x.clone(), label=torch.zeros(b, device=x.device)))
                own = dataset.feature
                paired = (dataset.raw_feature[:, None] - r).contiguous()
                paired = self.base._clean_feature(paired)
            else:
                own = self.base._prepare_feature(x)
                paired = self.base._prepare_feature(r.reshape(b * s, c, h, w))
                paired = paired.reshape(b, s, c, h, w).contiguous()
        ori, support, aggregate, _, _ = self.base.compute_prediction_components(
            own, paired, reference_labels)
        return dict(y_ori=ori, y_sup=support, y_sup_agg=aggregate,
                    prediction=(1 - self.alpha) * ori + self.alpha * aggregate)
