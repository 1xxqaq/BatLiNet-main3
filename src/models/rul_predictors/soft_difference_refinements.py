"""Independent, parameter-neutral controls for soft aligned differences."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from src.builders import MODELS
from .soft_aligned_difference_batlinet import (
    SoftAlignedDifferenceBlock, SoftAlignedDifferenceBatLiNetRULPredictor,
)
from .batlinet import mse


class ContentDifferenceBlock(SoftAlignedDifferenceBlock):
    """Position-bearing queries/keys, shared normalized content-only values."""

    def projected_difference(self, target, reference, target_content=None,
                             reference_content=None):
        if target_content is None or reference_content is None:
            raise ValueError('Content difference requires both content tensors.')
        heads = self.attn.num_heads
        width = self.attn.embed_dim // heads
        wq, wk, wv = self.attn.in_proj_weight.chunk(3)
        bq, bk, bv = self.attn.in_proj_bias.chunk(3)
        split = lambda x: x.reshape(x.shape[0], x.shape[1], heads, width).transpose(1, 2)
        q = split(F.linear(self.q_norm(target), wq, bq))
        k = split(F.linear(self.kv_norm(reference), wk, bk))
        vt = F.linear(self.kv_norm(target_content), wv, bv)
        vr = split(F.linear(self.kv_norm(reference_content), wv, bv))
        matched = F.scaled_dot_product_attention(
            q, k, vr, dropout_p=self.attn.dropout if self.training else 0.)
        return (split(vt) - matched).transpose(1, 2).contiguous().reshape_as(vt)

    def forward(self, target, reference, target_content, reference_content):
        delta = self.projected_difference(target, reference, target_content, reference_content)
        x = self.attn_dropout(self.attn.out_proj(delta))
        return x + self.ffn(self.ffn_norm(x))


@MODELS.register()
class SoftDifferenceRefinementBatLiNetRULPredictor(SoftAlignedDifferenceBatLiNetRULPredictor):
    def __init__(self, *args, content_only_values=False,
                 matching_dropout=None, pair_loss_weight=0., **kwargs):
        if not math.isfinite(pair_loss_weight) or pair_loss_weight < 0:
            raise ValueError('pair_loss_weight must be finite and nonnegative.')
        if matching_dropout is not None and not 0 <= matching_dropout < 1:
            raise ValueError('matching_dropout must be in [0, 1).')
        super().__init__(*args, **kwargs)
        self.content_only_values = content_only_values
        self.pair_loss_weight = float(pair_loss_weight)
        if content_only_values:
            old = self.cross_attention[0]
            with torch.random.fork_rng(devices=[]):
                block = ContentDifferenceBlock(
                    self.channels, old.attn.num_heads,
                    mlp_ratio=old.ffn[0].out_features // self.channels,
                    dropout=old.attn.dropout)
            block.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
            self.cross_attention = nn.ModuleList([block])
        if matching_dropout is not None:
            # Deliberately do not change head, output, encoder or FFN dropout.
            self.cross_attention[0].attn.dropout = float(matching_dropout)

    def _encode_parts(self, feature):
        encoder = self.cell_encoder
        content = encoder.net(feature).flatten(2).transpose(1, 2).contiguous()
        if content.shape[1] != encoder.position.shape[1]:
            raise ValueError('Unexpected token count.')
        return content, content + encoder.position

    def compute_prediction_components(self, feature, support_feature, support_label,
                                      return_features=False):
        if not self.content_only_values:
            return super().compute_prediction_components(
                feature, support_feature, support_label, return_features)
        b, s, c, h, w = support_feature.shape
        content, tokens = self._encode_parts(feature)
        ref_content, ref_tokens = self._encode_parts(support_feature.reshape(-1, c, h, w))
        t = tokens.shape[1]
        pair = lambda x: x[:, None].expand(-1, s, -1, -1).reshape(b*s, t, self.channels)
        relation = self.cross_attention[0](pair(tokens), ref_tokens, pair(content), ref_content)
        own = self.ori_head(tokens).reshape(-1)
        refs = self.support_head(relation).reshape(b, s) + support_label.reshape(b, s)
        agg = refs.mean(1) if self.training else refs.median(1).values
        result = (own, refs, agg, None, None)
        if return_features:
            return (*result, tokens, ref_tokens.reshape(b, s, t, self.channels))
        return result

    def forward(self, feature, label, support_feature, support_label, return_loss=False):
        own, refs, agg, _, _ = self.compute_prediction_components(feature, support_feature, support_label)
        if self.return_pointwise_predictions:
            return own, refs
        if return_loss:
            loss = (1-self.alpha)*mse(own, label) + self.alpha*mse(agg, label)
            if self.pair_loss_weight:
                # (pred_delta - (z_target-z_reference)) == (refs-z_target).
                # Average over both targets and references, in standardized log units.
                loss = loss + self.pair_loss_weight * (refs-label.reshape(-1, 1)).square().mean()
            return loss
        return self.combine_predictions(own, agg)
