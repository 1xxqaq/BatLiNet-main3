"""Single-layer target-reference matching followed by shared-space subtraction."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.builders import MODELS
from .latent_cross_attention_batlinet import (
    CrossAttentionBlock, LatentCrossAttentionBatLiNetRULPredictor,
)


class SoftAlignedDifferenceBlock(CrossAttentionBlock):
    """Keep all baseline parameters; replace target residual fusion by a difference.

    Queries use q_norm. Both target and reference values use kv_norm and the
    SAME V projection. Each head subtracts its attended reference values from
    its target values, before the common output projection. No raw target skip,
    extra context concatenation, extra loss, or additional attention layer.
    Attention dropout is retained during training, disabled during evaluation.
    """

    def projected_difference(self, target, reference):
        channels, heads = self.attn.embed_dim, self.attn.num_heads
        head_dim = channels // heads
        weights = self.attn.in_proj_weight.chunk(3, dim=0)
        biases = self.attn.in_proj_bias.chunk(3, dim=0)
        q = F.linear(self.q_norm(target), weights[0], biases[0])
        target_values = F.linear(self.kv_norm(target), weights[2], biases[2])
        reference_norm = self.kv_norm(reference)
        k = F.linear(reference_norm, weights[1], biases[1])
        v = F.linear(reference_norm, weights[2], biases[2])

        def split(x):
            return x.reshape(x.shape[0], x.shape[1], heads, head_dim).transpose(1, 2)

        matched = F.scaled_dot_product_attention(
            split(q), split(k), split(v),
            dropout_p=self.attn.dropout if self.training else 0.)
        difference = split(target_values) - matched
        return difference.transpose(1, 2).contiguous().reshape_as(target_values)

    def forward(self, query_tokens, reference_tokens):
        difference = self.projected_difference(query_tokens, reference_tokens)
        x = self.attn_dropout(self.attn.out_proj(difference))
        return x + self.ffn(self.ffn_norm(x))


@MODELS.register()
class SoftAlignedDifferenceBatLiNetRULPredictor(LatentCrossAttentionBatLiNetRULPredictor):
    def __init__(self, *args, **kwargs):
        if kwargs.get('attention_layers', 1) != 1:
            raise ValueError('Soft aligned difference requires exactly one attention layer.')
        super().__init__(*args, **kwargs)
        # Copy initialized modules, avoiding additional RNG draws or unused parameters.
        original = self.cross_attention[0]
        with torch.random.fork_rng(devices=[]):
            block = SoftAlignedDifferenceBlock(
                self.channels, original.attn.num_heads,
                mlp_ratio=original.ffn[0].out_features // self.channels,
                dropout=original.attn.dropout)
        block.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
        self.cross_attention = nn.ModuleList([block])
