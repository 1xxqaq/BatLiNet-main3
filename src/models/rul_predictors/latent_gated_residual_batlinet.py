"""Target-anchored gated residual cross-attention for BatLiNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.builders import MODELS

from .batlinet import mse
from .latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)


class ReferenceInnovationBlock(nn.Module):
    """Extract support-conditioned innovation without a target bypass."""

    def __init__(self,
                 channels: int,
                 num_heads: int,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.support_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        hidden_channels = channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self,
                target_tokens: torch.Tensor,
                support_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(target_tokens)
        support = self.support_norm(support_tokens)
        innovation, _ = self.attention(
            query,
            support,
            support,
            need_weights=False,
        )
        innovation = self.dropout(innovation)
        innovation = innovation + self.ffn(self.ffn_norm(innovation))
        return innovation


class RobustCorrectionAggregator(nn.Module):
    """Differentiable robust center for gated support corrections."""

    def __init__(self,
                 temperature: float = 0.5,
                 iterations: int = 3,
                 eps: float = 1e-6):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive.')
        if iterations < 1:
            raise ValueError('iterations must be at least one.')
        self.temperature = temperature
        self.iterations = iterations
        self.eps = eps

    def forward(self,
                corrections: torch.Tensor) -> tuple[torch.Tensor,
                                                     torch.Tensor]:
        center = corrections.mean(dim=1, keepdim=True)
        weights = torch.full_like(
            corrections,
            1.0 / corrections.size(1),
        )
        for _ in range(self.iterations):
            normalized_residual = (
                (corrections - center) / self.temperature
            )
            weights = torch.rsqrt(1.0 + normalized_residual.square())
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
                self.eps)
            center = (weights * corrections).sum(dim=1, keepdim=True)
        return center.squeeze(1), weights


class GatedResidualRelationHead(nn.Module):
    """Predict a support candidate and how strongly it should correct y_ori."""

    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float = 0.1,
                 gate_initial_bias: float = -2.0):
        super().__init__()
        pooled_channels = channels * 2
        context_channels = pooled_channels * 3 + 3
        self.context = nn.Sequential(
            nn.Linear(context_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(hidden_channels, 1)
        self.gate_head = nn.Linear(hidden_channels, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, gate_initial_bias)

    @staticmethod
    def pool(tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            tokens.mean(dim=-2),
            tokens.max(dim=-2).values,
        ], dim=-1)

    def forward(self,
                target_tokens: torch.Tensor,
                support_tokens: torch.Tensor,
                innovation_tokens: torch.Tensor,
                support_label: torch.Tensor,
                y_ori: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_pool = self.pool(target_tokens)
        support_pool = self.pool(support_tokens)
        innovation_pool = self.pool(innovation_tokens)
        support_label = support_label.unsqueeze(-1)
        y_ori = y_ori.detach().unsqueeze(-1)
        scalar_context = torch.cat([
            support_label,
            y_ori,
            support_label - y_ori,
        ], dim=-1)
        context = torch.cat([
            target_pool,
            support_pool,
            innovation_pool,
            scalar_context,
        ], dim=-1)
        hidden = self.context(context)
        reference_delta = self.delta_head(hidden).squeeze(-1)
        gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        return reference_delta, gate


@MODELS.register()
class LatentGatedResidualBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Use each support as a gated residual correction to the target branch."""

    def __init__(self,
                 *args,
                 final_loss_weight: float = 1.0,
                 ori_loss_weight: float = 0.3,
                 candidate_loss_weight: float = 0.2,
                 gate_loss_weight: float = 0.05,
                 candidate_smooth_l1_beta: float = 0.5,
                 robust_temperature: float = 0.5,
                 robust_iterations: int = 3,
                 gate_initial_bias: float = -2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        attention_channels = self.channels
        attention_heads = kwargs.get('attention_heads', 4)
        attention_mlp_ratio = kwargs.get('attention_mlp_ratio', 2)
        attention_dropout = kwargs.get('attention_dropout', 0.1)
        head_hidden_channels = (
            kwargs.get('head_hidden_channels') or attention_channels
        )

        self.innovation_block = ReferenceInnovationBlock(
            attention_channels,
            attention_heads,
            mlp_ratio=attention_mlp_ratio,
            dropout=attention_dropout,
        )
        self.relation_head = GatedResidualRelationHead(
            attention_channels,
            head_hidden_channels,
            dropout=attention_dropout,
            gate_initial_bias=gate_initial_bias,
        )
        self.robust_aggregator = RobustCorrectionAggregator(
            temperature=robust_temperature,
            iterations=robust_iterations,
        )
        # The inherited cross-attention and support head are intentionally
        # replaced so the support path cannot bypass reference innovation.
        del self.cross_attention
        del self.support_head

        self.final_loss_weight = final_loss_weight
        self.ori_loss_weight = ori_loss_weight
        self.candidate_loss_weight = candidate_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.candidate_smooth_l1_beta = candidate_smooth_l1_beta

    def combine_predictions(self,
                            y_ori: torch.Tensor,
                            y_sup_agg: torch.Tensor) -> torch.Tensor:
        return y_ori + self.alpha * (y_sup_agg - y_ori)

    def _compute_all(self,
                     feature: torch.Tensor,
                     support_feature: torch.Tensor,
                     support_label: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, support_size, channels, height, width = \
            support_feature.shape
        target_tokens = self.cell_encoder(feature)
        support_tokens = self.cell_encoder(
            support_feature.reshape(-1, channels, height, width))
        token_count = target_tokens.size(1)
        target_pair_tokens = (
            target_tokens.unsqueeze(1)
            .expand(-1, support_size, -1, -1)
            .reshape(
                batch_size * support_size,
                token_count,
                self.channels,
            )
        )

        innovation_tokens = self.innovation_block(
            target_pair_tokens,
            support_tokens,
        )
        y_ori = self.ori_head(target_tokens).reshape(-1)
        y_ori_pair = (
            y_ori.unsqueeze(1)
            .expand(-1, support_size)
            .reshape(-1)
        )
        flat_support_label = support_label.reshape(-1)
        reference_delta, gate = self.relation_head(
            target_pair_tokens,
            support_tokens,
            innovation_tokens,
            flat_support_label,
            y_ori_pair,
        )
        support_candidate = (
            flat_support_label + reference_delta
        ).reshape(batch_size, support_size)
        gate = gate.reshape(batch_size, support_size)
        corrections = gate * (support_candidate - y_ori.unsqueeze(1))
        correction_agg, robust_weight = self.robust_aggregator(corrections)
        y_sup = y_ori.unsqueeze(1) + corrections
        y_sup_agg = y_ori + correction_agg
        prediction = self.combine_predictions(y_ori, y_sup_agg)

        return {
            'prediction': prediction,
            'y_ori': y_ori,
            'y_sup': y_sup,
            'y_sup_agg': y_sup_agg,
            'support_candidate': support_candidate,
            'support_gate': gate,
            'support_weight': robust_weight,
            'target_tokens': target_tokens,
            'support_tokens': support_tokens.reshape(
                batch_size,
                support_size,
                token_count,
                self.channels,
            ),
            'innovation_tokens': innovation_tokens.reshape(
                batch_size,
                support_size,
                token_count,
                self.channels,
            ),
        }

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                support_feature: torch.Tensor,
                support_label: torch.Tensor,
                return_loss: bool = False):
        outputs = self._compute_all(
            feature,
            support_feature,
            support_label,
        )
        if self.return_pointwise_predictions:
            return outputs['y_ori'], outputs['y_sup']
        if not return_loss:
            return outputs['prediction']

        target_pair = label.unsqueeze(1).expand_as(
            outputs['support_candidate'])
        candidate_error = (
            outputs['support_candidate'] - target_pair
        ).abs()
        ori_error = (outputs['y_ori'] - label).abs().unsqueeze(1)
        gate_target = (candidate_error < ori_error).to(
            outputs['support_gate'].dtype).detach()

        final_loss = mse(outputs['prediction'], label)
        ori_loss = mse(outputs['y_ori'], label)
        candidate_loss = F.smooth_l1_loss(
            outputs['support_candidate'],
            target_pair,
            beta=self.candidate_smooth_l1_beta,
        )
        gate_loss = F.binary_cross_entropy(
            outputs['support_gate'],
            gate_target,
        )
        return sum([
            self.final_loss_weight * final_loss,
            self.ori_loss_weight * ori_loss,
            self.candidate_loss_weight * candidate_loss,
            self.gate_loss_weight * gate_loss,
        ])

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        outputs = self._compute_all(
            feature,
            support_feature,
            support_label,
        )
        components = (
            outputs['y_ori'],
            outputs['y_sup'],
            outputs['y_sup_agg'],
            outputs['support_weight'],
            outputs['support_gate'],
        )
        if return_features:
            return components + (
                outputs['target_tokens'],
                outputs['support_tokens'],
            )
        return components
