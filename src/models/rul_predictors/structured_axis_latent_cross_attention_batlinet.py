"""Cycle and capacity-axis representations for reference-based battery life."""

import torch
import torch.nn as nn

from src.builders import MODELS

from .latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)


def masked_mean_max(tokens: torch.Tensor,
                    valid: torch.Tensor = None) -> torch.Tensor:
    if valid is None:
        return torch.cat((tokens.mean(1), tokens.amax(1)), dim=-1)
    weights = valid.unsqueeze(-1).to(tokens.dtype)
    mean = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1)
    maximum = tokens.masked_fill(
        ~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min).amax(1)
    return torch.cat((mean, maximum), dim=-1)


class StructuredAxisEncoder(nn.Module):
    """Build one token per cycle and one token per capacity interval."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 patch_width: int = 40,
                 attention_heads: int = 4,
                 cycle_encoder_layers: int = 1,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if channels % attention_heads:
            raise ValueError('channels must be divisible by attention_heads.')
        if input_height < 1 or input_width < patch_width or patch_width < 1:
            raise ValueError('Invalid cycle count, width, or patch width.')
        if input_width % patch_width:
            raise ValueError('input_width must be divisible by patch_width.')

        self.input_height = input_height
        self.input_width = input_width
        self.patch_count = input_width // patch_width
        self.channels = channels
        # Each cycle is embedded independently. No sample-wise normalization
        # removes its voltage/current magnitude before patch extraction.
        self.patch_embedding = nn.Conv1d(
            in_channels, channels,
            kernel_size=patch_width,
            stride=patch_width,
            bias=False,
        )
        self.patch_activation = nn.GELU()
        self.cycle_position = nn.Parameter(
            torch.zeros(1, input_height, 1, channels))
        self.capacity_position = nn.Parameter(
            torch.zeros(1, 1, self.patch_count, channels))
        nn.init.trunc_normal_(self.cycle_position, std=0.02)
        nn.init.trunc_normal_(self.capacity_position, std=0.02)

        cycle_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=attention_heads,
            dim_feedforward=channels * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.cycle_encoder = nn.TransformerEncoder(
            cycle_layer,
            num_layers=cycle_encoder_layers,
            enable_nested_tensor=False,
        )
        # The convolution explicitly sees neighbouring cycles at each fixed
        # capacity interval. Attention pooling then retains one token there.
        self.capacity_temporal_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False)
        self.capacity_temporal_score = nn.Linear(channels, 1)

    def forward(self, feature: torch.Tensor):
        if feature.ndim != 4:
            raise ValueError('Expected [batch, channels, cycles, width].')
        batch, in_channels, cycles, width = feature.shape
        if (cycles, width) != (self.input_height, self.input_width):
            raise ValueError('Unexpected cycle count or capacity width.')
        valid_cycle = feature.abs().amax(dim=(1, 3)) > 0
        if (~valid_cycle).all(dim=1).any():
            raise ValueError('Each battery needs at least one valid cycle.')

        curves = feature.permute(0, 2, 1, 3).reshape(
            batch * cycles, in_channels, width)
        # Signed log compression is monotone in magnitude and limits outliers.
        curves = curves.sign() * torch.log1p(curves.abs())
        patches = self.patch_activation(self.patch_embedding(curves))
        patches = patches.transpose(1, 2).reshape(
            batch, cycles, self.patch_count, self.channels)
        patches = patches * valid_cycle[:, :, None, None]

        cycle_tokens = patches.mean(dim=2) + self.cycle_position.squeeze(2)
        cycle_tokens = cycle_tokens * valid_cycle.unsqueeze(-1)
        cycle_tokens = self.cycle_encoder(
            cycle_tokens, src_key_padding_mask=~valid_cycle)
        cycle_tokens = cycle_tokens * valid_cycle.unsqueeze(-1)

        ordered_patches = patches + self.cycle_position + self.capacity_position
        ordered_patches = ordered_patches * valid_cycle[:, :, None, None]
        per_interval = ordered_patches.permute(0, 2, 3, 1).reshape(
            batch * self.patch_count, self.channels, cycles)
        temporal_change = self.capacity_temporal_conv(per_interval)
        temporal_change = self.patch_activation(temporal_change)
        temporal_change = temporal_change.transpose(1, 2).reshape(
            batch, self.patch_count, cycles, self.channels)
        per_interval = ordered_patches.permute(0, 2, 1, 3) + temporal_change
        scores = self.capacity_temporal_score(per_interval).squeeze(-1)
        scores = scores.masked_fill(
            ~valid_cycle[:, None, :], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        capacity_tokens = (per_interval * weights.unsqueeze(-1)).sum(dim=2)
        return cycle_tokens, capacity_tokens, valid_cycle


class MaskedRelationBlock(nn.Module):
    """Target queries attend to valid reference tokens of the same view."""

    def __init__(self, channels: int, attention_heads: int,
                 mlp_ratio: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.reference_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, attention_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * mlp_ratio, channels),
            nn.Dropout(dropout),
        )

    def forward(self, target: torch.Tensor, reference: torch.Tensor,
                target_mask: torch.Tensor = None,
                reference_mask: torch.Tensor = None) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(target),
            self.reference_norm(reference),
            self.reference_norm(reference),
            key_padding_mask=(~reference_mask if reference_mask is not None
                              else None),
            need_weights=False,
        )
        relation = target + self.attention_dropout(attended)
        relation = relation + self.ffn(self.ffn_norm(relation))
        if target_mask is not None:
            relation = relation * target_mask.unsqueeze(-1)
        return relation


class TargetHead(nn.Module):
    def __init__(self, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(channels * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cycle: torch.Tensor, capacity: torch.Tensor,
                cycle_mask: torch.Tensor) -> torch.Tensor:
        summary = torch.cat((
            masked_mean_max(cycle, cycle_mask),
            masked_mean_max(capacity),
        ), dim=-1)
        return self.regressor(summary).squeeze(-1)


class ReferenceDeltaHead(nn.Module):
    def __init__(self, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(channels * 12, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, target_cycle: torch.Tensor,
                reference_cycle: torch.Tensor,
                relation_cycle: torch.Tensor,
                target_capacity: torch.Tensor,
                reference_capacity: torch.Tensor,
                relation_capacity: torch.Tensor,
                target_mask: torch.Tensor,
                reference_mask: torch.Tensor) -> torch.Tensor:
        summary = torch.cat((
            masked_mean_max(target_cycle, target_mask),
            masked_mean_max(reference_cycle, reference_mask),
            masked_mean_max(relation_cycle, target_mask),
            masked_mean_max(target_capacity),
            masked_mean_max(reference_capacity),
            masked_mean_max(relation_capacity),
        ), dim=-1)
        return self.regressor(summary).squeeze(-1)


@MODELS.register()
class StructuredAxisLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Reference-based life predictor with explicit cycle/capacity views."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 attention_channels: int = 64,
                 attention_heads: int = 4,
                 attention_layers: int = 1,
                 attention_dropout: float = 0.1,
                 attention_mlp_ratio: int = 2,
                 head_hidden_channels: int = None,
                 encoder_dropout: float = 0.1,
                 patch_width: int = 40,
                 cycle_encoder_layers: int = 1,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_height=input_height,
            input_width=input_width,
            attention_channels=attention_channels,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            attention_dropout=attention_dropout,
            attention_mlp_ratio=attention_mlp_ratio,
            head_hidden_channels=head_hidden_channels,
            encoder_dropout=encoder_dropout,
            **kwargs,
        )
        self.cell_encoder = StructuredAxisEncoder(
            in_channels=in_channels,
            channels=attention_channels,
            input_height=input_height,
            input_width=input_width,
            patch_width=patch_width,
            attention_heads=attention_heads,
            cycle_encoder_layers=cycle_encoder_layers,
            mlp_ratio=attention_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.cross_attention = nn.ModuleList()
        self.cycle_relation = nn.ModuleList([
            MaskedRelationBlock(attention_channels, attention_heads,
                                attention_mlp_ratio, attention_dropout)
            for _ in range(attention_layers)
        ])
        self.capacity_relation = nn.ModuleList([
            MaskedRelationBlock(attention_channels, attention_heads,
                                attention_mlp_ratio, attention_dropout)
            for _ in range(attention_layers)
        ])
        hidden = head_hidden_channels or attention_channels
        self.ori_head = TargetHead(
            attention_channels, hidden, attention_dropout)
        self.support_head = ReferenceDeltaHead(
            attention_channels, hidden, attention_dropout)

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        batch, support_count, in_channels, cycles, width = support_feature.shape
        target_cycle, target_capacity, target_mask = self.cell_encoder(feature)
        reference_cycle, reference_capacity, reference_mask = self.cell_encoder(
            support_feature.reshape(
                batch * support_count, in_channels, cycles, width))
        patches = target_capacity.size(1)
        target_cycle_pair = target_cycle[:, None].expand(
            -1, support_count, -1, -1).reshape(
                batch * support_count, cycles, self.channels)
        target_capacity_pair = target_capacity[:, None].expand(
            -1, support_count, -1, -1).reshape(
                batch * support_count, patches, self.channels)
        target_mask_pair = target_mask[:, None].expand(
            -1, support_count, -1).reshape(batch * support_count, cycles)

        cycle_relation = target_cycle_pair
        for block in self.cycle_relation:
            cycle_relation = block(
                cycle_relation, reference_cycle,
                target_mask_pair, reference_mask)
        capacity_relation = target_capacity_pair
        for block in self.capacity_relation:
            capacity_relation = block(
                capacity_relation, reference_capacity)

        y_ori = self.ori_head(target_cycle, target_capacity, target_mask)
        delta = self.support_head(
            target_cycle_pair, reference_cycle, cycle_relation,
            target_capacity_pair, reference_capacity, capacity_relation,
            target_mask_pair, reference_mask)
        y_sup = delta.view(batch, support_count) + support_label.view(
            batch, support_count)
        if self.training:
            y_sup_agg = y_sup.mean(dim=1)
        else:
            y_sup_agg = y_sup.median(dim=1)[0]

        if return_features:
            target_tokens = torch.cat((target_cycle, target_capacity), dim=1)
            reference_tokens = torch.cat(
                (reference_cycle, reference_capacity), dim=1)
            return (
                y_ori, y_sup, y_sup_agg, None, None,
                target_tokens,
                reference_tokens.reshape(
                    batch, support_count, cycles + patches, self.channels),
            )
        return y_ori, y_sup, y_sup_agg, None, None
