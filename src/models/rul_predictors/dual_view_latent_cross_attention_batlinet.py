import torch
import torch.nn as nn

from src.builders import MODELS

from .latent_cross_attention_batlinet import (
    CrossAttentionBlock,
    LatentCrossAttentionBatLiNetRULPredictor,
)


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose a valid GroupNorm group count."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _masked_mean_max(tokens: torch.Tensor,
                     valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Pool tokens without allowing invalid cycles to affect statistics."""
    if valid_mask is None:
        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1)[0]
    else:
        weights = valid_mask.unsqueeze(-1).to(tokens.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        mean_pool = (tokens * weights).sum(dim=1) / denominator
        masked_tokens = tokens.masked_fill(
            ~valid_mask.unsqueeze(-1),
            torch.finfo(tokens.dtype).min,
        )
        max_pool = masked_tokens.max(dim=1)[0]
    return torch.cat([mean_pool, max_pool], dim=-1)


class DualViewCellEncoder(nn.Module):
    """Encode a battery into cycle, capacity-local and residual views."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 capacity_bins: int = 32,
                 stem_channels: int = 32,
                 attention_heads: int = 4,
                 cycle_encoder_layers: int = 1,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        if channels % attention_heads != 0:
            raise ValueError('channels must be divisible by attention_heads.')
        if input_height <= 0 or input_width <= 0 or capacity_bins <= 0:
            raise ValueError('Input sizes and capacity_bins must be positive.')

        self.input_height = input_height
        self.input_width = input_width
        self.capacity_bins = capacity_bins
        self.channels = channels

        # Every cycle is processed independently here. In particular, an
        # all-zero dropped cycle cannot leak into neighbouring cycles.
        self.local_stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                stem_channels,
                kernel_size=9,
                padding=4,
                padding_mode='replicate',
                bias=False,
            ),
            nn.GroupNorm(_group_count(stem_channels), stem_channels),
            nn.GELU(),
            nn.Conv1d(
                stem_channels,
                channels,
                kernel_size=7,
                padding=3,
                padding_mode='replicate',
                bias=False,
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(capacity_bins),
        )
        self.local_dropout = nn.Dropout(dropout)

        self.cycle_position = nn.Parameter(
            torch.zeros(1, input_height, 1, channels))
        self.capacity_position = nn.Parameter(
            torch.zeros(1, 1, capacity_bins, channels))
        nn.init.trunc_normal_(self.cycle_position, std=0.02)
        nn.init.trunc_normal_(self.capacity_position, std=0.02)

        self.cycle_pool_score = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 1),
        )
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
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )

        # One learned query per capacity region aggregates its evolution over
        # valid cycles. Parameters are shared by target and support batteries.
        self.capacity_temporal_query = nn.Parameter(
            torch.zeros(1, capacity_bins, channels))
        nn.init.trunc_normal_(self.capacity_temporal_query, std=0.02)
        self.capacity_temporal_norm = nn.LayerNorm(channels)
        self.capacity_temporal_attention = nn.MultiheadAttention(
            channels,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.capacity_output_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError('Expected a [B, C, H, W] battery tensor.')
        B, _, H, W = x.shape
        if H != self.input_height or W != self.input_width:
            raise ValueError(
                f'Unexpected input shape {(H, W)}; expected '
                f'({self.input_height}, {self.input_width}).')

        cycle_mask = x.abs().amax(dim=(1, 3)) > 0
        if (~cycle_mask).all(dim=1).any():
            raise ValueError('Each battery must contain at least one valid cycle.')

        local = self.local_stem(
            x.permute(0, 2, 1, 3).reshape(B * H, x.size(1), W))
        local = local.view(
            B, H, self.channels, self.capacity_bins)
        local = local.permute(0, 1, 3, 2).contiguous()
        local = self.local_dropout(local)
        local = local * cycle_mask[:, :, None, None].to(local.dtype)

        positioned = (
            local + self.cycle_position + self.capacity_position)
        positioned = positioned * cycle_mask[:, :, None, None].to(
            positioned.dtype)

        cycle_scores = self.cycle_pool_score(positioned).squeeze(-1)
        cycle_weights = torch.softmax(cycle_scores, dim=-1)
        cycle_tokens = (positioned * cycle_weights.unsqueeze(-1)).sum(dim=2)
        cycle_tokens = self.cycle_encoder(
            cycle_tokens,
            src_key_padding_mask=~cycle_mask,
        )
        cycle_tokens = cycle_tokens * cycle_mask.unsqueeze(-1).to(
            cycle_tokens.dtype)

        temporal_source = positioned.permute(0, 2, 1, 3).reshape(
            B * self.capacity_bins, H, self.channels)
        temporal_source = self.capacity_temporal_norm(temporal_source)
        capacity_query = self.capacity_temporal_query.expand(B, -1, -1)
        capacity_query = capacity_query.reshape(
            B * self.capacity_bins, 1, self.channels)
        temporal_mask = (~cycle_mask).unsqueeze(1).expand(
            -1, self.capacity_bins, -1)
        temporal_mask = temporal_mask.reshape(B * self.capacity_bins, H)
        capacity_tokens, _ = self.capacity_temporal_attention(
            capacity_query,
            temporal_source,
            temporal_source,
            key_padding_mask=temporal_mask,
            need_weights=False,
        )
        capacity_tokens = capacity_tokens.reshape(
            B, self.capacity_bins, self.channels)
        capacity_tokens = self.capacity_output_norm(
            capacity_tokens + self.capacity_temporal_query)

        residual_tokens = local.reshape(
            B, H * self.capacity_bins, self.channels)
        residual_mask = cycle_mask.unsqueeze(-1).expand(
            -1, -1, self.capacity_bins).reshape(B, -1)
        residual_summary = _masked_mean_max(
            residual_tokens,
            residual_mask,
        )
        return cycle_tokens, capacity_tokens, residual_summary, cycle_mask


class MaskedCrossAttentionBlock(CrossAttentionBlock):
    """Cross-attention block with explicit source and query masks."""

    def forward(self,
                query_tokens: torch.Tensor,
                reference_tokens: torch.Tensor,
                reference_mask: torch.Tensor = None,
                query_mask: torch.Tensor = None) -> torch.Tensor:
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(reference_tokens)
        key_padding_mask = None
        if reference_mask is not None:
            key_padding_mask = ~reference_mask
        attended, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = query_tokens + self.attn_dropout(attended)
        x = x + self.ffn(self.ffn_norm(x))
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(x.dtype)
        return x


class DualViewOriginalHead(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 6, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self,
                cycle_tokens: torch.Tensor,
                capacity_tokens: torch.Tensor,
                residual_summary: torch.Tensor,
                cycle_mask: torch.Tensor) -> torch.Tensor:
        features = torch.cat([
            _masked_mean_max(cycle_tokens, cycle_mask),
            _masked_mean_max(capacity_tokens),
            residual_summary,
        ], dim=-1)
        return self.net(features).squeeze(-1)


class DualViewSupportHead(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 10, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self,
                cycle_relation: torch.Tensor,
                capacity_relation: torch.Tensor,
                target_residual: torch.Tensor,
                support_residual: torch.Tensor,
                target_cycle_mask: torch.Tensor) -> torch.Tensor:
        features = torch.cat([
            _masked_mean_max(cycle_relation, target_cycle_mask),
            _masked_mean_max(capacity_relation),
            target_residual,
            support_residual,
            target_residual - support_residual,
        ], dim=-1)
        return self.net(features).squeeze(-1)


@MODELS.register()
class DualViewLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """BatLiNet with axis-aligned cycle and capacity cross-attention."""

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
                 capacity_bins: int = 32,
                 stem_channels: int = 32,
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
        self.capacity_bins = capacity_bins
        self.cell_encoder = DualViewCellEncoder(
            in_channels=in_channels,
            channels=attention_channels,
            input_height=input_height,
            input_width=input_width,
            capacity_bins=capacity_bins,
            stem_channels=stem_channels,
            attention_heads=attention_heads,
            cycle_encoder_layers=cycle_encoder_layers,
            mlp_ratio=attention_mlp_ratio,
            dropout=encoder_dropout,
        )
        self.cross_attention = nn.ModuleList()
        self.cycle_cross_attention = nn.ModuleList([
            MaskedCrossAttentionBlock(
                attention_channels,
                attention_heads,
                mlp_ratio=attention_mlp_ratio,
                dropout=attention_dropout,
            )
            for _ in range(attention_layers)
        ])
        self.capacity_cross_attention = nn.ModuleList([
            MaskedCrossAttentionBlock(
                attention_channels,
                attention_heads,
                mlp_ratio=attention_mlp_ratio,
                dropout=attention_dropout,
            )
            for _ in range(attention_layers)
        ])
        hidden_channels = head_hidden_channels or attention_channels
        self.ori_head = DualViewOriginalHead(
            attention_channels,
            hidden_channels,
            attention_dropout,
        )
        self.support_head = DualViewSupportHead(
            attention_channels,
            hidden_channels,
            attention_dropout,
        )

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        B, S, C, H, W = support_feature.shape
        target_cycle, target_capacity, target_residual, target_mask = \
            self.cell_encoder(feature)
        support_cycle, support_capacity, support_residual, support_mask = \
            self.cell_encoder(support_feature.reshape(B * S, C, H, W))

        target_cycle_pair = target_cycle.unsqueeze(1).expand(
            -1, S, -1, -1).reshape(B * S, H, self.channels)
        target_capacity_pair = target_capacity.unsqueeze(1).expand(
            -1, S, -1, -1).reshape(
                B * S, self.capacity_bins, self.channels)
        target_residual_pair = target_residual.unsqueeze(1).expand(
            -1, S, -1).reshape(B * S, self.channels * 2)
        target_mask_pair = target_mask.unsqueeze(1).expand(
            -1, S, -1).reshape(B * S, H)

        cycle_relation = target_cycle_pair
        for block in self.cycle_cross_attention:
            cycle_relation = block(
                cycle_relation,
                support_cycle,
                reference_mask=support_mask,
                query_mask=target_mask_pair,
            )

        capacity_relation = target_capacity_pair
        for block in self.capacity_cross_attention:
            capacity_relation = block(
                capacity_relation,
                support_capacity,
            )

        y_ori = self.ori_head(
            target_cycle,
            target_capacity,
            target_residual,
            target_mask,
        ).view(-1)
        y_sup = self.support_head(
            cycle_relation,
            capacity_relation,
            target_residual_pair,
            support_residual,
            target_mask_pair,
        ).view(B, S)
        y_sup = y_sup + support_label.view(B, S)

        if self.training:
            y_sup_agg = y_sup.mean(dim=1).view(-1)
        else:
            y_sup_agg = y_sup.median(dim=1)[0].view(-1)

        if return_features:
            target_tokens = torch.cat(
                [target_cycle, target_capacity], dim=1)
            support_tokens = torch.cat(
                [support_cycle, support_capacity], dim=1)
            return (
                y_ori,
                y_sup,
                y_sup_agg,
                None,
                None,
                target_tokens,
                support_tokens.view(
                    B, S, H + self.capacity_bins, self.channels),
            )
        return y_ori, y_sup, y_sup_agg, None, None
