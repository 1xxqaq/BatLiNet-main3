import torch
import torch.nn as nn

from src.builders import MODELS

from .latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)


def _group_count(channels: int, max_groups: int = 8) -> int:
    """Choose a valid GroupNorm group count without batch statistics."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class CapacityScaleBranch(nn.Module):
    """Extract dense local patterns along the normalized-capacity axis."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                'capacity kernel sizes must be positive odd integers.')

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            padding_mode='replicate',
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.depthwise(x)))


class AxisAwareMultiScaleTokenEncoder(nn.Module):
    """Build cycle-capacity tokens with separated axis-aware operations."""

    def __init__(self,
                 in_channels: int,
                 token_channels: int,
                 input_height: int,
                 input_width: int,
                 branch_channels: int = 16,
                 capacity_kernel_sizes=(9, 33, 97),
                 cycle_kernel_size: int = 3,
                 cycle_pool_size: int = 4,
                 capacity_pool_size: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        capacity_kernel_sizes = tuple(capacity_kernel_sizes)
        if not capacity_kernel_sizes:
            raise ValueError('At least one capacity scale is required.')
        if cycle_kernel_size <= 0 or cycle_kernel_size % 2 == 0:
            raise ValueError(
                'cycle_kernel_size must be a positive odd integer.')
        if cycle_pool_size <= 0 or capacity_pool_size <= 0:
            raise ValueError('Pooling sizes must be positive integers.')

        self.input_height = input_height
        self.input_width = input_width

        # Mix the six physical feature channels at each cycle-capacity
        # coordinate before applying the same latent basis at every scale.
        self.channel_stem = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1,
                      bias=False),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.GELU(),
        )
        self.capacity_branches = nn.ModuleList([
            CapacityScaleBranch(branch_channels, kernel_size)
            for kernel_size in capacity_kernel_sizes
        ])

        multiscale_channels = branch_channels * len(capacity_kernel_sizes)
        self.cycle_depthwise = nn.Conv2d(
            multiscale_channels,
            multiscale_channels,
            kernel_size=(cycle_kernel_size, 1),
            padding=(cycle_kernel_size // 2, 0),
            padding_mode='replicate',
            groups=multiscale_channels,
            bias=False,
        )
        self.cycle_norm = nn.GroupNorm(
            _group_count(multiscale_channels), multiscale_channels)
        self.cycle_act = nn.GELU()

        self.channel_fusion = nn.Sequential(
            nn.Conv2d(multiscale_channels, token_channels, kernel_size=1,
                      bias=False),
            nn.GroupNorm(_group_count(token_channels), token_channels),
            nn.GELU(),
        )
        self.pool = nn.AvgPool2d(
            kernel_size=(cycle_pool_size, capacity_pool_size),
            stride=(cycle_pool_size, capacity_pool_size),
        )
        self.dropout = nn.Dropout2d(dropout)

        out_h = input_height // cycle_pool_size
        out_w = input_width // capacity_pool_size
        if out_h <= 0 or out_w <= 0:
            raise ValueError('Token encoder output shape is empty.')
        self.output_height = out_h
        self.output_width = out_w
        self.num_tokens = out_h * out_w
        self.token_channels = token_channels

        # Keep cycle-stage and capacity-region positions explicit instead of
        # assigning one opaque position vector after flattening the 2D grid.
        self.cycle_position = nn.Parameter(
            torch.zeros(1, out_h, 1, token_channels))
        self.capacity_position = nn.Parameter(
            torch.zeros(1, 1, out_w, token_channels))
        nn.init.trunc_normal_(self.cycle_position, std=0.02)
        nn.init.trunc_normal_(self.capacity_position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError('Expected a [B, C, H, W] battery tensor.')
        if x.size(-2) != self.input_height \
                or x.size(-1) != self.input_width:
            raise ValueError(
                f'Unexpected input shape {tuple(x.shape[-2:])}; expected '
                f'({self.input_height}, {self.input_width}).')

        stem = self.channel_stem(x)
        x = torch.cat(
            [branch(stem) for branch in self.capacity_branches], dim=1)

        cycle_context = self.cycle_act(
            self.cycle_norm(self.cycle_depthwise(x)))
        x = x + cycle_context
        x = self.dropout(self.pool(self.channel_fusion(x)))

        x = x.permute(0, 2, 3, 1).contiguous()
        if x.size(1) != self.output_height \
                or x.size(2) != self.output_width:
            raise ValueError(
                f'Unexpected token grid {tuple(x.shape[1:3])}; expected '
                f'({self.output_height}, {self.output_width}).')
        x = x + self.cycle_position + self.capacity_position
        return x.view(x.size(0), self.num_tokens, self.token_channels)


@MODELS.register()
class AxisAwareMultiScaleLatentCrossAttentionBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Latent cross-attention with an axis-aware multi-scale tokenizer."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 attention_channels: int = 64,
                 encoder_dropout: float = 0.1,
                 branch_channels: int = 16,
                 capacity_kernel_sizes=(9, 33, 97),
                 cycle_kernel_size: int = 3,
                 cycle_pool_size: int = 4,
                 capacity_pool_size: int = 32,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_height=input_height,
            input_width=input_width,
            attention_channels=attention_channels,
            encoder_dropout=encoder_dropout,
            **kwargs,
        )
        self.cell_encoder = AxisAwareMultiScaleTokenEncoder(
            in_channels=in_channels,
            token_channels=attention_channels,
            input_height=input_height,
            input_width=input_width,
            branch_channels=branch_channels,
            capacity_kernel_sizes=capacity_kernel_sizes,
            cycle_kernel_size=cycle_kernel_size,
            cycle_pool_size=cycle_pool_size,
            capacity_pool_size=capacity_pool_size,
            dropout=encoder_dropout,
        )
