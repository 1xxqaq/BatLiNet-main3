"""Context-enhanced 2D local tokens for reference-conditioned life prediction.

This model uses scripts/context_grid_lifetime.py, not the legacy pipeline.
All fitted normalization statistics are checkpoint buffers.
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(x, mask):
    weights = mask.unsqueeze(-1).to(x.dtype)
    mean = (x * weights).sum(1) / weights.sum(1).clamp_min(1)
    maximum = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).amax(1)
    maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), -1)


def early_difference(curves, mask):
    # Earliest three valid cycles per cell, with pointwise observed intersections.
    cycles = mask.any(-1)
    first = cycles & (cycles.long().cumsum(1) <= 3)
    available = mask & first.unsqueeze(-1)
    base = curves.masked_fill(~available.unsqueeze(1), float('nan')).nanmedian(2, keepdim=True).values
    valid = mask & torch.isfinite(base).all(1)
    delta = torch.where(valid.unsqueeze(1), curves - base.nan_to_num(), 0)
    return delta, valid


class TrainStatistics(nn.Module):
    def __init__(self):
        super().__init__()
        for name, n in [('raw', 3), ('delta', 3), ('descriptor', 8)]:
            self.register_buffer(name + '_mean', torch.zeros(n))
            self.register_buffer(name + '_scale', torch.ones(n))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit(self, data):
        # Accumulate in float64 in small chunks; do not load test/val statistics.
        totals = {k: [torch.zeros(n, dtype=torch.float64) for _ in range(3)]
                  for k, n in [('raw', 3), ('delta', 3), ('descriptor', 8)]}
        for start in range(0, len(data['curves']), 16):
            raw = data['curves'][start:start + 16].cpu()
            mask = data['mask'][start:start + 16].cpu()
            delta, dm = early_difference(raw, mask)
            desc = data['descriptors'][start:start + 16].cpu()
            dmask = data['descriptor_mask'][start:start + 16].cpu()
            for key, values, valid in [('raw', raw.movedim(1, -1), mask.unsqueeze(-1).expand_as(raw.movedim(1, -1))),
                                       ('delta', delta.movedim(1, -1), dm.unsqueeze(-1).expand_as(delta.movedim(1, -1))),
                                       ('descriptor', desc, dmask)]:
                if not torch.isfinite(values[valid]).all():
                    raise ValueError('Non-finite observed training values.')
                values = torch.where(valid, values, 0).double().reshape(-1, values.size(-1))
                valid = valid.reshape_as(values)
                totals[key][0] += values.sum(0)
                totals[key][1] += values.square().sum(0)
                totals[key][2] += valid.sum(0)
        for key, (total, squared, count) in totals.items():
            mean = total / count.clamp_min(1)
            scale = (squared / count.clamp_min(1) - mean.square()).clamp_min(0).sqrt()
            scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
            getattr(self, key + '_mean').copy_(mean)
            getattr(self, key + '_scale').copy_(scale)
        self.fitted.fill_(True)

    def forward(self, data):
        if not bool(self.fitted):
            raise RuntimeError('Fit statistics using training cells before prediction.')
        raw, mask = data['curves'], data['mask']
        # Ignore arbitrary placeholders BEFORE difference and normalization.
        raw = torch.where(mask.unsqueeze(1), raw, 0)
        delta, delta_mask = early_difference(raw, mask)
        raw = (raw - self.raw_mean[None, :, None, None]) / self.raw_scale[None, :, None, None]
        delta = (delta - self.delta_mean[None, :, None, None]) / self.delta_scale[None, :, None, None]
        desc = (data['descriptors'] - self.descriptor_mean) / self.descriptor_scale
        return (torch.where(mask.unsqueeze(1), raw, 0),
                torch.where(delta_mask.unsqueeze(1), delta, 0), delta_mask,
                torch.where(data['descriptor_mask'], desc, 0))


class PartialConv(nn.Module):
    """Keep missing output locations masked and normalize observed support."""
    def __init__(self, in_channels, channels, kernel, dilation=(1, 1)):
        super().__init__()
        pad = tuple((k // 2) * d for k, d in zip(kernel, dilation))
        self.conv = nn.Conv2d(in_channels, channels, kernel, padding=pad,
                              dilation=dilation, bias=False)
        self.register_buffer('ones', torch.ones(1, 1, *kernel))
        self.pad, self.dilation = pad, dilation

    def forward(self, x, mask):
        weights = mask.to(x.dtype)
        count = F.conv2d(weights, self.ones.to(x.dtype), padding=self.pad, dilation=self.dilation)
        # Edge padding isn't an observation; normalize only across actual grid.
        possible = F.conv2d(torch.ones_like(weights), self.ones.to(x.dtype),
                            padding=self.pad, dilation=self.dilation)
        return self.conv(x * weights) * possible / count.clamp_min(1) * weights


def pool_grid(x, mask, size):
    weights = mask.to(x.dtype)
    denominator = F.adaptive_avg_pool2d(weights, size)
    pooled = F.adaptive_avg_pool2d(x * weights, size) / denominator.clamp_min(1e-8)
    return pooled, denominator > 0


class ContextGridEncoder(nn.Module):
    def __init__(self, cycles, phase_points=256, channels=64, bins=32,
                 heads=4, dropout=.1, use_context=True, cycle_points=32):
        super().__init__()
        if cycles < 4 or bins % 2 or phase_points < bins // 2 or channels % heads:
            raise ValueError('Invalid grid or attention dimensions.')
        self.cycles, self.phase_points = cycles, phase_points
        self.channels, self.bins, self.cycle_points = channels, bins, cycle_points
        self.use_context = use_context
        self.statistics = TrainStatistics()
        self.raw_stem = nn.ModuleList([PartialConv(3, channels // 2, (3, 9)),
                                      PartialConv(channels // 2, channels // 2, (3, 5))])
        self.delta_stem = nn.ModuleList([PartialConv(3, channels // 2, (3, 9)),
                                        PartialConv(channels // 2, channels // 2, (3, 5))])
        self.phase_position = nn.Parameter(torch.randn(1, channels, 1, 2) * .02)
        self.capacity_position = nn.Parameter(torch.randn(1, channels, 1, bins) * .02)
        self.cycle_position = nn.Parameter(torch.randn(1, channels, cycles, 1) * .02)
        if use_context:
            # Full-cycle positional projection, separate from local convolution.
            self.cycle_embed = nn.Linear(2 * cycle_points * 8 + 16, channels)
            layer = nn.TransformerEncoderLayer(channels, heads, channels * 2,
                                               dropout, activation='gelu', batch_first=True, norm_first=True)
            self.cycle_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.temporal = nn.ModuleList([PartialConv(channels, channels, (3, 1)),
                                          PartialConv(channels, channels, (3, 1), (2, 1))])
            self.capacity_head = nn.Sequential(nn.Linear(channels * 4, channels), nn.GELU(),
                                               nn.Linear(channels, channels))
            self.cycle_project = nn.Linear(channels, channels)
            self.capacity_project = nn.Linear(channels, channels)
            self.gates = nn.Parameter(torch.full((2,), .1))
        self.fusion = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(channels * 2, channels))
        self.fusion_norm = nn.LayerNorm(channels)

    def forward(self, data):
        raw, delta, delta_mask, desc = self.statistics(data)
        batch, _, h, width = raw.shape
        if (h, width) != (self.cycles, 2 * self.phase_points):
            raise ValueError('Input window does not match the model configuration.')
        observed = data['mask'].unsqueeze(1)
        cycle_valid = observed[:, 0].any(-1)
        if not cycle_valid.any(1).all():
            raise ValueError('Every cell needs an observed early cycle.')
        grids, masks, full_cycles = [], [], []
        for phase in range(2):
            sl = slice(phase * self.phase_points, (phase + 1) * self.phase_points)
            m, dm = observed[..., sl], delta_mask.unsqueeze(1)[..., sl]
            r, d = raw[..., sl], delta[..., sl]
            for conv in self.raw_stem:
                r = F.gelu(conv(r, m))
            for conv in self.delta_stem:
                d = F.gelu(conv(d, dm))
            grid, gm = pool_grid(torch.cat((r, d), 1), m, (h, self.bins // 2))
            grids.append(grid + self.phase_position[..., phase:phase + 1])
            masks.append(gm)
            if self.use_context:
                rr, _ = pool_grid(raw[..., sl], m, (h, self.cycle_points))
                dd, _ = pool_grid(delta[..., sl], dm, (h, self.cycle_points))
                full_cycles.append(torch.cat((rr, dd,
                    F.adaptive_avg_pool2d(m.float(), (h, self.cycle_points)),
                    F.adaptive_avg_pool2d(dm.float(), (h, self.cycle_points))), 1))
        valid = torch.cat(masks, -1)
        local = (torch.cat(grids, -1) + self.capacity_position + self.cycle_position) * valid
        if self.use_context:
            whole = torch.cat(full_cycles, -1).permute(0, 2, 1, 3).flatten(2)
            whole = torch.cat((whole, desc, data['descriptor_mask'].to(desc.dtype)), -1)
            c = self.cycle_embed(whole) + self.cycle_position.squeeze(-1).transpose(1, 2)
            c = self.cycle_encoder(c, src_key_padding_mask=~cycle_valid)
            c = c * cycle_valid.unsqueeze(-1)
            q = local
            for conv in self.temporal:
                q = q + F.gelu(conv(q, valid))
            sequence = q.permute(0, 3, 2, 1).reshape(batch * self.bins, h, self.channels)
            vm = valid[:, 0].transpose(1, 2).reshape(batch * self.bins, h)
            first = vm.long().argmax(1)
            last = h - 1 - vm.flip(1).long().argmax(1)
            row = torch.arange(len(sequence), device=q.device)
            ends = torch.cat((sequence[row, first], sequence[row, last]), -1) * vm.any(1, keepdim=True)
            q = self.capacity_head(torch.cat((masked_pool(sequence, vm), ends), -1))
            q = q.reshape(batch, self.bins, self.channels)
            local = (local + self.gates[0] * self.cycle_project(c).transpose(1, 2).unsqueeze(-1)
                     + self.gates[1] * self.capacity_project(q).transpose(1, 2).unsqueeze(2)) * valid
        local = local.permute(0, 2, 3, 1)
        local = local + self.fusion(self.fusion_norm(local))
        local = local.permute(0, 3, 1, 2) * valid
        tokens, tm = pool_grid(local, valid, (h // 4, self.bins))
        return tokens.flatten(2).transpose(1, 2), tm.flatten(1)


class MaskedCrossAttention(nn.Module):
    def __init__(self, channels, heads, dropout):
        super().__init__()
        self.q_norm, self.kv_norm = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(channels * 2, channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, reference, target_mask, reference_mask):
        kv = self.kv_norm(reference)
        attended, _ = self.attention(self.q_norm(target), kv, kv,
                                    key_padding_mask=~reference_mask, need_weights=False)
        x = target + self.dropout(attended)
        return (x + self.ffn(self.norm(x))) * target_mask.unsqueeze(-1)


class ContextGridLifetime(nn.Module):
    def __init__(self, cycles=20, phase_points=256, channels=64, bins=32,
                 heads=4, dropout=.1, use_context=True, alpha=.5):
        super().__init__()
        self.alpha = alpha
        self.encoder = ContextGridEncoder(cycles, phase_points, channels, bins, heads, dropout, use_context)
        self.relation = MaskedCrossAttention(channels, heads, dropout)
        def head():
            return nn.Sequential(nn.Linear(channels * 2, channels), nn.LayerNorm(channels),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(channels, 1))
        self.ori_head, self.support_head = head(), head()

    def components_from_tokens(self, target, target_mask, references, reference_mask, labels):
        b, s, t, d = references.shape
        paired = target[:, None].expand(-1, s, -1, -1).reshape(b * s, t, d)
        pm = target_mask[:, None].expand(-1, s, -1).reshape(b * s, t)
        relation = self.relation(paired, references.reshape(b * s, t, d), pm,
                                 reference_mask.reshape(b * s, t))
        ori = self.ori_head(masked_pool(target, target_mask)).flatten()
        support = self.support_head(masked_pool(relation, pm)).reshape(b, s) + labels
        aggregate = support.mean(1) if self.training else support.median(1).values
        return dict(prediction=(1 - self.alpha) * ori + self.alpha * aggregate,
                    y_ori=ori, y_sup=support, y_sup_agg=aggregate)

    def forward(self, target, references, reference_labels):
        b, s = reference_labels.shape
        tokens, mask = self.encoder(target)
        flat = {k: v.flatten(0, 1) for k, v in references.items()}
        rt, rm = self.encoder(flat)
        return self.components_from_tokens(tokens, mask, rt.reshape(b, s, *rt.shape[1:]),
                                           rm.reshape(b, s, -1), reference_labels)
