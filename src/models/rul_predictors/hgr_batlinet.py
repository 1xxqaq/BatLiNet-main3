import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.builders import MODELS
from src.data.databundle import DataBundle

from .batlinet import mse
from .latent_cross_attention_batlinet import (
    CrossAttentionBlock,
    LatentCrossAttentionBatLiNetRULPredictor,
)


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class HierarchicalDegradationEncoder(nn.Module):
    """Model curve morphology first and cycle-wise degradation second."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_cycles: int,
                 num_heads: int,
                 cycle_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        hidden_channels = max(channels // 2, 16)
        self.capacity_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=(1, 15),
                padding=(0, 7),
            ),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=(1, 9),
                padding=(0, 4),
            ),
            nn.GroupNorm(_group_count(channels), channels),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, 5),
                padding=(0, 2),
                groups=channels,
            ),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.capacity_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.curve_fusion = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.degradation_update = nn.Sequential(
            nn.Linear(channels * 2, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.degradation_gate = nn.Linear(channels * 3, channels)
        nn.init.constant_(self.degradation_gate.bias, -1.0)

        self.cycle_position = nn.Parameter(
            torch.zeros(1, num_cycles, channels))
        nn.init.trunc_normal_(self.cycle_position, std=0.02)
        cycle_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.cycle_encoder = nn.TransformerEncoder(
            cycle_layer,
            num_layers=cycle_layers,
        )
        self.output_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curves = self.capacity_encoder(x)
        capacity_weight = torch.softmax(
            self.capacity_score(curves), dim=-1)
        attentive_curve = (curves * capacity_weight).sum(dim=-1)
        maximum_curve = curves.max(dim=-1)[0]
        cycle_tokens = torch.cat(
            [attentive_curve, maximum_curve], dim=1)
        cycle_tokens = cycle_tokens.transpose(1, 2).contiguous()
        cycle_tokens = self.curve_fusion(cycle_tokens)

        relative_to_first = cycle_tokens - cycle_tokens[:, :1]
        adjacent_change = torch.zeros_like(cycle_tokens)
        adjacent_change[:, 1:] = cycle_tokens[:, 1:] - cycle_tokens[:, :-1]
        degradation_input = torch.cat(
            [relative_to_first, adjacent_change], dim=-1)
        degradation_update = self.degradation_update(degradation_input)
        gate_input = torch.cat(
            [cycle_tokens, relative_to_first, adjacent_change], dim=-1)
        degradation_gate = torch.sigmoid(self.degradation_gate(gate_input))
        cycle_tokens = cycle_tokens + degradation_gate * degradation_update

        if cycle_tokens.size(1) != self.cycle_position.size(1):
            raise ValueError(
                f'Unexpected cycle count {cycle_tokens.size(1)}; '
                f'expected {self.cycle_position.size(1)}.')
        cycle_tokens = cycle_tokens + self.cycle_position
        cycle_tokens = self.cycle_encoder(cycle_tokens)
        return self.output_norm(cycle_tokens)


class AttentiveStatisticsPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels // 2),
            nn.Tanh(),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weight = torch.softmax(self.score(tokens), dim=-2)
        mean = (weight * tokens).sum(dim=-2)
        variance = (weight * (tokens - mean.unsqueeze(-2)).square()).sum(
            dim=-2)
        standard_deviation = torch.sqrt(variance.clamp_min(1e-6))
        maximum = tokens.max(dim=-2)[0]
        return torch.cat([mean, standard_deviation, maximum], dim=-1)


class ProbabilisticRegressionHead(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.pool = AttentiveStatisticsPool(channels)
        self.representation = nn.Sequential(
            nn.Linear(channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean = nn.Linear(hidden_channels, 1)
        self.log_variance = nn.Linear(hidden_channels, 1)

    def forward(self, tokens: torch.Tensor):
        representation = self.representation(self.pool(tokens))
        mean = self.mean(representation).squeeze(-1)
        log_variance = self.log_variance(representation).squeeze(-1)
        return mean, log_variance.clamp(-5.0, 5.0), representation


class GatedBidirectionalRelationBlock(nn.Module):
    """Exchange target/reference context without overwriting either stream."""

    def __init__(self,
                 channels: int,
                 num_heads: int,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.target_from_reference = CrossAttentionBlock(
            channels, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        self.reference_from_target = CrossAttentionBlock(
            channels, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
        relation_channels = channels * 4
        gate_channels = channels * 3
        self.target_update = nn.Sequential(
            nn.Linear(relation_channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.reference_update = nn.Sequential(
            nn.Linear(relation_channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )
        self.target_gate = nn.Linear(gate_channels, channels)
        self.reference_gate = nn.Linear(gate_channels, channels)
        self.target_norm = nn.LayerNorm(channels)
        self.reference_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.target_update[-1].weight)
        nn.init.zeros_(self.target_update[-1].bias)
        nn.init.zeros_(self.reference_update[-1].weight)
        nn.init.zeros_(self.reference_update[-1].bias)
        nn.init.constant_(self.target_gate.bias, -1.0)
        nn.init.constant_(self.reference_gate.bias, -1.0)

    def forward(self,
                target_tokens: torch.Tensor,
                reference_tokens: torch.Tensor):
        target_context = self.target_from_reference(
            target_tokens, reference_tokens)
        reference_context = self.reference_from_target(
            reference_tokens, target_tokens)
        aligned_difference = target_tokens - reference_tokens
        absolute_difference = aligned_difference.abs()

        relation = torch.cat([
            target_context - target_tokens,
            reference_context - reference_tokens,
            aligned_difference,
            absolute_difference,
        ], dim=-1)
        target_gate_input = torch.cat([
            target_tokens, reference_tokens, absolute_difference], dim=-1)
        reference_gate_input = torch.cat([
            reference_tokens, target_tokens, absolute_difference], dim=-1)
        target_tokens = self.target_norm(
            target_tokens
            + torch.sigmoid(self.target_gate(target_gate_input))
            * self.target_update(relation)
        )
        reference_tokens = self.reference_norm(
            reference_tokens
            + torch.sigmoid(self.reference_gate(reference_gate_input))
            * self.reference_update(relation)
        )
        return target_tokens, reference_tokens


class ReferenceCandidateHead(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.pair_fusion = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regressor = ProbabilisticRegressionHead(
            channels, hidden_channels, dropout=dropout)
        self.gain_head = nn.Sequential(
            nn.Linear(hidden_channels + 4, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self,
                target_tokens: torch.Tensor,
                reference_tokens: torch.Tensor,
                y_ori: torch.Tensor,
                support_label: torch.Tensor):
        difference = target_tokens - reference_tokens
        pair_tokens = self.pair_fusion(torch.cat([
            target_tokens,
            reference_tokens,
            difference,
            difference.abs(),
        ], dim=-1))
        delta, log_variance, representation = self.regressor(pair_tokens)
        y_support = support_label + delta
        correction = y_support - y_ori
        gain_input = torch.cat([
            representation,
            y_support.unsqueeze(-1),
            y_ori.unsqueeze(-1),
            support_label.unsqueeze(-1),
            correction.abs().unsqueeze(-1),
        ], dim=-1)
        gain_score = self.gain_head(gain_input).squeeze(-1)
        return y_support, delta, log_variance, gain_score, pair_tokens


class GainAwareRobustAggregator(nn.Module):
    def __init__(self,
                 target_representation_channels: int,
                 hidden_channels: int,
                 temperature: float = 1.0,
                 robust_scale: float = 2.5,
                 uncertainty_penalty: float = 0.25,
                 dropout: float = 0.1,
                 use_gain_aware_weights: bool = True,
                 use_adaptive_gate: bool = True,
                 fixed_gate: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.robust_scale = robust_scale
        self.uncertainty_penalty = uncertainty_penalty
        self.use_gain_aware_weights = use_gain_aware_weights
        self.use_adaptive_gate = use_adaptive_gate
        self.fixed_gate = fixed_gate
        gate_input_channels = target_representation_channels + 5
        self.gate = nn.Sequential(
            nn.Linear(gate_input_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.0)

    def forward(self,
                y_ori: torch.Tensor,
                y_ori_log_variance: torch.Tensor,
                target_representation: torch.Tensor,
                y_support: torch.Tensor,
                support_log_variance: torch.Tensor,
                gain_score: torch.Tensor):
        center = y_support.median(dim=1, keepdim=True)[0].detach()
        deviation = (y_support - center).abs()
        scale = deviation.median(dim=1, keepdim=True)[0].detach()
        scale = scale.clamp_min(1e-3)
        robust_weight = 1.0 / (
            1.0 + (deviation / (self.robust_scale * scale)).square())
        logits = robust_weight.clamp_min(1e-6).log()
        if self.use_gain_aware_weights:
            logits = (
                logits
                + gain_score
                - self.uncertainty_penalty * support_log_variance
            )
        weight = torch.softmax(logits / self.temperature, dim=1)

        correction = (
            weight * (y_support - y_ori.unsqueeze(1))).sum(dim=1)
        aggregated_support = y_ori + correction
        dispersion = torch.sqrt((
            weight
            * (y_support - aggregated_support.unsqueeze(1)).square()
        ).sum(dim=1).clamp_min(1e-6))
        weighted_log_variance = (
            weight * support_log_variance).sum(dim=1)
        if y_support.size(1) > 1:
            entropy = -(
                weight * weight.clamp_min(1e-8).log()).sum(dim=1)
            entropy = entropy / torch.log(
                weight.new_tensor(float(y_support.size(1))))
        else:
            entropy = torch.zeros_like(correction)

        if self.use_adaptive_gate:
            gate_input = torch.cat([
                target_representation,
                y_ori_log_variance.unsqueeze(-1),
                weighted_log_variance.unsqueeze(-1),
                dispersion.unsqueeze(-1),
                entropy.unsqueeze(-1),
                correction.abs().unsqueeze(-1),
            ], dim=-1)
            correction_gate = torch.sigmoid(
                self.gate(gate_input).squeeze(-1))
        else:
            correction_gate = torch.full_like(correction, self.fixed_gate)
        final_prediction = y_ori + correction_gate * correction
        return {
            'prediction': final_prediction,
            'y_sup_agg': aggregated_support,
            'support_weight': weight,
            'correction_gate': correction_gate,
            'support_dispersion': dispersion,
            'support_entropy': entropy,
        }


@MODELS.register()
class HGRBatLiNetRULPredictor(LatentCrossAttentionBatLiNetRULPredictor):
    """Hierarchical, gain-aware relational BatLiNet."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 hierarchy_channels: int = 64,
                 hierarchy_heads: int = 4,
                 hierarchy_cycle_layers: int = 2,
                 relation_layers: int = 1,
                 relation_mlp_ratio: int = 2,
                 head_hidden_channels: int = 96,
                 model_dropout: float = 0.1,
                 train_support_size_min: int = None,
                 aggregation_temperature: float = 1.0,
                 aggregation_robust_scale: float = 2.5,
                 aggregation_uncertainty_penalty: float = 0.25,
                 use_bidirectional_relation: bool = True,
                 use_gain_aware_weights: bool = True,
                 use_adaptive_gate: bool = True,
                 fixed_gate: float = 0.5,
                 final_loss_weight: float = 1.0,
                 ori_loss_weight: float = 0.3,
                 pair_delta_loss_weight: float = 0.3,
                 gain_loss_weight: float = 0.1,
                 ranking_loss_weight: float = 0.1,
                 uncertainty_loss_weight: float = 0.05,
                 ranking_margin: float = 0.05,
                 max_grad_norm: float = 5.0,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            input_height=input_height,
            input_width=input_width,
            attention_channels=hierarchy_channels,
            attention_heads=hierarchy_heads,
            attention_layers=1,
            attention_dropout=model_dropout,
            attention_mlp_ratio=relation_mlp_ratio,
            head_hidden_channels=head_hidden_channels,
            encoder_dropout=model_dropout,
            **kwargs,
        )
        self.channels = hierarchy_channels
        self.train_support_size_min = (
            train_support_size_min or self.train_support_size)
        if self.train_support_size_min > self.train_support_size:
            raise ValueError(
                'train_support_size_min cannot exceed train_support_size.')
        self.use_bidirectional_relation = use_bidirectional_relation
        self.final_loss_weight = final_loss_weight
        self.ori_loss_weight = ori_loss_weight
        self.pair_delta_loss_weight = pair_delta_loss_weight
        self.gain_loss_weight = gain_loss_weight
        self.ranking_loss_weight = ranking_loss_weight
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.ranking_margin = ranking_margin
        self.max_grad_norm = max_grad_norm

        self.cell_encoder = HierarchicalDegradationEncoder(
            in_channels,
            hierarchy_channels,
            input_height,
            hierarchy_heads,
            cycle_layers=hierarchy_cycle_layers,
            dropout=model_dropout,
        )
        self.cross_attention = nn.ModuleList()
        self.ori_head = ProbabilisticRegressionHead(
            hierarchy_channels,
            head_hidden_channels,
            dropout=model_dropout,
        )
        self.relation_blocks = nn.ModuleList([
            GatedBidirectionalRelationBlock(
                hierarchy_channels,
                hierarchy_heads,
                mlp_ratio=relation_mlp_ratio,
                dropout=model_dropout,
            )
            for _ in range(relation_layers)
        ])
        self.support_head = ReferenceCandidateHead(
            hierarchy_channels,
            head_hidden_channels,
            dropout=model_dropout,
        )
        self.aggregator = GainAwareRobustAggregator(
            head_hidden_channels,
            head_hidden_channels,
            temperature=aggregation_temperature,
            robust_scale=aggregation_robust_scale,
            uncertainty_penalty=aggregation_uncertainty_penalty,
            dropout=model_dropout,
            use_gain_aware_weights=use_gain_aware_weights,
            use_adaptive_gate=use_adaptive_gate,
            fixed_gate=fixed_gate,
        )

    def _compute_all(self,
                     feature: torch.Tensor,
                     support_feature: torch.Tensor,
                     support_label: torch.Tensor):
        batch_size, support_size, channels, height, width = \
            support_feature.size()
        target_tokens = self.cell_encoder(feature)
        reference_tokens = self.cell_encoder(
            support_feature.view(-1, channels, height, width))
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

        relation_target = target_pair_tokens
        relation_reference = reference_tokens
        if self.use_bidirectional_relation:
            for block in self.relation_blocks:
                relation_target, relation_reference = block(
                    relation_target, relation_reference)

        y_ori, y_ori_log_variance, target_representation = \
            self.ori_head(target_tokens)
        flat_support_label = support_label.reshape(-1)
        flat_y_ori = (
            y_ori.unsqueeze(1)
            .expand(-1, support_size)
            .reshape(-1)
        )
        (
            flat_y_support,
            flat_delta,
            flat_support_log_variance,
            flat_gain_score,
            pair_tokens,
        ) = self.support_head(
            relation_target,
            relation_reference,
            flat_y_ori,
            flat_support_label,
        )
        y_support = flat_y_support.view(batch_size, support_size)
        support_delta = flat_delta.view(batch_size, support_size)
        support_log_variance = flat_support_log_variance.view(
            batch_size, support_size)
        gain_score = flat_gain_score.view(batch_size, support_size)

        aggregation = self.aggregator(
            y_ori,
            y_ori_log_variance,
            target_representation,
            y_support,
            support_log_variance,
            gain_score,
        )
        return {
            'prediction': aggregation['prediction'],
            'y_ori': y_ori,
            'y_ori_log_variance': y_ori_log_variance,
            'y_sup': y_support,
            'y_sup_agg': aggregation['y_sup_agg'],
            'support_delta': support_delta,
            'support_log_variance': support_log_variance,
            'support_score': gain_score,
            'support_weight': aggregation['support_weight'],
            'correction_gate': aggregation['correction_gate'],
            'support_dispersion': aggregation['support_dispersion'],
            'support_entropy': aggregation['support_entropy'],
            'target_tokens': target_tokens,
            'reference_tokens': reference_tokens.view(
                batch_size, support_size, token_count, self.channels),
            'pair_tokens': pair_tokens.view(
                batch_size, support_size, token_count, self.channels),
        }

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                support_feature: torch.Tensor,
                support_label: torch.Tensor,
                return_loss: bool = False):
        outputs = self._compute_all(
            feature, support_feature, support_label)
        if not return_loss:
            return outputs['prediction']
        return self._training_loss(outputs, label, support_label)

    def _training_loss(self,
                       outputs: dict,
                       label: torch.Tensor,
                       support_label: torch.Tensor):
        y_true = label.view(-1)
        y_pair_true = y_true.unsqueeze(1)
        true_delta = y_pair_true - support_label

        final_loss = mse(outputs['prediction'], y_true)
        ori_loss = mse(outputs['y_ori'], y_true)
        pair_delta_loss = mse(outputs['support_delta'], true_delta)

        ori_error = (outputs['y_ori'].detach() - y_true).abs()
        support_error = (outputs['y_sup'].detach() - y_pair_true).abs()
        true_gain = ori_error.unsqueeze(1) - support_error
        true_gain = true_gain - true_gain.mean(dim=1, keepdim=True)
        true_gain = true_gain / true_gain.std(
            dim=1, keepdim=True, unbiased=False).clamp_min(0.1)
        gain_loss = F.smooth_l1_loss(
            outputs['support_score'], true_gain)
        ranking_loss = self._ranking_loss(
            outputs['support_score'], true_gain)

        ori_squared_error = (outputs['y_ori'] - y_true).square()
        ori_nll = (
            torch.exp(-outputs['y_ori_log_variance'])
            * ori_squared_error
            + outputs['y_ori_log_variance']
        ).mean()
        support_squared_error = (
            outputs['y_sup'] - y_pair_true).square()
        support_nll = (
            torch.exp(-outputs['support_log_variance'])
            * support_squared_error
            + outputs['support_log_variance']
        ).mean()
        uncertainty_loss = 0.5 * (ori_nll + support_nll)

        return sum([
            self.final_loss_weight * final_loss,
            self.ori_loss_weight * ori_loss,
            self.pair_delta_loss_weight * pair_delta_loss,
            self.gain_loss_weight * gain_loss,
            self.ranking_loss_weight * ranking_loss,
            self.uncertainty_loss_weight * uncertainty_loss,
        ])

    def _ranking_loss(self,
                      predicted_gain: torch.Tensor,
                      true_gain: torch.Tensor):
        if predicted_gain.size(1) < 2:
            return predicted_gain.new_zeros(())
        predicted_difference = (
            predicted_gain.unsqueeze(2) - predicted_gain.unsqueeze(1))
        true_difference = true_gain.unsqueeze(2) - true_gain.unsqueeze(1)
        pair_mask = torch.triu(
            torch.ones_like(true_difference, dtype=torch.bool), diagonal=1)
        pair_mask &= true_difference.abs() > self.ranking_margin
        if not pair_mask.any():
            return predicted_gain.new_zeros(())
        direction = true_difference.sign()
        return F.softplus(
            -direction[pair_mask] * predicted_difference[pair_mask]
        ).mean()

    def fit(self, dataset: DataBundle, timestamp: str):
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        train_dataset = self.build_cell_dataset(dataset.train_data)
        support_feature = train_dataset.feature
        ori_loader = DataLoader(
            train_dataset, self.train_batch_size, shuffle=True)

        latest = None
        optimizer.zero_grad(set_to_none=True)
        for epoch in tqdm(range(self.train_epochs), desc='Training'):
            self.train()
            for batch_index, data_batch in enumerate(ori_loader):
                x, y = data_batch.values()
                sup_x, sup_y = self.get_support_set(
                    x,
                    support_feature,
                    dataset.train_data.label,
                    support_is_prepared=True,
                )
                loss = self.forward(
                    x, y, sup_x, sup_y, return_loss=True)
                (loss / self.grad_accum_steps).backward()

                should_step = (
                    batch_index == len(ori_loader) - 1
                    or (batch_index + 1) % self.grad_accum_steps == 0
                )
                if should_step:
                    if self.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if (
                self.workspace is not None
                and self.checkpoint_freq is not None
                and (epoch + 1) % self.checkpoint_freq == 0
            ):
                filename = self.workspace / (
                    f'{timestamp}_seed_{self.seed}_epoch_{epoch+1}.ckpt')
                self.dump_checkpoint(filename)
                latest = filename

            if (epoch + 1) % self.evaluate_freq == 0:
                prediction = self.predict(dataset)
                score = dataset.evaluate(prediction, 'RMSE')
                print(
                    f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}',
                    flush=True,
                )

        if latest is not None and self.workspace is not None:
            self.link_latest_checkpoint(latest)

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        outputs = self._compute_all(
            feature, support_feature, support_label)
        components = (
            outputs['y_ori'],
            outputs['y_sup'],
            outputs['y_sup_agg'],
            outputs['support_weight'],
            outputs['support_score'],
        )
        if return_features:
            return components + (
                outputs['target_tokens'],
                outputs['reference_tokens'],
            )
        return components

    @torch.no_grad()
    def predict(self,
                dataset: DataBundle,
                return_diagnostics: bool = False) -> torch.Tensor:
        self.eval()
        test_dataset = self.build_cell_dataset(dataset.test_data)
        ori_loader = DataLoader(
            test_dataset, self.test_batch_size, shuffle=False)
        fixed_indices = self.load_fixed_test_support_indices(dataset)
        support_feature = self._prepare_feature(dataset.train_data.feature)
        predictions = []
        diagnostics = None
        if return_diagnostics:
            diagnostic_names = [
                'y_ori',
                'y_ori_log_variance',
                'y_sup',
                'y_sup_agg',
                'support_delta',
                'support_log_variance',
                'support_score',
                'support_weight',
                'correction_gate',
                'support_dispersion',
                'support_entropy',
            ]
            diagnostics = {name: [] for name in diagnostic_names}
            diagnostics['support_index'] = []
            diagnostics['final_prediction'] = []

        offset = 0
        for data_batch in ori_loader:
            x, _ = data_batch.values()
            batch_fixed_indices = None
            if fixed_indices is not None:
                batch_fixed_indices = fixed_indices[offset:offset + len(x)]
                offset += len(x)
            sup_x, sup_y, sup_index = self.get_support_set(
                x,
                support_feature,
                dataset.train_data.label,
                fixed_indices=batch_fixed_indices,
                return_indices=True,
                support_is_prepared=True,
            )
            outputs = self._compute_all(x, sup_x, sup_y)
            predictions.append(outputs['prediction'])
            if return_diagnostics:
                for name in diagnostics:
                    if name in outputs:
                        diagnostics[name].append(outputs[name])
                diagnostics['support_index'].append(sup_index)
                diagnostics['final_prediction'].append(outputs['prediction'])

        predictions = torch.cat(predictions)
        if not return_diagnostics:
            return predictions
        diagnostics = {
            name: torch.cat(values) for name, values in diagnostics.items()
        }
        return predictions, diagnostics

    @torch.no_grad()
    def get_support_set(self,
                        x,
                        sup_feat,
                        sup_label,
                        fixed_indices=None,
                        return_indices: bool = False,
                        support_is_prepared: bool = False):
        if not support_is_prepared:
            sup_feat = self._prepare_feature(sup_feat)
        if fixed_indices is not None:
            index = fixed_indices.to(x.device).long().contiguous()
            if index.dim() != 2 or index.size(0) != len(x):
                raise ValueError(
                    'fixed_indices must have shape [batch, support_size].')
        else:
            if self.training:
                support_size = int(torch.randint(
                    self.train_support_size_min,
                    self.train_support_size + 1,
                    (1,),
                ).item())
            else:
                support_size = self.test_support_size
            index = torch.randint(
                len(sup_feat),
                (len(x), support_size),
                device=x.device,
            )
        batch_size = x.size(0)
        flat_index = index.reshape(-1)
        feature = sup_feat[flat_index].view(
            batch_size, -1, *x.shape[1:])
        label = sup_label[flat_index].view(batch_size, -1)
        if return_indices:
            return feature, label, index.view(batch_size, -1)
        return feature, label
