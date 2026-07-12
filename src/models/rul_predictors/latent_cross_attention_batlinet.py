from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.builders import MODELS
from src.data.databundle import DataBundle, Dataset

from .batlinet import mse, remove_glitches, smoothing
from ..nn_model import NNModel


class ConvTokenEncoder(nn.Module):
    """Encode one battery curve tensor into local latent tokens."""

    def __init__(self,
                 in_channels: int,
                 token_channels: int,
                 input_height: int,
                 input_width: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, token_channels // 2,
                      kernel_size=(3, 9), padding=(1, 4)),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Conv2d(token_channels // 2, token_channels,
                      kernel_size=(3, 7), padding=(1, 3)),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(2, 4)),
            nn.Conv2d(token_channels, token_channels,
                      kernel_size=(3, 5), padding=(1, 2)),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Dropout2d(dropout),
        )
        out_h = input_height // 2 // 2
        out_w = input_width // 4 // 4 // 2
        if out_h <= 0 or out_w <= 0:
            raise ValueError('Token encoder output shape is empty.')
        self.num_tokens = out_h * out_w
        self.token_channels = token_channels
        self.position = nn.Parameter(
            torch.zeros(1, self.num_tokens, token_channels))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        if x.size(1) != self.position.size(1):
            raise ValueError(
                f'Unexpected token count {x.size(1)}; '
                f'expected {self.position.size(1)}.')
        return x + self.position


class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 num_heads: int,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(channels)
        self.kv_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
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
                query_tokens: torch.Tensor,
                reference_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(query_tokens)
        kv = self.kv_norm(reference_tokens)
        attended, _ = self.attn(q, kv, kv, need_weights=False)
        x = query_tokens + self.attn_dropout(attended)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TokenRegressionHead(nn.Module):
    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mean_pool = tokens.mean(dim=-2)
        max_pool = tokens.max(dim=-2)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.net(pooled).squeeze(-1)


class RelationTokenFusionHead(nn.Module):
    """Fuse explicit latent relation tokens before support delta regression."""

    def __init__(self,
                 channels: int,
                 hidden_channels: int,
                 relation_parts: int = 5,
                 mlp_ratio: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(channels * relation_parts, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
            nn.GELU(),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * mlp_ratio, channels),
            nn.Dropout(dropout),
        )
        self.regressor = TokenRegressionHead(
            channels,
            hidden_channels,
            dropout=dropout,
        )

    def forward(self,
                target_tokens: torch.Tensor,
                support_tokens: torch.Tensor,
                relation_tokens: torch.Tensor) -> torch.Tensor:
        relation_delta = relation_tokens - target_tokens
        relation_product = target_tokens * relation_tokens
        fused_tokens = torch.cat([
            target_tokens,
            support_tokens,
            relation_tokens,
            relation_delta,
            relation_product,
        ], dim=-1)
        fused_tokens = self.fusion(fused_tokens)
        fused_tokens = fused_tokens + self.context_mlp(fused_tokens)
        return self.regressor(fused_tokens)


@MODELS.register()
class LatentCrossAttentionBatLiNetRULPredictor(NNModel):
    """BatLiNet variant with pure latent target-support cross-attention."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 gradient_accumulation_steps: int = 1,
                 support_size: int = 1,
                 lr: float = 1e-3,
                 filter_cycles: bool = True,
                 features_to_drop: list = None,
                 cycles_to_drop: list = None,
                 return_pointwise_predictions: bool = False,
                 fixed_test_support_index_path: str = None,
                 seed: int = 0,
                 attention_channels: int = 64,
                 attention_heads: int = 4,
                 attention_layers: int = 1,
                 attention_dropout: float = 0.1,
                 attention_mlp_ratio: int = 2,
                 head_hidden_channels: int = None,
                 encoder_dropout: float = 0.1,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        if attention_channels % attention_heads != 0:
            raise ValueError(
                'attention_channels must be divisible by attention_heads.')

        self.alpha = alpha
        self.channels = attention_channels
        self.train_support_size = train_support_size or support_size
        self.test_support_size = test_support_size or support_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.filter_cycles = filter_cycles
        if isinstance(features_to_drop, int):
            features_to_drop = [features_to_drop]
        self.features_to_drop = features_to_drop
        if isinstance(cycles_to_drop, int):
            cycles_to_drop = [cycles_to_drop]
        self.cycles_to_drop = cycles_to_drop
        self.return_pointwise_predictions = return_pointwise_predictions
        self.fixed_test_support_index_path = fixed_test_support_index_path
        self._fixed_test_support_index = None
        self.lr = lr
        self.seed = seed

        self.cell_encoder = ConvTokenEncoder(
            in_channels,
            attention_channels,
            input_height,
            input_width,
            dropout=encoder_dropout,
        )
        self.cross_attention = nn.ModuleList([
            CrossAttentionBlock(
                attention_channels,
                attention_heads,
                mlp_ratio=attention_mlp_ratio,
                dropout=attention_dropout,
            )
            for _ in range(attention_layers)
        ])
        head_hidden_channels = head_hidden_channels or attention_channels
        self.ori_head = TokenRegressionHead(
            attention_channels,
            head_hidden_channels,
            dropout=attention_dropout,
        )
        self.support_head = TokenRegressionHead(
            attention_channels,
            head_hidden_channels,
            dropout=attention_dropout,
        )

    def forward(self,
                feature: torch.Tensor,
                label: torch.Tensor,
                support_feature: torch.Tensor,
                support_label: torch.Tensor,
                return_loss: bool = False):
        y_ori, y_sup, y_sup_agg, _, _ = self.compute_prediction_components(
            feature, support_feature, support_label)

        if self.return_pointwise_predictions:
            return y_ori, y_sup

        if return_loss:
            loss = sum([
                (1. - self.alpha) * mse(y_ori, label),
                self.alpha * mse(y_sup_agg, label)
            ])
            return loss

        return self.combine_predictions(y_ori, y_sup_agg)

    def combine_predictions(self,
                            y_ori: torch.Tensor,
                            y_sup_agg: torch.Tensor) -> torch.Tensor:
        """Combine the target-only and aggregated support predictions."""
        return (1. - self.alpha) * y_ori + self.alpha * y_sup_agg

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        B, S, C, H, W = support_feature.size()

        target_tokens = self.cell_encoder(feature)
        support_tokens = self.cell_encoder(
            support_feature.view(-1, C, H, W))

        T = target_tokens.size(1)
        target_pair_tokens = (
            target_tokens.unsqueeze(1)
            .expand(-1, S, -1, -1)
            .reshape(B * S, T, self.channels)
        )
        relation_tokens = target_pair_tokens
        for block in self.cross_attention:
            relation_tokens = block(relation_tokens, support_tokens)

        y_ori = self.ori_head(target_tokens).view(-1)
        y_sup = self.support_head(relation_tokens).view(B, S)
        y_sup = y_sup + support_label.view(B, S)

        if self.training:
            y_sup_agg = y_sup.mean(1).view(-1)
        else:
            y_sup_agg = y_sup.median(1)[0].view(-1)

        if return_features:
            return (
                y_ori,
                y_sup,
                y_sup_agg,
                None,
                None,
                target_tokens,
                support_tokens.view(B, S, T, self.channels),
            )
        return y_ori, y_sup, y_sup_agg, None, None

    def fit(self, dataset: DataBundle, timestamp: str):
        self.train()
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        train_dataset = self.build_cell_dataset(dataset.train_data)
        support_feature = train_dataset.feature
        ori_loader = DataLoader(
            train_dataset, self.train_batch_size, shuffle=False)

        latest = None
        for epoch in tqdm(range(self.train_epochs), desc='Training'):
            self.train()

            for indx, data_batch in enumerate(ori_loader):
                x, y = data_batch.values()
                sup_x, sup_y = self.get_support_set(
                    x, support_feature, dataset.train_data.label,
                    support_is_prepared=True)
                loss = self.forward(x, y, sup_x, sup_y, return_loss=True)
                (loss / self.grad_accum_steps).backward()

                if (
                    indx == len(ori_loader) - 1
                    or (indx + 1) % self.grad_accum_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()

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
                del loss, sup_x, sup_y, x, y
                pred = self.predict(dataset)
                score = dataset.evaluate(pred, 'RMSE')
                message = f'[{epoch+1}/{self.train_epochs}] RMSE {score:.2f}'
                print(message, flush=True)
                del pred

        if latest is not None and self.workspace is not None:
            self.link_latest_checkpoint(latest)

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
        diagnostics = {
            'y_ori': [],
            'y_sup': [],
            'y_sup_agg': [],
            'support_index': [],
            'support_weight': [],
            'support_score': [],
        } if return_diagnostics else None
        offset = 0
        for _, data_batch in enumerate(ori_loader):
            x, y = data_batch.values()
            batch_fixed_indices = None
            if fixed_indices is not None:
                batch_fixed_indices = fixed_indices[offset:offset + len(x)]
                offset += len(x)
            sup_x, sup_y, sup_indx = self.get_support_set(
                x,
                support_feature,
                dataset.train_data.label,
                fixed_indices=batch_fixed_indices,
                return_indices=True,
                support_is_prepared=True)
            if return_diagnostics:
                y_ori, y_sup, y_sup_agg, weight, score = \
                    self.compute_prediction_components(x, sup_x, sup_y)
                pred = self.combine_predictions(y_ori, y_sup_agg)
                predictions.append(pred)
                diagnostics['y_ori'].append(y_ori)
                diagnostics['y_sup'].append(y_sup)
                diagnostics['y_sup_agg'].append(y_sup_agg)
                diagnostics['support_index'].append(sup_indx)
                if weight is not None:
                    diagnostics['support_weight'].append(weight)
                if score is not None:
                    diagnostics['support_score'].append(score)
            else:
                predictions.append(self.forward(x, y, sup_x, sup_y))

        if self.return_pointwise_predictions:
            predictions = (
                torch.cat([x[0] for x in predictions]),
                torch.cat([x[1] for x in predictions]),
            )
        else:
            predictions = torch.cat(predictions)
        if not return_diagnostics:
            return predictions

        support_weight = None
        if diagnostics['support_weight']:
            support_weight = torch.cat(diagnostics['support_weight'])
        support_score = None
        if diagnostics['support_score']:
            support_score = torch.cat(diagnostics['support_score'])
        diagnostics = {
            'y_ori': torch.cat(diagnostics['y_ori']),
            'y_sup': torch.cat(diagnostics['y_sup']),
            'y_sup_agg': torch.cat(diagnostics['y_sup_agg']),
            'support_index': torch.cat(diagnostics['support_index']),
            'support_weight': support_weight,
            'support_score': support_score,
        }
        return predictions, diagnostics

    @torch.no_grad()
    def build_cell_dataset(self, dataset: Dataset):
        feature = self._prepare_feature(dataset.feature)
        return Dataset(feature, dataset.label)

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
            indx = fixed_indices.to(x.device).long().contiguous()
            if indx.dim() != 2 or indx.size(0) != len(x):
                raise ValueError(
                    'fixed_indices must have shape [batch, support_size].')
        else:
            if self.training:
                size = (len(x) * self.train_support_size,)
            else:
                size = (len(x) * self.test_support_size,)
            indx = torch.randint(len(sup_feat), size, device=x.device)
        B = x.size(0)
        flat_indx = indx.view(-1)
        feature = sup_feat[flat_indx].view(B, -1, *x.shape[1:])
        label = sup_label[flat_indx].view(B, -1)
        if return_indices:
            return feature, label, indx.view(B, -1)
        return feature, label

    def load_fixed_test_support_indices(self, dataset: DataBundle):
        if self.fixed_test_support_index_path is None:
            return None
        if self._fixed_test_support_index is None:
            path = Path(self.fixed_test_support_index_path)
            payload = torch.load(path, map_location='cpu')
            if isinstance(payload, dict):
                indices = payload.get('indices')
            else:
                indices = payload
            if indices is None:
                raise ValueError(
                    f'No support indices found in '
                    f'{self.fixed_test_support_index_path}.')
            if indices.dim() != 2:
                raise ValueError(
                    'Fixed test support indices must be a 2D tensor.')
            if indices.size(0) != len(dataset.test_data):
                raise ValueError(
                    'Fixed test support protocol does not match the number '
                    'of test samples.')
            if indices.size(1) != self.test_support_size:
                raise ValueError(
                    'Fixed test support protocol does not match '
                    'test_support_size.')
            self._fixed_test_support_index = indices.long().contiguous()
        return self._fixed_test_support_index

    def _prepare_feature(self, feature):
        feature = feature.clone()
        if self.features_to_drop is not None:
            mask = [x for x in range(feature.size(1))
                    if x not in self.features_to_drop]
            feature = feature[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            feature[:, :, self.cycles_to_drop] = 0.
        feature = self._clean_feature(feature)
        return feature

    def _clean_feature(self, feature):
        num = 50
        feature[..., :num] = smoothing(feature[..., :num])
        feature[..., -num:] = smoothing(feature[..., -num:])
        feature = remove_glitches(feature)
        feature = self._filter_cycles(feature)
        return feature

    def _filter_cycles(self, feature):
        if not self.filter_cycles:
            return feature
        feature = feature.clone()

        max_val = feature.abs().amax(-1)
        max_val_med = max_val.median(-1, keepdim=True)[0]
        max_val_diff = (max_val - max_val_med).abs()
        mask = max_val_diff > max_val_diff.std(-1, keepdim=True) * 5

        mean_val = feature.mean(-1)
        mean_val_med = mean_val.median(-1, keepdim=True)[0]
        mean_val_diff = (mean_val - mean_val_med).abs()
        mask |= mean_val_diff > mean_val_diff.std(-1, keepdim=True) * 5

        feature[mask] = 0.
        return feature

@MODELS.register()
class LatentCrossAttentionRelationTokensBatLiNetRULPredictor(
        LatentCrossAttentionBatLiNetRULPredictor):
    """Latent cross-attention with explicit token-level relation features."""

    def __init__(self,
                 *args,
                 relation_mlp_ratio: int = 2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.support_head = RelationTokenFusionHead(
            self.channels,
            kwargs.get('head_hidden_channels') or self.channels,
            relation_parts=5,
            mlp_ratio=relation_mlp_ratio,
            dropout=kwargs.get('attention_dropout', 0.1),
        )

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        B, S, C, H, W = support_feature.size()

        target_tokens = self.cell_encoder(feature)
        support_tokens = self.cell_encoder(
            support_feature.view(-1, C, H, W))

        T = target_tokens.size(1)
        target_pair_tokens = (
            target_tokens.unsqueeze(1)
            .expand(-1, S, -1, -1)
            .reshape(B * S, T, self.channels)
        )
        relation_tokens = target_pair_tokens
        for block in self.cross_attention:
            relation_tokens = block(relation_tokens, support_tokens)

        y_ori = self.ori_head(target_tokens).view(-1)
        y_sup = self.support_head(
            target_pair_tokens,
            support_tokens,
            relation_tokens,
        ).view(B, S)
        y_sup = y_sup + support_label.view(B, S)

        if self.training:
            y_sup_agg = y_sup.mean(1).view(-1)
        else:
            y_sup_agg = y_sup.median(1)[0].view(-1)

        if return_features:
            return (
                y_ori,
                y_sup,
                y_sup_agg,
                None,
                None,
                target_tokens,
                support_tokens.view(B, S, T, self.channels),
            )
        return y_ori, y_sup, y_sup_agg, None, None
