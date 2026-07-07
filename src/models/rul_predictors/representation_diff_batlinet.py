from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from src.builders import MODELS
from src.data.databundle import DataBundle, Dataset

from .batlinet import build_module, mse, remove_glitches, smoothing
from ..nn_model import NNModel


@MODELS.register()
class RepresentationDiffBatLiNetRULPredictor(NNModel):
    """BatLiNet variant that builds target-support differences in latent space."""

    def __init__(self,
                 in_channels: int,
                 channels: int,
                 input_height: int,
                 input_width: int,
                 alpha: float = 0.5,
                 kernel_size: int = 3,
                 train_support_size: int = None,
                 test_support_size: int = None,
                 gradient_accumulation_steps: int = 1,
                 support_size: int = 1,
                 lr: float = 1e-3,
                 act_fn: str = 'relu',
                 filter_cycles: bool = True,
                 features_to_drop: list = None,
                 cycles_to_drop: list = None,
                 return_pointwise_predictions: bool = False,
                 fixed_test_support_index_path: str = None,
                 seed: int = 0,
                 **kwargs):
        NNModel.__init__(self, **kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_height < kernel_size[0]:
            kernel_size = (input_height, kernel_size[1])
        if input_width < kernel_size[1]:
            kernel_size = (kernel_size[0], input_width)

        self.alpha = alpha
        self.channels = channels
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

        self.cell_encoder = build_module(
            in_channels, channels,
            input_height, input_width,
            kernel_size, act_fn)
        self.fc_ori = nn.Linear(channels, 1, bias=False)
        self.fc_sup = nn.Linear(channels, 1, bias=False)
        self.lr = lr
        self.seed = seed

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

        return (1. - self.alpha) * y_ori + self.alpha * y_sup_agg

    def compute_prediction_components(self,
                                      feature: torch.Tensor,
                                      support_feature: torch.Tensor,
                                      support_label: torch.Tensor,
                                      return_features: bool = False):
        B, S, C, H, W = support_feature.size()

        z_ori = self.cell_encoder(feature).view(B, self.channels)
        z_sup = self.cell_encoder(
            support_feature.view(-1, C, H, W)).view(B, S, self.channels)
        z_diff = z_ori.unsqueeze(1) - z_sup

        y_ori = self.fc_ori(z_ori).view(-1)
        y_sup = self.fc_sup(z_diff).view(B, S)
        y_sup += support_label.view(B, S)

        if self.training:
            y_sup_agg = y_sup.mean(1).view(-1)
        else:
            y_sup_agg = y_sup.median(1)[0].view(-1)

        if return_features:
            return y_ori, y_sup, y_sup_agg, None, None, z_ori, z_sup
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
                loss.backward()

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
                filename = self.workspace / f'{timestamp}_seed_{self.seed}_epoch_{epoch+1}.ckpt'
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
                y_ori, y_sup, y_sup_agg, weight, _ = \
                    self.compute_prediction_components(x, sup_x, sup_y)
                pred = (1. - self.alpha) * y_ori + self.alpha * y_sup_agg
                predictions.append(pred)
                diagnostics['y_ori'].append(y_ori)
                diagnostics['y_sup'].append(y_sup)
                diagnostics['y_sup_agg'].append(y_sup_agg)
                diagnostics['support_index'].append(sup_indx)
                if weight is not None:
                    diagnostics['support_weight'].append(weight)
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
        diagnostics = {
            'y_ori': torch.cat(diagnostics['y_ori']),
            'y_sup': torch.cat(diagnostics['y_sup']),
            'y_sup_agg': torch.cat(diagnostics['y_sup_agg']),
            'support_index': torch.cat(diagnostics['support_index']),
            'support_weight': support_weight,
        }
        return predictions, diagnostics

    @torch.no_grad()
    def build_cell_dataset(self, dataset: Dataset):
        feature = self._prepare_feature(dataset.feature)
        return Dataset(feature, dataset.label)

    @torch.no_grad()
    def build_cycle_diff_dataset(self, dataset: Dataset):
        return self.build_cell_dataset(dataset)

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
        B, C, H, W = x.size()
        flat_indx = indx.view(-1)
        feature = sup_feat[flat_indx].view(B, -1, C, H, W)
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
                    f'No support indices found in {self.fixed_test_support_index_path}.')
            if indices.dim() != 2:
                raise ValueError('Fixed test support indices must be a 2D tensor.')
            if indices.size(0) != len(dataset.test_data):
                raise ValueError(
                    'Fixed test support protocol does not match the number of test samples.')
            if indices.size(1) != self.test_support_size:
                raise ValueError(
                    'Fixed test support protocol does not match test_support_size.')
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


