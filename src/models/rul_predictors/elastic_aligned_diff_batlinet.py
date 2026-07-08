import torch

from src.builders import MODELS

from .batlinet import BatLiNetRULPredictor


@MODELS.register()
class ElasticAlignedDiffBatLiNetRULPredictor(BatLiNetRULPredictor):
    """BatLiNet variant using local elastic alignment for support differences."""

    def __init__(self,
                 elastic_window_size: int = 8,
                 elastic_cost_reduction: str = 'channel_mean',
                 **kwargs):
        super().__init__(**kwargs)
        if elastic_window_size < 0:
            raise ValueError('elastic_window_size must be non-negative.')
        if elastic_cost_reduction != 'channel_mean':
            raise ValueError(
                "Only elastic_cost_reduction='channel_mean' is supported.")
        self.elastic_window_size = int(elastic_window_size)
        self.elastic_cost_reduction = elastic_cost_reduction

    @torch.no_grad()
    def get_support_set(self,
                        x,
                        sup_feat,
                        sup_label,
                        fixed_indices=None,
                        return_indices: bool = False):
        if self.features_to_drop is not None:
            mask = [i for i in range(sup_feat.size(1))
                    if i not in self.features_to_drop]
            sup_feat = sup_feat[:, mask].contiguous()
        if self.cycles_to_drop is not None:
            sup_feat[:, :, :, self.cycles_to_drop] = 0.

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
        selected_support = sup_feat[flat_indx].view(B, -1, C, H, W)
        feature = self._build_elastic_aligned_diff(x, selected_support)
        label = sup_label[flat_indx].view(B, -1)
        feature = self._clean_feature(feature)
        if return_indices:
            return feature, label, indx.view(B, -1)
        return feature, label

    def _build_elastic_aligned_diff(self,
                                    target: torch.Tensor,
                                    support: torch.Tensor) -> torch.Tensor:
        """Match each target point to a nearby support point before subtraction.

        target has shape [B, C, H, W], support has shape [B, S, C, H, W].
        The returned tensor keeps BatLiNet's original support shape
        [B, S, C, H, W].
        """
        if self.elastic_window_size == 0:
            return target.unsqueeze(1) - support

        target = target.unsqueeze(1)
        best_cost = None
        best_support = None
        for offset in range(-self.elastic_window_size,
                            self.elastic_window_size + 1):
            shifted_support = self._shift_capacity_axis(support, offset)
            cost = (target - shifted_support).pow(2).mean(dim=2)
            if best_cost is None:
                best_cost = cost
                best_support = shifted_support
                continue
            update_mask = cost < best_cost
            best_cost = torch.where(update_mask, cost, best_cost)
            best_support = torch.where(
                update_mask.unsqueeze(2), shifted_support, best_support)
        return target - best_support

    @staticmethod
    def _shift_capacity_axis(x: torch.Tensor, offset: int) -> torch.Tensor:
        if offset == 0:
            return x
        width = x.size(-1)
        index = torch.arange(width, device=x.device) + offset
        index = index.clamp_(0, width - 1)
        return x.index_select(-1, index)
