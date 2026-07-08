import torch

from src.builders import MODELS

from .batlinet import BatLiNetRULPredictor


@MODELS.register()
class RelativeDegradationDiffBatLiNetRULPredictor(BatLiNetRULPredictor):
    """BatLiNet variant using target-reference differences of early degradation."""

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

        target_rel = x - x[:, :, [self.diff_base]]
        support_rel = sup_feat - sup_feat[:, :, [self.diff_base]]
        if self.cycles_to_drop is not None:
            target_rel[:, :, self.cycles_to_drop] = 0.
            support_rel[:, :, self.cycles_to_drop] = 0.

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

        B, C, H, W = target_rel.size()
        flat_indx = indx.view(-1)
        feature = (
            target_rel.unsqueeze(1)
            - support_rel[flat_indx].view(B, -1, C, H, W)
        )
        label = sup_label[flat_indx].view(B, -1)
        feature = self._clean_feature(feature)
        if return_indices:
            return feature, label, indx.view(B, -1)
        return feature, label
