"""Branch ablations of the cycle-mixer/shared-value-difference model.

The three modes start from the same *untrained* complete model. Unused modules
are deleted, rather than merely assigning their losses zero weights. Prediction
selection is explicit and independent of the fixed training-loss coefficients.
"""
import copy

import torch

from .batlinet import mse
from .cycle_mixer_soft_aligned_difference_batlinet import (
    CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor,
)


BRANCH_MODES = ('full', 'self_only', 'reference_only')


class BranchAblationBatLiNetRULPredictor(
        CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor):
    """Keep only modules required by the selected branch-training condition.

    Use :func:`build_branch_model` for seed-aligned, RNG-isolated construction.
    Inputs to ``forward`` and ``predict_components`` are prepared by default,
    matching the original predictor. Set ``prepared=False`` for raw features.
    The experiment runner handles training and support sampling explicitly.
    """

    def __init__(self, *args, mode='full', **kwargs):
        if mode not in BRANCH_MODES:
            raise ValueError(f'Unknown branch mode: {mode!r}.')
        if kwargs.get('alpha', 0.5) != 0.5:
            raise ValueError('Branch ablation fixes both full-model weights at 0.5.')
        super().__init__(*args, **kwargs)
        self.branch_mode = mode
        if mode == 'self_only':
            del self.cross_attention
            del self.support_head
        elif mode == 'reference_only':
            del self.ori_head

    @property
    def requires_reference(self):
        return self.branch_mode != 'self_only'

    def predict_components(self, feature, support_feature=None,
                           support_label=None, prepared=True):
        """Return predictions in the standardized log-lifetime space.

        ``y_sup`` contains each reference's delta prediction plus its known
        label; ``y_sup_agg`` is the mean during training and median in eval.
        Missing branches are ``None``. The self-only mode ignores reference
        arguments entirely and never prepares or encodes reference inputs.
        """
        if feature.ndim != 4:
            raise ValueError('feature must have shape [batch, channels, cycles, points].')
        if not prepared:
            feature = self._prepare_feature(feature)

        if self.requires_reference:
            if support_feature is None or support_label is None:
                raise ValueError('This branch mode requires reference features and labels.')
            if support_feature.ndim != 5:
                raise ValueError('support_feature must have shape [batch, references, channels, cycles, points].')
            batch, count, channels, height, width = support_feature.shape
            if batch != len(feature) or count < 1:
                raise ValueError('Reference batch must match targets and contain at least one reference.')
            if tuple(support_label.shape) != (batch, count):
                raise ValueError('support_label must have shape [batch, references].')
            if not prepared:
                support_feature = self._prepare_feature(
                    support_feature.reshape(-1, channels, height, width))
                support_feature = support_feature.reshape(
                    batch, count, *support_feature.shape[1:])
            if support_feature.shape[2:] != feature.shape[1:]:
                raise ValueError('Prepared target and reference feature shapes must match.')

        target_tokens = self.cell_encoder(feature)
        if self.branch_mode == 'self_only':
            y_ori = self.ori_head(target_tokens).view(-1)
            return dict(prediction=y_ori, y_ori=y_ori, y_sup=None, y_sup_agg=None)

        # Keep the full-model operation order identical to the existing model,
        # including the order of dropout calls in the two heads.
        support_tokens = self.cell_encoder(support_feature.reshape(
            batch * count, *support_feature.shape[2:]))
        tokens = target_tokens.size(1)
        relation_tokens = (target_tokens.unsqueeze(1)
                           .expand(-1, count, -1, -1)
                           .reshape(batch * count, tokens, self.channels))
        for block in self.cross_attention:
            relation_tokens = block(relation_tokens, support_tokens)
        y_ori = (self.ori_head(target_tokens).view(-1)
                 if self.branch_mode == 'full' else None)
        y_sup = self.support_head(relation_tokens).view(batch, count) + support_label
        y_sup_agg = (y_sup.mean(1) if self.training else y_sup.median(1)[0]).view(-1)
        prediction = (0.5 * y_ori + 0.5 * y_sup_agg
                      if self.branch_mode == 'full' else y_sup_agg)
        return dict(prediction=prediction, y_ori=y_ori,
                    y_sup=y_sup, y_sup_agg=y_sup_agg)

    def loss_from_components(self, components, label):
        """Apply the same 0.5 coefficient to every retained branch loss."""
        if label is None:
            raise ValueError('Training loss requires target labels.')
        if self.branch_mode == 'self_only':
            return 0.5 * mse(components['y_ori'], label)
        if self.branch_mode == 'reference_only':
            return 0.5 * mse(components['y_sup_agg'], label)
        return (0.5 * mse(components['y_ori'], label)
                + 0.5 * mse(components['y_sup_agg'], label))

    def forward(self, feature, label=None, support_feature=None,
                support_label=None, return_loss=False, prepared=True):
        components = self.predict_components(
            feature, support_feature, support_label, prepared=prepared)
        if return_loss:
            return self.loss_from_components(components, label)
        return components['prediction']

    def compute_prediction_components(self, feature, support_feature=None,
                                      support_label=None, return_features=False):
        """Compatibility with the original five-value diagnostic interface."""
        if return_features:
            raise ValueError('Use predict_components for branch-ablation diagnostics.')
        result = self.predict_components(feature, support_feature, support_label)
        return result['y_ori'], result['y_sup'], result['y_sup_agg'], None, None

    def combine_predictions(self, y_ori, y_sup_agg):
        if self.branch_mode == 'self_only':
            return y_ori
        if self.branch_mode == 'reference_only':
            return y_sup_agg
        return 0.5 * y_ori + 0.5 * y_sup_agg


def build_branch_model(config, seed, mode=None):
    """Initialize on CPU without changing the caller's random-number stream.

    ``config`` may be a complete experiment configuration or its ``model``
    dictionary. All modes construct the complete combination first, so every
    retained parameter is identical at the same seed, including the two heads.
    Only the CPU default generator is reseeded; CUDA RNG states are untouched.
    """
    settings = copy.deepcopy(config.get('model', config))
    settings.pop('name', None)
    settings.pop('seed', None)
    configured_mode = settings.pop('branch_mode', None)
    legacy_mode = settings.pop('mode', None)
    if legacy_mode is not None:
        if configured_mode is not None and configured_mode != legacy_mode:
            raise ValueError('Conflicting branch mode fields in model configuration.')
        configured_mode = legacy_mode
    if mode is not None and configured_mode is not None and mode != configured_mode:
        raise ValueError('Explicit branch mode does not match model configuration.')
    mode = mode if mode is not None else (configured_mode or 'full')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return BranchAblationBatLiNetRULPredictor(
            **settings, seed=seed, mode=mode)
