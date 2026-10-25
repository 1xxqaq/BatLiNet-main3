"""Two untied branch networks with additive supervision and fixed fusion."""
import copy

import torch
from torch import nn

from .batlinet import mse
from .branch_ablation_batlinet import build_branch_model


STREAM_NAMES = ('self_only', 'reference_only')


class BranchRandomStreams:
    """Independent dropout streams matching separately seeded branch training.

    Both branches start at the same seed, as the existing single-branch runs
    do. Each call temporarily installs that branch's CPU and selected CUDA RNG
    states, then restores the caller's states. Reference sampling belongs to a
    separate generator managed by the runner. Calls are sequential, not thread
    safe; this class does not switch Python or NumPy randomness.
    """

    def __init__(self, seed, device='cpu'):
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Branch random streams support CPU or CUDA only.')
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA streams require an available CUDA device.')
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            self.device = torch.device('cuda', index)
            self._cuda_devices = [index]
        else:
            self.device = torch.device('cpu')
            self._cuda_devices = []
        cpu_state = torch.Generator(device='cpu').manual_seed(self.seed).get_state()
        cuda_state = (torch.Generator(device=self.device).manual_seed(self.seed).get_state()
                      if self._cuda_devices else None)
        self._states = {name: dict(cpu=cpu_state.clone(),
                                   cuda=cuda_state.clone() if cuda_state is not None else None)
                        for name in STREAM_NAMES}

    def call(self, name, function, *args, **kwargs):
        if name not in STREAM_NAMES:
            raise ValueError(f'Unknown branch random stream: {name!r}.')
        state = self._states[name]
        with torch.random.fork_rng(devices=self._cuda_devices):
            torch.set_rng_state(state['cpu'])
            if self._cuda_devices:
                torch.cuda.set_rng_state(state['cuda'], self.device)
            try:
                return function(*args, **kwargs)
            finally:
                self._states[name] = dict(cpu=torch.get_rng_state().clone(),
                    cuda=torch.cuda.get_rng_state(self.device).cpu().clone()
                    if self._cuda_devices else None)

    def state_dict(self):
        return dict(schema=1, seed=self.seed, device=str(self.device),
                    states=copy.deepcopy(self._states))

    def load_state_dict(self, payload):
        if (payload.get('schema') != 1 or payload.get('seed') != self.seed
                or payload.get('device') != str(self.device)
                or set(payload.get('states', {})) != set(STREAM_NAMES)):
            raise ValueError('Branch random state does not match seed, device or schema.')
        restored = {}
        for name in STREAM_NAMES:
            state = payload['states'][name]
            if set(state) != {'cpu', 'cuda'}:
                raise ValueError('Branch random state is incomplete.')
            cpu = self._validate_state(state['cpu'], 'cpu')
            cuda = (self._validate_state(state['cuda'], self.device)
                    if self._cuda_devices else None)
            if not self._cuda_devices and state['cuda'] is not None:
                raise ValueError('CPU branch state must not contain a CUDA state.')
            restored[name] = dict(cpu=cpu, cuda=cuda)
        self._states = restored

    @staticmethod
    def _validate_state(value, device):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or value.ndim != 1 or value.device.type != 'cpu'):
            raise ValueError('Random state must be a one-dimensional CPU byte tensor.')
        # Local generator validation leaves every global random stream intact.
        generator = torch.Generator(device=device)
        try:
            generator.set_state(value)
        except RuntimeError as error:
            raise ValueError('Invalid branch random-generator state.') from error
        return value.clone()


class NaiveBranchEnsembleBatLiNetRULPredictor(nn.Module):
    """No encoder sharing between branches; no learned or supervised fusion."""
    requires_reference = True
    branch_mode = 'naive_ensemble'

    def __init__(self, config, seed):
        super().__init__()
        settings = copy.deepcopy(config.get('model', config))
        for key in ('branch_mode', 'mode'):
            mode = settings.pop(key, None)
            if mode is not None and mode != self.branch_mode:
                raise ValueError('Configuration does not describe a naive branch ensemble.')
        self.self_branch = build_branch_model(settings, seed, 'self_only')
        self.reference_branch = build_branch_model(settings, seed, 'reference_only')
        for name in ('train_batch_size', 'test_batch_size', 'train_support_size',
                     'test_support_size', 'grad_accum_steps', 'train_epochs', 'lr', 'seed'):
            setattr(self, name, getattr(self.reference_branch, name))

    @property
    def own(self):
        return self.self_branch

    @property
    def relation(self):
        return self.reference_branch

    def _prepare_feature(self, feature):
        return self.relation._prepare_feature(feature)

    def get_support_set(self, *args, **kwargs):
        return self.relation.get_support_set(*args, **kwargs)

    def predict_components(self, feature, support_feature=None, support_label=None,
                           prepared=True, streams=None):
        if streams is not None and streams.device != feature.device:
            raise ValueError('Branch random streams must match the input device.')
        if streams is None:
            own = self.own.predict_components(feature, prepared=prepared)
            relation = self.relation.predict_components(
                feature, support_feature, support_label, prepared=prepared)
        else:
            own = streams.call('self_only', self.own.predict_components,
                               feature, prepared=prepared)
            relation = streams.call('reference_only', self.relation.predict_components,
                                    feature, support_feature, support_label,
                                    prepared=prepared)
        return dict(prediction=0.5 * own['y_ori'] + 0.5 * relation['y_sup_agg'],
                    y_ori=own['y_ori'], y_sup=relation['y_sup'],
                    y_sup_agg=relation['y_sup_agg'])

    def loss_from_components(self, components, label):
        if label is None:
            raise ValueError('Training loss requires target labels.')
        return (0.5 * mse(components['y_ori'], label)
                + 0.5 * mse(components['y_sup_agg'], label))

    def forward(self, feature, label=None, support_feature=None, support_label=None,
                return_loss=False, prepared=True, streams=None):
        components = self.predict_components(feature, support_feature, support_label,
                                             prepared=prepared, streams=streams)
        return (self.loss_from_components(components, label) if return_loss
                else components['prediction'])


def build_naive_model(config, seed):
    """Create two independently owned copies of same-seed branch parameters."""
    return NaiveBranchEnsembleBatLiNetRULPredictor(config, seed)
