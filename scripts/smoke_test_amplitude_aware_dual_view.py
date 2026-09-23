"""Check amplitude sensitivity and training for the dual-view variant."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rul_predictors.amplitude_aware_dual_view_batlinet import (
    AmplitudeAwareDualViewLatentCrossAttentionBatLiNetRULPredictor,
)


def check_task(cycles: int):
    torch.manual_seed(cycles)
    model = AmplitudeAwareDualViewLatentCrossAttentionBatLiNetRULPredictor(
        in_channels=6,
        channels=32,
        input_height=cycles,
        input_width=1000,
        train_support_size=2,
        test_support_size=32,
        attention_channels=64,
        attention_heads=4,
        attention_layers=1,
        capacity_bins=32,
        stem_channels=32,
        cycle_encoder_layers=1,
        epochs=1,
        train_batch_size=1,
        test_batch_size=1,
    )
    target = torch.randn(1, 6, cycles, 1000)
    support = torch.randn(1, 2, 6, cycles, 1000)
    if cycles == 20:
        target[:, :, 10] = 0
        support[:, :, :, 10] = 0
    label = torch.randn(1)
    support_label = torch.randn(1, 2)

    model.eval()
    with torch.no_grad():
        stem = model.cell_encoder.local_stem
        curve = target[:, :, 0]
        normalized = stem.normalized_stem(curve)
        normalized_scaled = stem.normalized_stem(2 * curve)
        encoded = stem(curve)
        encoded_scaled = stem(2 * curve)
        normalized_change = (
            (normalized_scaled - normalized).norm() / normalized.norm())
        amplitude_change = (
            (encoded_scaled - encoded).norm() / encoded.norm())
        assert normalized_change < 1e-4
        assert amplitude_change > 1e-3

        cycle_tokens, capacity_tokens, _, mask = model.cell_encoder(target)
        assert cycle_tokens.shape == (1, cycles, 64)
        assert capacity_tokens.shape == (1, 32, 64)
        assert mask.shape == (1, cycles)
        if cycles == 20:
            assert not mask[:, 10].any()
            assert torch.equal(
                cycle_tokens[:, 10],
                torch.zeros_like(cycle_tokens[:, 10]),
            )

    model.train()
    loss = model(target, label, support, support_label, return_loss=True)
    loss.backward()
    assert torch.isfinite(loss)
    gate_gradient = model.cell_encoder.local_stem.gate_logit.grad
    assert gate_gradient is not None and torch.isfinite(gate_gradient).all()
    assert gate_gradient.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        prediction = model(target, label, support, support_label)
        reordered = model(
            target,
            label,
            support.flip(1),
            support_label.flip(1),
        )
    assert prediction.shape == (1,) and torch.isfinite(prediction).all()
    assert torch.allclose(prediction, reordered, atol=1e-5)
    print(
        f'H={cycles}: normalized_change={normalized_change.item():.6g}, '
        f'amplitude_change={amplitude_change.item():.6g}, '
        f'loss={loss.item():.6g}'
    )


if __name__ == '__main__':
    check_task(20)
    check_task(100)
