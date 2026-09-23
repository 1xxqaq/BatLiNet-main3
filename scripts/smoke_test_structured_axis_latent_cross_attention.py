"""Data-free checks for the independent cycle/capacity-axis model."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rul_predictors.structured_axis_latent_cross_attention_batlinet import (
    StructuredAxisLatentCrossAttentionBatLiNetRULPredictor,
)


def check_task(cycles: int):
    torch.manual_seed(cycles)
    model = StructuredAxisLatentCrossAttentionBatLiNetRULPredictor(
        in_channels=6,
        channels=32,
        input_height=cycles,
        input_width=1000,
        attention_channels=64,
        attention_heads=4,
        attention_layers=1,
        patch_width=40,
        train_support_size=2,
        test_support_size=32,
        epochs=1,
        train_batch_size=1,
        test_batch_size=1,
    )
    target = torch.randn(1, 6, cycles, 1000)
    supports = torch.randn(1, 2, 6, cycles, 1000)
    if cycles == 20:
        target[:, :, 10] = 0
        supports[:, :, :, 10] = 0
    label = torch.randn(1)
    support_labels = torch.randn(1, 2)

    model.eval()
    with torch.no_grad():
        cycle_tokens, capacity_tokens, valid = model.cell_encoder(target)
        assert cycle_tokens.shape == (1, cycles, 64)
        assert capacity_tokens.shape == (1, 25, 64)
        assert valid.shape == (1, cycles)
        if cycles == 20:
            assert not valid[0, 10]
            assert torch.equal(
                cycle_tokens[:, 10], torch.zeros_like(cycle_tokens[:, 10]))
        scaled = model.cell_encoder(2 * target)
        relative_change = (
            (cycle_tokens - scaled[0]).norm() / cycle_tokens.norm())
        assert relative_change > 1e-3

    model.train()
    loss = model(
        target, label, supports, support_labels, return_loss=True)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients)

    model.eval()
    with torch.no_grad():
        prediction = model(target, label, supports, support_labels)
        reordered = model(
            target, label, supports.flip(1), support_labels.flip(1))
        details = model.compute_prediction_components(
            target, supports, support_labels, return_features=True)
    assert prediction.shape == (1,) and torch.isfinite(prediction).all()
    assert torch.allclose(prediction, reordered, atol=1e-5)
    assert details[-2].shape == (1, cycles + 25, 64)
    assert details[-1].shape == (1, 2, cycles + 25, 64)
    print(
        f'H={cycles}: scale_change={relative_change.item():.6g}, '
        f'loss={loss.item():.6g}'
    )


if __name__ == '__main__':
    check_task(20)
    check_task(100)
