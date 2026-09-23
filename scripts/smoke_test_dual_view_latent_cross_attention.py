"""Run data-free checks for the dual-view latent cross-attention model."""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.rul_predictors.dual_view_latent_cross_attention_batlinet import (
    DualViewLatentCrossAttentionBatLiNetRULPredictor,
)


def main():
    torch.manual_seed(0)
    batch_size = 2
    support_size = 2
    model = DualViewLatentCrossAttentionBatLiNetRULPredictor(
        in_channels=6,
        channels=32,
        input_height=20,
        input_width=1000,
        train_support_size=support_size,
        test_support_size=32,
        attention_channels=64,
        attention_heads=4,
        attention_layers=1,
        attention_mlp_ratio=2,
        head_hidden_channels=64,
        capacity_bins=32,
        stem_channels=32,
        cycle_encoder_layers=1,
        epochs=1,
        train_batch_size=batch_size,
        test_batch_size=1,
    )

    feature = torch.randn(batch_size, 6, 20, 1000)
    support_feature = torch.randn(
        batch_size, support_size, 6, 20, 1000)
    feature[:, :, 10] = 0
    support_feature[:, :, :, 10] = 0
    label = torch.randn(batch_size)
    support_label = torch.randn(batch_size, support_size)

    cycle, capacity, residual, mask = model.cell_encoder(feature)
    assert cycle.shape == (batch_size, 20, 64)
    assert capacity.shape == (batch_size, 32, 64)
    assert residual.shape == (batch_size, 128)
    assert mask.shape == (batch_size, 20)
    assert not mask[:, 10].any()
    assert torch.equal(cycle[:, 10], torch.zeros_like(cycle[:, 10]))

    model.train()
    loss = model(
        feature,
        label,
        support_feature,
        support_label,
        return_loss=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients and all(gradients)
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert not missing_gradients, missing_gradients

    model.eval()
    with torch.no_grad():
        prediction = model(
            feature,
            label,
            support_feature,
            support_label,
        )
        permuted_prediction = model(
            feature,
            label,
            support_feature.flip(1),
            support_label.flip(1),
        )
        components = model.compute_prediction_components(
            feature,
            support_feature,
            support_label,
            return_features=True,
        )
    assert prediction.shape == (batch_size,)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction, permuted_prediction, atol=1e-5)
    assert components[-2].shape == (batch_size, 52, 64)
    assert components[-1].shape == (
        batch_size, support_size, 52, 64)

    print(
        'Dual-view latent cross-attention smoke test passed: '
        f'loss={loss.item():.6f}, '
        f'parameters={sum(p.numel() for p in model.parameters()):,}'
    )


if __name__ == '__main__':
    main()
