"""Run a data-free smoke test for the axis-aware multiscale model."""

import torch

from src.models.rul_predictors.axis_aware_multiscale_batlinet import (
    AxisAwareMultiScaleLatentCrossAttentionBatLiNetRULPredictor,
)


def main():
    torch.manual_seed(0)
    batch_size = 2
    support_size = 2
    model = AxisAwareMultiScaleLatentCrossAttentionBatLiNetRULPredictor(
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
        branch_channels=16,
        capacity_kernel_sizes=(9, 33, 97),
        cycle_kernel_size=3,
        cycle_pool_size=4,
        capacity_pool_size=32,
        epochs=1,
        train_batch_size=batch_size,
        test_batch_size=1,
    )
    feature = torch.randn(batch_size, 6, 20, 1000)
    label = torch.randn(batch_size)
    support_feature = torch.randn(
        batch_size,
        support_size,
        6,
        20,
        1000,
    )
    support_label = torch.randn(batch_size, support_size)

    tokens = model.cell_encoder(feature)
    assert tokens.shape == (batch_size, 155, 64)

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
    finite_gradients = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert finite_gradients and all(finite_gradients)

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
        repeated_tokens = model.cell_encoder(feature.clone())
    assert prediction.shape == (batch_size,)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction, permuted_prediction, atol=1e-5)
    assert torch.equal(repeated_tokens, model.cell_encoder(feature))

    print(
        'Axis-aware multiscale smoke test passed: '
        f'loss={loss.item():.6f}, '
        f'tokens={tuple(tokens.shape)}, '
        f'parameters={sum(p.numel() for p in model.parameters()):,}'
    )


if __name__ == '__main__':
    main()
