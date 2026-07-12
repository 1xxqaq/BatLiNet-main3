"""Run a data-free smoke test for the gated residual latent model."""

import torch

from src.models.rul_predictors.latent_gated_residual_batlinet import (
    LatentGatedResidualBatLiNetRULPredictor,
)


def main():
    torch.manual_seed(0)
    batch_size = 2
    support_size = 4
    model = LatentGatedResidualBatLiNetRULPredictor(
        in_channels=6,
        channels=32,
        input_height=20,
        input_width=1000,
        train_support_size=support_size,
        test_support_size=32,
        attention_channels=32,
        attention_heads=4,
        attention_layers=1,
        attention_mlp_ratio=2,
        head_hidden_channels=48,
        robust_iterations=2,
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

    model.train()
    outputs = model._compute_all(
        feature,
        support_feature,
        support_label,
    )
    loss = model(
        feature,
        label,
        support_feature,
        support_label,
        return_loss=True,
    )
    loss.backward()

    assert outputs['prediction'].shape == (batch_size,)
    assert outputs['y_sup'].shape == (batch_size, support_size)
    assert outputs['support_gate'].shape == (batch_size, support_size)
    assert outputs['support_weight'].shape == (batch_size, support_size)
    assert torch.allclose(
        outputs['support_weight'].sum(dim=1),
        torch.ones(batch_size),
        atol=1e-5,
    )
    assert torch.all(outputs['support_gate'] >= 0.0)
    assert torch.all(outputs['support_gate'] <= 1.0)
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
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_prediction = model(
            feature,
            label,
            support_feature[:, permutation],
            support_label[:, permutation],
        )
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction, permuted_prediction, atol=1e-5)
    print(
        'Gated residual latent smoke test passed: '
        f'loss={loss.item():.6f}, '
        f'parameters={sum(p.numel() for p in model.parameters()):,}'
    )


if __name__ == '__main__':
    main()
