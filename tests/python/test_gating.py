"""Tests for soft gating module."""

import torch

from neurips.models.gating import GatedIntegrator, SoftGating


class TestSoftGating:
    def test_output_shape(self):
        gate = SoftGating(seq_dim=640, tree_dim=256)
        seq_feat = torch.randn(4, 640)
        tree_feat = torch.randn(4, 256)
        weights = gate(seq_feat, tree_feat)
        assert weights.shape == (4, 2)

    def test_weights_sum_to_one(self):
        gate = SoftGating(seq_dim=640, tree_dim=256)
        seq_feat = torch.randn(8, 640)
        tree_feat = torch.randn(8, 256)
        weights = gate(seq_feat, tree_feat)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_weights_are_positive(self):
        gate = SoftGating(seq_dim=640, tree_dim=256)
        seq_feat = torch.randn(4, 640)
        tree_feat = torch.randn(4, 256)
        weights = gate(seq_feat, tree_feat)
        assert (weights >= 0).all()


class TestGatedIntegrator:
    def test_output_shape(self):
        model = GatedIntegrator(seq_dim=640, tree_dim=256, vocab_size=256)
        seq_feat = torch.randn(4, 640)
        tree_feat = torch.randn(4, 256)
        logits = model(seq_feat, tree_feat)
        assert logits.shape == (4, 256)

    def test_gradient_flow(self):
        model = GatedIntegrator(seq_dim=64, tree_dim=32, vocab_size=10)
        seq_feat = torch.randn(2, 64, requires_grad=True)
        tree_feat = torch.randn(2, 32, requires_grad=True)
        logits = model(seq_feat, tree_feat)
        loss = logits.sum()
        loss.backward()
        assert seq_feat.grad is not None
        assert tree_feat.grad is not None
