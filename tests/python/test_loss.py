"""Tests for equivalence-class loss functions (Sprint 1)."""

import pytest
import torch

from neurips.training.loss import equivalence_class_ce, standard_ce


class TestStandardCE:
    def test_shape(self):
        logits = torch.randn(2, 5, 10)
        targets = torch.randint(0, 10, (2, 5))
        loss = standard_ce(logits, targets)
        assert loss.shape == ()

    def test_perfect_prediction_low_loss(self):
        B, T, V = 2, 5, 10
        targets = torch.randint(0, V, (B, T))
        logits = torch.zeros(B, T, V)
        for b in range(B):
            for t in range(T):
                logits[b, t, targets[b, t]] = 100.0
        loss = standard_ce(logits, targets)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        logits = torch.randn(2, 5, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 5))
        loss = standard_ce(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestEquivalenceClassCE:
    def test_shape(self):
        B, T, V, K = 2, 5, 10, 3
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, K, T))
        mask = torch.ones(B, K, T, dtype=torch.bool)
        loss = equivalence_class_ce(logits, targets, mask)
        assert loss.shape == ()

    def test_min_over_k(self):
        """Loss should be the minimum across K equivalent targets."""
        B, T, V, K = 1, 3, 10, 2
        logits = torch.randn(B, T, V)

        # Target 0: random, Target 1: matches logits perfectly
        targets = torch.randint(0, V, (B, K, T))
        targets[0, 1, :] = logits[0].argmax(dim=-1)

        mask = torch.ones(B, K, T, dtype=torch.bool)
        eq_loss = equivalence_class_ce(logits, targets, mask)

        # Compare to single-target loss using the better target
        single_loss = standard_ce(logits, targets[:, 1, :])
        assert eq_loss.item() <= single_loss.item() + 0.01

    def test_mask_excludes_padded(self):
        B, T, V, K = 1, 5, 10, 3
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, K, T))
        mask = torch.ones(B, K, T, dtype=torch.bool)
        # Mask out target 1 entirely
        mask[0, 1, :] = False
        loss = equivalence_class_ce(logits, targets, mask)
        # Loss should still be finite (not NaN/inf)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        B, T, V, K = 2, 5, 10, 4
        logits = torch.randn(B, T, V, requires_grad=True)
        targets = torch.randint(0, V, (B, K, T))
        mask = torch.ones(B, K, T, dtype=torch.bool)
        loss = equivalence_class_ce(logits, targets, mask)
        loss.backward()
        assert logits.grad is not None

    def test_single_k_equals_standard(self):
        """With K=1, equivalence-class CE should equal standard CE."""
        B, T, V = 2, 5, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        mask = torch.ones(B, 1, T, dtype=torch.bool)
        eq_loss = equivalence_class_ce(
            logits, targets.unsqueeze(1), mask
        )
        std_loss = standard_ce(logits, targets)
        assert abs(eq_loss.item() - std_loss.item()) < 1e-4
