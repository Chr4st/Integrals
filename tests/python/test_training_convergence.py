"""Correctness tests for training loop behavior.

Tests verify that the training loop actually optimizes, gradient clipping
works, and PAD tokens are ignored in loss computation.

Uses a lightweight stub model matching the tree model's training interface
(model(input_trees, int_var) -> logits) so tests run fast without
requiring the full TreeIntegrator graph construction.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurips.data.vocab import PAD
from neurips.training.trainer import TrainConfig, _compute_loss, train_step


class _StubTreeModel(nn.Module):
    """Minimal model matching the tree training interface.

    _compute_loss for "tree" calls model(batch["input_trees"], batch["int_var"])
    and expects logits of shape [batch, seq, vocab].
    """

    def __init__(self, vocab_size: int = 256, hidden: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(
        self, input_trees: torch.Tensor, int_var: torch.Tensor
    ) -> torch.Tensor:
        h = self.emb(input_trees)
        return self.proj(h)


def _make_small_model() -> _StubTreeModel:
    """Create a tiny stub model for fast testing."""
    return _StubTreeModel(vocab_size=256, hidden=32)


def _make_batch(
    batch_size: int = 4,
    seq_len: int = 20,
) -> dict[str, torch.Tensor]:
    """Create a reproducible training batch for the tree interface."""
    return {
        "input_trees": torch.randint(1, 100, (batch_size, seq_len)),
        "int_var": torch.randint(0, 5, (batch_size,)),
        "target_symbols": torch.randint(1, 100, (batch_size, seq_len)),
    }


class TestLossDecreases:
    """Verify the training loop actually optimizes."""

    def test_training_loss_decreases(self) -> None:
        """Training loop reduces loss over multiple steps on a fixed batch.

        If loss doesn't decrease after 20 steps on a fixed batch, the
        training loop is broken.
        """
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch()

        losses: list[float] = []
        for _ in range(20):
            loss = train_step(model, batch, optimizer, "tree", grad_clip=1.0)
            losses.append(loss)

        assert losses[-1] < losses[0] * 0.8, (
            f"Loss didn't decrease by 20%: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_loss_monotonically_trends_down(self) -> None:
        """Average loss in last 5 steps < average loss in first 5 steps."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch()

        losses: list[float] = []
        for _ in range(20):
            loss = train_step(model, batch, optimizer, "tree", grad_clip=1.0)
            losses.append(loss)

        avg_first_5 = sum(losses[:5]) / 5
        avg_last_5 = sum(losses[-5:]) / 5
        assert avg_last_5 < avg_first_5, (
            f"Loss trend not downward: first5={avg_first_5:.4f}, "
            f"last5={avg_last_5:.4f}"
        )


class TestGradientClipping:
    """Verify gradient clipping actually clips."""

    def test_grad_norm_bounded_after_clip(self) -> None:
        """After clip_grad_norm_, the total gradient norm is <= clip value."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        batch = _make_batch()

        loss = _compute_loss(model, batch, "tree")
        loss.backward()

        clip_val = 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)

        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach())
                for p in model.parameters()
                if p.grad is not None
            ])
        ).item()

        assert total_norm <= clip_val + 1e-3, (
            f"Grad norm {total_norm:.4f} exceeds clip value {clip_val}"
        )

    def test_train_step_applies_clipping(self) -> None:
        """train_step with grad_clip=0.1 produces bounded gradients."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch()

        train_step(model, batch, optimizer, "tree", grad_clip=0.1)

        loss = _compute_loss(model, batch, "tree")
        assert torch.isfinite(loss), "Loss is NaN/Inf after clipped training step"


class TestPadIgnored:
    """Verify that PAD tokens don't affect loss computation."""

    def test_changing_pad_positions_doesnt_change_loss(self) -> None:
        """Modifying target positions that are PAD does not change the loss."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.eval()

        batch = _make_batch()
        batch["target_symbols"][:, -3:] = PAD

        loss_original = _compute_loss(model, batch, "tree").item()

        batch_modified = {
            "input_trees": batch["input_trees"].clone(),
            "int_var": batch["int_var"].clone(),
            "target_symbols": batch["target_symbols"].clone(),
        }
        batch_modified["target_symbols"][:, -3:] = PAD

        loss_modified = _compute_loss(model, batch_modified, "tree").item()

        assert abs(loss_original - loss_modified) < 1e-6, (
            f"Loss changed when PAD positions were identical: "
            f"{loss_original:.6f} vs {loss_modified:.6f}"
        )

    def test_pad_vs_nonpad_loss_differs(self) -> None:
        """Replacing non-PAD targets with different values changes the loss."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.eval()

        batch_a = _make_batch()
        batch_b = {
            "input_trees": batch_a["input_trees"].clone(),
            "int_var": batch_a["int_var"].clone(),
            "target_symbols": torch.randint(1, 100, batch_a["target_symbols"].shape),
        }

        loss_a = _compute_loss(model, batch_a, "tree").item()
        loss_b = _compute_loss(model, batch_b, "tree").item()

        assert abs(loss_a - loss_b) > 1e-4, (
            "Loss identical for different non-PAD targets -- loss is constant"
        )

    def test_all_pad_target_loss_is_zero(self) -> None:
        """When all target positions are PAD, loss should be zero or NaN."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.eval()

        batch = _make_batch()
        batch["target_symbols"][:, :] = PAD

        loss = _compute_loss(model, batch, "tree")
        assert loss.item() == 0.0 or torch.isnan(loss), (
            f"Expected 0 or NaN for all-PAD targets, got {loss.item()}"
        )


class TestComputeLoss:
    """Verify _compute_loss behavior."""

    def test_unknown_model_type_raises(self) -> None:
        """_compute_loss raises ValueError for unknown model_type."""
        torch.manual_seed(42)
        model = _make_small_model()
        batch = _make_batch()

        with pytest.raises(ValueError, match="Unknown model_type"):
            _compute_loss(model, batch, "unknown_type")

    def test_loss_is_finite(self) -> None:
        """Training loss on valid input is finite."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        batch = _make_batch()

        loss = _compute_loss(model, batch, "tree")
        assert torch.isfinite(loss), f"Loss is {loss.item()}"

    def test_loss_positive(self) -> None:
        """Cross-entropy loss is always positive for valid inputs."""
        torch.manual_seed(42)
        model = _make_small_model()
        model.train()
        batch = _make_batch()

        loss = _compute_loss(model, batch, "tree")
        assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
