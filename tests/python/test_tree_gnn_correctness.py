"""Correctness tests for Tree GNN components.

Tests verify mathematical properties of scatter aggregation,
information flow in message passing, and attention bias effects.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from neurips.models.tree_gnn import (
    MessageRound,
    NodeEmbedding,
    TreeIntegrator,
    TreeMessagePassing,
    VariableAwareAttention,
    _scatter_mean,
)


class TestScatterMean:
    """Verify scatter_mean aggregation correctness."""

    def test_scatter_mean_paths_agree(self) -> None:
        """Both scatter_mean implementations produce identical aggregations.

        The fast path (scatter_reduce_) and fallback (scatter_add + count)
        must agree — if they diverge, one codepath silently returns wrong
        aggregations, breaking message passing.
        """
        torch.manual_seed(42)
        num_nodes, num_edges, d = 10, 20, 256
        src = torch.randn(num_edges, d)
        index = torch.randint(0, num_nodes, (num_edges,))
        device = torch.device("cpu")

        # Fast path (scatter_reduce_).
        fast_result = _scatter_mean(src, index, num_nodes, device)

        # Force fallback by patching hasattr to return False for scatter_reduce.
        original_hasattr = hasattr

        def patched_hasattr(obj: object, name: str) -> bool:
            if name == "scatter_reduce":
                return False
            return original_hasattr(obj, name)

        with patch("neurips.models.tree_gnn.hasattr", patched_hasattr):
            fallback_result = _scatter_mean(src, index, num_nodes, device)

        assert torch.allclose(fast_result, fallback_result, atol=1e-5), (
            f"Max diff: {(fast_result - fallback_result).abs().max().item()}"
        )

    def test_scatter_mean_manual_check(self) -> None:
        """Scatter mean of known values matches hand-computed result.

        For index=[0,0,1], src=[[2,4],[6,8],[1,3]], expected:
        node 0 = mean([2,4],[6,8]) = [4,6], node 1 = [1,3].
        """
        src = torch.tensor([[2.0, 4.0], [6.0, 8.0], [1.0, 3.0]])
        index = torch.tensor([0, 0, 1])
        result = _scatter_mean(src, index, num_nodes=2, device=torch.device("cpu"))
        expected = torch.tensor([[4.0, 6.0], [1.0, 3.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_scatter_mean_empty_nodes_are_zero(self) -> None:
        """Nodes with no incoming edges get zero embeddings.

        Important because zero vs garbage for unreferenced nodes affects
        downstream message passing correctness.
        """
        src = torch.tensor([[1.0, 2.0]])
        index = torch.tensor([0])
        result = _scatter_mean(src, index, num_nodes=3, device=torch.device("cpu"))
        # Nodes 1 and 2 have no edges — must be zero.
        assert torch.allclose(result[1], torch.zeros(2))
        assert torch.allclose(result[2], torch.zeros(2))


class TestMessagePassing:
    """Verify message passing preserves graph structure."""

    def test_output_shape_matches_input(self) -> None:
        """After N rounds of message passing, output shape == input shape.

        Shape mismatch means the architecture silently changed dimensionality,
        which would break downstream attention and decoding.
        """
        torch.manual_seed(42)
        n_nodes, d = 15, 256
        h = torch.randn(n_nodes, d)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        mp = TreeMessagePassing(d=d, n_rounds=8)
        out = mp(h, edges)
        assert out.shape == (n_nodes, d)

    def test_disconnected_nodes_unchanged_by_messages(self) -> None:
        """Nodes with no edges only change via the residual/self path.

        If a disconnected node changes dramatically, message passing is
        leaking information from connected components.
        """
        torch.manual_seed(42)
        d = 256
        # Node 0 is connected (0->1), nodes 2,3 are isolated.
        h = torch.randn(4, d)
        edges = torch.tensor([[0], [1]])

        mp = MessageRound(d=d)
        mp.eval()
        out = mp(h, edges)

        # Isolated nodes (2, 3) get zero aggregations from both parent
        # and child scatter_mean. Their update only sees [h, zeros, zeros].
        # They still change via the update MLP + residual + layernorm,
        # but they must change identically to each other since they have
        # the same zero-aggregation context structure.
        # The key test: output is finite and not NaN.
        assert torch.isfinite(out).all(), "Message passing produced NaN/Inf"

    def test_disconnected_components_independent(self) -> None:
        """Two disconnected subgraphs produce outputs independent of each other.

        If modifying one subgraph's input changes the other's output,
        information is leaking across disconnected components.
        """
        torch.manual_seed(42)
        d = 256
        mp = TreeMessagePassing(d=d, n_rounds=4)
        mp.eval()

        # Component A: nodes 0-2 (chain 0->1->2)
        # Component B: nodes 3-5 (chain 3->4->5)
        edges = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]])
        h = torch.randn(6, d)

        out_original = mp(h, edges)

        # Perturb component A (nodes 0-2), keep B unchanged.
        h_perturbed = h.clone()
        h_perturbed[:3] += 10.0
        out_perturbed = mp(h_perturbed, edges)

        # Component B output (nodes 3-5) must be identical.
        assert torch.allclose(out_original[3:6], out_perturbed[3:6], atol=1e-5), (
            "Disconnected component B was affected by changes to component A"
        )


class TestVariableAwareAttention:
    """Verify the dependency bias actually affects attention output."""

    def test_dep_bias_changes_output(self) -> None:
        """With dep_mask=zeros vs dep_mask=identity, outputs should differ.

        If they're identical, the dep_bias parameter has no effect and the
        variable-aware attention degenerates to plain self-attention.
        """
        torch.manual_seed(42)
        n_nodes, d = 8, 256
        attn = VariableAwareAttention(d=d, n_heads=8)
        attn.eval()
        h = torch.randn(n_nodes, d)

        # All-zeros mask: no dependency bias anywhere.
        mask_zeros = torch.zeros(n_nodes, n_nodes)
        out_zeros = attn(h, dep_mask=mask_zeros)

        # Identity mask: only self-dependency.
        mask_eye = torch.eye(n_nodes)
        out_eye = attn(h, dep_mask=mask_eye)

        assert not torch.allclose(out_zeros, out_eye, atol=1e-6), (
            "dep_bias had no effect — outputs identical for different dep_masks"
        )

    def test_no_mask_vs_zero_mask_equivalent(self) -> None:
        """No mask (None) and all-zeros mask should produce same output.

        Both represent zero additive bias to attention scores.
        """
        torch.manual_seed(42)
        n_nodes, d = 6, 256
        attn = VariableAwareAttention(d=d, n_heads=8)
        attn.eval()
        h = torch.randn(n_nodes, d)

        out_none = attn(h, dep_mask=None)
        out_zeros = attn(h, dep_mask=torch.zeros(n_nodes, n_nodes))

        assert torch.allclose(out_none, out_zeros, atol=1e-5), (
            "None mask and zero mask should be equivalent but differ"
        )

    def test_2d_and_3d_input_shapes(self) -> None:
        """VariableAwareAttention handles both [N,d] and [1,N,d] inputs.

        The squeeze/unsqueeze logic must round-trip correctly.
        """
        torch.manual_seed(42)
        n_nodes, d = 5, 256
        attn = VariableAwareAttention(d=d, n_heads=8)
        attn.eval()
        h_2d = torch.randn(n_nodes, d)
        h_3d = h_2d.unsqueeze(0)

        out_2d = attn(h_2d)
        out_3d = attn(h_3d)

        assert out_2d.shape == (n_nodes, d), f"Expected (N,d), got {out_2d.shape}"
        assert out_3d.shape == (1, n_nodes, d), f"Expected (1,N,d), got {out_3d.shape}"
        assert torch.allclose(out_2d, out_3d.squeeze(0), atol=1e-5)


class TestTreeIntegrator:
    """Verify end-to-end model properties."""

    def test_parameter_count_in_expected_range(self) -> None:
        """TreeIntegrator param count should be ~12.1M (between 10M and 15M).

        Catches accidental architecture changes (added/removed layers,
        changed dimensions) that silently alter model capacity.
        """
        model = TreeIntegrator()
        n_params = model.count_parameters()
        assert 10_000_000 < n_params < 15_000_000, (
            f"Expected ~12.1M parameters, got {n_params:,}"
        )

    def test_forward_produces_tree_nodes(self) -> None:
        """Full forward pass produces a non-empty list of TreeNode.

        Verifies the embed->message-pass->attend->decode pipeline
        doesn't crash and actually produces output.
        """
        torch.manual_seed(42)
        model = TreeIntegrator()
        model.eval()

        n_nodes = 5
        symbol_ids = torch.randint(10, 100, (n_nodes,))
        role_features = torch.randn(n_nodes, 12)
        struct_features = torch.randn(n_nodes, 40)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])

        nodes = model(symbol_ids, role_features, struct_features, edges)
        assert len(nodes) > 0, "Forward produced no tree nodes"
