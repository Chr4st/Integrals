"""Correctness tests for TreeDecoder.

Tests verify KV cache equivalence, tree structure validity,
determinism of inference, and training-mode output shapes.
"""
from __future__ import annotations

import pytest
import torch

from neurips.models.tree_decoder import TreeDecoder, TreeNode


class TestKVCacheEquivalence:
    """Verify that cached and uncached attention produce identical results."""

    def test_kv_cache_equivalence(self) -> None:
        """Cross-attention with cached KV produces identical output to uncached.

        The KV cache is an optimization for inference speed. If it changes
        results, decoded trees will differ between cached/uncached paths,
        making inference non-deterministic in a way that's hard to debug.
        """
        torch.manual_seed(42)
        d_model = 64
        decoder = TreeDecoder(d_model=d_model, n_heads=4, n_layers=2, vocab_size=64)
        decoder.eval()

        memory = torch.randn(1, 10, d_model)
        query = torch.randn(1, 3, d_model)

        with torch.no_grad():
            out_no_cache = decoder._attend(query, memory)
            cache = decoder._build_kv_cache(memory)
            out_cached = decoder._attend(query, memory, cached_kv=cache)

        assert torch.allclose(out_no_cache, out_cached, atol=1e-5), (
            f"Max diff: {(out_no_cache - out_cached).abs().max().item()}"
        )

    def test_kv_cache_different_query_lengths(self) -> None:
        """KV cache works correctly for various query lengths.

        The cache stores memory projections which are query-independent.
        Varying query length must not break the cache.
        """
        torch.manual_seed(42)
        d_model = 64
        decoder = TreeDecoder(d_model=d_model, n_heads=4, n_layers=2, vocab_size=64)
        decoder.eval()

        memory = torch.randn(1, 8, d_model)
        cache = decoder._build_kv_cache(memory)

        for q_len in [1, 5, 12]:
            query = torch.randn(1, q_len, d_model)
            with torch.no_grad():
                out_no_cache = decoder._attend(query, memory)
                out_cached = decoder._attend(query, memory, cached_kv=cache)
            assert torch.allclose(out_no_cache, out_cached, atol=1e-5), (
                f"Cache diverged for query length {q_len}"
            )


class TestDecodeTreeStructure:
    """Verify that decode_tree produces valid tree structures."""

    @pytest.fixture()
    def decoder(self) -> TreeDecoder:
        torch.manual_seed(42)
        return TreeDecoder(d_model=64, n_heads=4, n_layers=2, vocab_size=256)

    def test_decode_tree_valid_structure(self, decoder: TreeDecoder) -> None:
        """Every TreeNode has valid arity, parent_idx, and level.

        Invalid tree structure (e.g., parent_idx pointing beyond the node
        list, or negative arity) would cause crashes in downstream
        tree-to-expression conversion.
        """
        torch.manual_seed(42)
        decoder.eval()
        encoder_out = torch.randn(1, 6, 64)

        with torch.no_grad():
            nodes = decoder.decode_tree(encoder_out, max_levels=4)

        assert len(nodes) > 0, "decode_tree produced no nodes"

        for i, node in enumerate(nodes):
            assert node.arity >= 0, f"Node {i} has negative arity {node.arity}"

            if i == 0:
                assert node.parent_idx is None, "Root node must have parent_idx=None"
            else:
                assert node.parent_idx is not None, (
                    f"Non-root node {i} has parent_idx=None"
                )
                assert 0 <= node.parent_idx < i, (
                    f"Node {i} has invalid parent_idx={node.parent_idx} "
                    f"(must be in [0, {i}))"
                )

    def test_children_count_matches_arity(self, decoder: TreeDecoder) -> None:
        """The number of children pointing to a parent matches that parent's arity.

        If this invariant breaks, the tree is structurally inconsistent:
        a node claims arity=2 but has 3 children, or vice versa.

        NOTE: Only fully-expanded parents (not truncated by max_levels)
        are checked, since frontier nodes may have unexpanded children.
        """
        torch.manual_seed(42)
        decoder.eval()
        encoder_out = torch.randn(1, 8, 64)

        with torch.no_grad():
            nodes = decoder.decode_tree(encoder_out, max_levels=5)

        if len(nodes) == 0:
            pytest.skip("No nodes decoded")

        # Count children for each node.
        child_count: dict[int, int] = {}
        for i, node in enumerate(nodes):
            if node.parent_idx is not None:
                child_count[node.parent_idx] = child_count.get(node.parent_idx, 0) + 1

        # Collect all parent indices that appear — these are "expanded" nodes.
        # Parents whose children were NOT generated (frontier at max_levels)
        # may have fewer children than arity — only check fully expanded ones.
        max_level = max(n.level for n in nodes) if nodes else 0
        for idx, node in enumerate(nodes):
            actual_children = child_count.get(idx, 0)
            # Only check if this node has at least some children generated
            # AND is not at the deepest level (where children would be cut off).
            if actual_children > 0 and node.level < max_level - 1:
                assert actual_children == node.arity, (
                    f"Node {idx} (symbol={node.symbol_id}, level={node.level}) "
                    f"has arity={node.arity} but {actual_children} children"
                )

    def test_level_consistent_with_parent(self, decoder: TreeDecoder) -> None:
        """Each node's level is strictly greater than its parent's level.

        Level monotonicity is required for breadth-first ordering.
        A child at the same or lower level than its parent indicates
        a bug in the level-tracking logic.
        """
        torch.manual_seed(42)
        decoder.eval()
        encoder_out = torch.randn(1, 6, 64)

        with torch.no_grad():
            nodes = decoder.decode_tree(encoder_out, max_levels=4)

        for i, node in enumerate(nodes):
            if node.parent_idx is not None:
                parent = nodes[node.parent_idx]
                assert node.level > parent.level, (
                    f"Node {i} level={node.level} not > parent {node.parent_idx} "
                    f"level={parent.level}"
                )


class TestDecodeTreeDeterminism:
    """Verify inference determinism."""

    def test_same_input_same_output(self) -> None:
        """decode_tree is deterministic: same input -> same output.

        Since decode_tree uses @torch.no_grad() and the model is in eval mode
        (no dropout), identical inputs must produce identical trees.
        Non-determinism here would make evaluation unreproducible.
        """
        torch.manual_seed(42)
        decoder = TreeDecoder(d_model=64, n_heads=4, n_layers=2, vocab_size=256)
        decoder.eval()

        encoder_out = torch.randn(1, 8, 64)

        with torch.no_grad():
            nodes_a = decoder.decode_tree(encoder_out, max_levels=4)
            nodes_b = decoder.decode_tree(encoder_out, max_levels=4)

        assert len(nodes_a) == len(nodes_b), (
            f"Different node counts: {len(nodes_a)} vs {len(nodes_b)}"
        )
        for i, (a, b) in enumerate(zip(nodes_a, nodes_b)):
            assert a.symbol_id == b.symbol_id, f"Node {i} symbol mismatch"
            assert a.arity == b.arity, f"Node {i} arity mismatch"
            assert a.level == b.level, f"Node {i} level mismatch"
            assert a.parent_idx == b.parent_idx, f"Node {i} parent_idx mismatch"


class TestForwardTrainingMode:
    """Verify training-mode forward pass output shapes."""

    def test_forward_logit_shapes(self) -> None:
        """Given [batch, n_nodes, d_model] input, output is [batch, n_nodes, vocab_size].

        Wrong output shape causes silent shape mismatches in loss computation
        that either crash or produce meaningless gradients.
        """
        torch.manual_seed(42)
        vocab_size = 128
        d_model = 64
        decoder = TreeDecoder(
            d_model=d_model, n_heads=4, n_layers=2, vocab_size=vocab_size
        )

        batch, n_nodes = 4, 12
        node_query = torch.randn(batch, n_nodes, d_model)
        encoder_out = torch.randn(batch, 8, d_model)

        logits = decoder(node_query, encoder_out)
        assert logits.shape == (batch, n_nodes, vocab_size), (
            f"Expected ({batch}, {n_nodes}, {vocab_size}), got {logits.shape}"
        )

    def test_forward_produces_gradients(self) -> None:
        """Training forward pass produces non-zero gradients.

        If gradients are all zero, the model cannot learn — this catches
        dead layers, detached tensors, or broken computation graphs.
        """
        torch.manual_seed(42)
        d_model = 64
        decoder = TreeDecoder(d_model=d_model, n_heads=4, n_layers=2, vocab_size=64)

        query = torch.randn(2, 5, d_model)
        memory = torch.randn(2, 8, d_model)

        logits = decoder(query, memory)
        loss = logits.sum()
        loss.backward()

        has_nonzero_grad = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in decoder.parameters()
        )
        assert has_nonzero_grad, "All gradients are zero — model cannot learn"
