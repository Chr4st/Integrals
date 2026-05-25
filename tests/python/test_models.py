"""Tests for ML models (Phases 10-11: Tree GNN)."""
import pytest
import torch


class TestTreeGNN:
    def test_node_embedding_shape(self):
        from neurips.models.tree_gnn import NodeEmbedding
        emb = NodeEmbedding()
        sym = torch.randint(0, 256, (10,))
        role = torch.randn(10, 12)
        struct = torch.randn(10, 40)
        out = emb(sym, role, struct)
        assert out.shape == (10, 256)

    def test_message_passing_shape(self):
        from neurips.models.tree_gnn import TreeMessagePassing
        mp = TreeMessagePassing(d=256, n_rounds=2)
        nodes = torch.randn(5, 256)
        # Simple chain: 0→1, 1→2, 2→3, 3→4
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        out = mp(nodes, edges)
        assert out.shape == (5, 256)


class TestTreeDecoder:
    def test_import(self):
        from neurips.models.tree_decoder import TreeDecoder

    def test_symbol_head_shape(self):
        from neurips.models.tree_decoder import TreeDecoder
        dec = TreeDecoder(d_model=256, n_layers=4, n_heads=4)
        assert dec.symbol_head.out_features == 256
