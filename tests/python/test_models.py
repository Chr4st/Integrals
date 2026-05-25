"""Tests for ML models (Phases 8, 10-11)."""
import pytest
import torch


class TestSeqTransformer:
    def test_import(self):
        from neurips.models.seq_transformer import SeqTransformer

    def test_forward_shape(self):
        from neurips.models.seq_transformer import SeqTransformer
        model = SeqTransformer(
            d_model=64, n_heads=4, n_layers=2, d_ff=256,
            dropout=0.1, vocab_size=256, max_seq_len=128,
        )
        batch = 4
        src = torch.randint(0, 256, (batch, 20))
        tgt = torch.randint(0, 256, (batch, 15))
        feats = torch.randn(batch, 344)
        logits = model(src, tgt, feats)
        assert logits.shape == (batch, 15, 256)

    def test_param_count_small(self):
        from neurips.models.seq_transformer import SeqTransformer
        model = SeqTransformer(
            d_model=64, n_heads=4, n_layers=2, d_ff=256,
            dropout=0.1, vocab_size=256, max_seq_len=128,
        )
        n = sum(p.numel() for p in model.parameters())
        assert n > 0


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
