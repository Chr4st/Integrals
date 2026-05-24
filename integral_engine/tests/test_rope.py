"""Tests for Rotary Position Embeddings (RoPE) module."""
import torch
import torch.nn as nn
import pytest

from integral_engine.rope import (
    RotaryEmbedding,
    RoPEDecoderLayer,
    RoPEEncoderLayer,
    RoPEMultiheadAttention,
    apply_rotary_pos_emb,
)

# Small dimensions for fast tests
D_MODEL = 64
NHEAD = 4
HEAD_DIM = D_MODEL // NHEAD
DIM_FF = 128
BATCH = 2
SEQ_LEN = 10
TGT_LEN = 8
MEM_LEN = 12


class TestRotaryEmbedding:
    """Test RotaryEmbedding output shapes and properties."""

    def test_output_shapes(self):
        rope = RotaryEmbedding(HEAD_DIM, max_len=512)
        cos, sin = rope(SEQ_LEN)
        assert cos.shape == (1, SEQ_LEN, HEAD_DIM)
        assert sin.shape == (1, SEQ_LEN, HEAD_DIM)

    def test_output_shapes_various_lengths(self):
        rope = RotaryEmbedding(HEAD_DIM, max_len=256)
        for length in [1, 5, 32, 100]:
            cos, sin = rope(length)
            assert cos.shape == (1, length, HEAD_DIM)
            assert sin.shape == (1, length, HEAD_DIM)

    def test_different_positions_produce_different_embeddings(self):
        rope = RotaryEmbedding(HEAD_DIM, max_len=512)
        cos, sin = rope(SEQ_LEN)
        # Position 0 and position 1 should have different cos values
        assert not torch.equal(cos[0, 0], cos[0, 1])
        assert not torch.equal(sin[0, 0], sin[0, 1])

    def test_position_zero_cos_is_one(self):
        """At position 0, cos(0 * theta) = 1 for all frequencies."""
        rope = RotaryEmbedding(HEAD_DIM, max_len=64)
        cos, sin = rope(4)
        assert torch.allclose(cos[0, 0], torch.ones(HEAD_DIM), atol=1e-6)
        assert torch.allclose(sin[0, 0], torch.zeros(HEAD_DIM), atol=1e-6)

    def test_cache_extends_for_longer_sequences(self):
        rope = RotaryEmbedding(HEAD_DIM, max_len=16)
        # Request longer than max_len -- should rebuild cache
        cos, sin = rope(32)
        assert cos.shape == (1, 32, HEAD_DIM)

    def test_even_dim_required(self):
        with pytest.raises(ValueError, match="d_model must be even"):
            RotaryEmbedding(d_model=7)

    def test_no_nan(self):
        rope = RotaryEmbedding(HEAD_DIM, max_len=512)
        cos, sin = rope(100)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_cos_sin_unit_circle_property(self):
        """cos^2 + sin^2 = 1 at every position and dimension."""
        rope = RotaryEmbedding(HEAD_DIM, max_len=512)
        cos, sin = rope(50)
        identity = cos ** 2 + sin ** 2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)


class TestApplyRotaryPosEmb:
    """Test the apply_rotary_pos_emb function."""

    def test_output_shapes(self):
        q = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        cos = torch.ones(1, 1, SEQ_LEN, HEAD_DIM)
        sin = torch.zeros(1, 1, SEQ_LEN, HEAD_DIM)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_identity_rotation(self):
        """cos=1, sin=0 should leave vectors unchanged."""
        q = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        cos = torch.ones(1, 1, SEQ_LEN, HEAD_DIM)
        sin = torch.zeros(1, 1, SEQ_LEN, HEAD_DIM)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_rot, q, atol=1e-6)
        assert torch.allclose(k_rot, k, atol=1e-6)

    def test_rotation_preserves_norm(self):
        """RoPE is an orthogonal rotation -- norms should be preserved."""
        torch.manual_seed(42)
        q = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH, NHEAD, SEQ_LEN, HEAD_DIM)
        rope = RotaryEmbedding(HEAD_DIM)
        cos, sin = rope(SEQ_LEN)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        q_norms = torch.norm(q, dim=-1)
        q_rot_norms = torch.norm(q_rot, dim=-1)
        assert torch.allclose(q_norms, q_rot_norms, atol=1e-4)

    def test_different_positions_give_different_rotations(self):
        """Same vector at different positions should produce different outputs."""
        vec = torch.randn(1, 1, 1, HEAD_DIM).expand(1, 1, 2, HEAD_DIM)
        rope = RotaryEmbedding(HEAD_DIM)
        cos, sin = rope(2)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot, _ = apply_rotary_pos_emb(vec, vec, cos, sin)
        # Position 0 and position 1 rotations should differ
        assert not torch.equal(q_rot[0, 0, 0], q_rot[0, 0, 1])


class TestRoPEMultiheadAttention:
    """Test RoPEMultiheadAttention module."""

    def test_output_shape(self):
        attn = RoPEMultiheadAttention(D_MODEL, NHEAD, dropout=0.0)
        q = torch.randn(BATCH, TGT_LEN, D_MODEL)
        k = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        v = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = attn(q, k, v)
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)

    def test_self_attention_shape(self):
        attn = RoPEMultiheadAttention(D_MODEL, NHEAD, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = attn(x, x, x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_with_attention_mask(self):
        attn = RoPEMultiheadAttention(D_MODEL, NHEAD, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        causal = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1
        )
        out = attn(x, x, x, attn_mask=causal)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert not torch.isnan(out).any()

    def test_with_key_padding_mask(self):
        attn = RoPEMultiheadAttention(D_MODEL, NHEAD, dropout=0.0)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        pad_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        pad_mask[:, -3:] = True
        out = attn(x, x, x, key_padding_mask=pad_mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert not torch.isnan(out).any()

    def test_d_model_not_divisible_by_nhead_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            RoPEMultiheadAttention(d_model=65, nhead=4)


class TestRoPEEncoderLayer:
    """Test RoPEEncoderLayer as drop-in for nn.TransformerEncoderLayer."""

    def test_forward_shape(self):
        layer = RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        src = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = layer(src)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_forward_with_masks(self):
        layer = RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        src = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        src_mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=1
        )
        pad_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.bool)
        pad_mask[:, -2:] = True
        out = layer(src, src_mask=src_mask, src_key_padding_mask=pad_mask)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        layer = RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        layer.train()
        src = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        out = layer(src)
        loss = out.sum()
        loss.backward()
        assert src.grad is not None
        assert src.grad.abs().sum() > 0
        # Also verify internal parameters got gradients
        has_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters()
        )
        assert has_param_grad

    def test_no_nan_output(self):
        layer = RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        layer.eval()
        src = torch.randn(BATCH, 20, D_MODEL)
        out = layer(src)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_output_differs_from_standard_encoder(self):
        """Confirm RoPE encoder produces different output than standard."""
        torch.manual_seed(99)
        rope_layer = RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)

        torch.manual_seed(99)
        std_layer = torch.nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FF,
            dropout=0.0, batch_first=True, norm_first=True,
        )

        src = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        rope_out = rope_layer(src)
        std_out = std_layer(src)

        # Different attention mechanisms should produce different outputs.
        # This confirms RoPE is not a no-op.
        assert not torch.allclose(rope_out, std_out, atol=1e-4)


class TestRoPEDecoderLayer:
    """Test RoPEDecoderLayer as drop-in for nn.TransformerDecoderLayer."""

    def test_forward_shape(self):
        layer = RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL)
        memory = torch.randn(BATCH, MEM_LEN, D_MODEL)
        out = layer(tgt, memory)
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)

    def test_forward_with_all_masks(self):
        layer = RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL)
        memory = torch.randn(BATCH, MEM_LEN, D_MODEL)
        tgt_mask = torch.triu(
            torch.ones(TGT_LEN, TGT_LEN, dtype=torch.bool), diagonal=1
        )
        tgt_pad = torch.zeros(BATCH, TGT_LEN, dtype=torch.bool)
        tgt_pad[:, -1:] = True
        mem_pad = torch.zeros(BATCH, MEM_LEN, dtype=torch.bool)
        mem_pad[:, -2:] = True
        out = layer(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad,
        )
        assert out.shape == (BATCH, TGT_LEN, D_MODEL)
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        layer = RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        layer.train()
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL, requires_grad=True)
        memory = torch.randn(BATCH, MEM_LEN, D_MODEL, requires_grad=True)
        out = layer(tgt, memory)
        loss = out.sum()
        loss.backward()
        # Gradient flows to both tgt and memory
        assert tgt.grad is not None and tgt.grad.abs().sum() > 0
        assert memory.grad is not None and memory.grad.abs().sum() > 0
        # Internal parameters also get gradients
        has_param_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in layer.parameters()
        )
        assert has_param_grad

    def test_no_nan_output(self):
        layer = RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        layer.eval()
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL)
        memory = torch.randn(BATCH, MEM_LEN, D_MODEL)
        out = layer(tgt, memory)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_causal_masking_is_causal(self):
        """Verify that with a causal mask, position i cannot attend to j > i."""
        layer = RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
        layer.eval()

        seq_len = 6
        tgt = torch.randn(1, seq_len, D_MODEL)
        memory = torch.randn(1, 4, D_MODEL)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )

        # Modify last position -- only output at last position should change
        tgt_modified = tgt.clone()
        tgt_modified[0, -1, :] += 10.0

        out_orig = layer(tgt, memory, tgt_mask=causal)
        out_mod = layer(tgt_modified, memory, tgt_mask=causal)

        # Positions 0..seq_len-2 should be identical (they can't see position -1)
        assert torch.allclose(
            out_orig[0, :seq_len - 1], out_mod[0, :seq_len - 1], atol=1e-5
        )
        # Position -1 should differ
        assert not torch.allclose(
            out_orig[0, -1], out_mod[0, -1], atol=1e-3
        )


class TestMultiLayerStack:
    """Test stacking multiple RoPE layers (integration sanity)."""

    def test_encoder_stack(self):
        layers = nn.ModuleList([
            RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
            for _ in range(3)
        ])
        src = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        x = src
        for layer in layers:
            x = layer(x)
        assert x.shape == (BATCH, SEQ_LEN, D_MODEL)
        assert not torch.isnan(x).any()

    def test_decoder_stack(self):
        layers = nn.ModuleList([
            RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
            for _ in range(3)
        ])
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL)
        memory = torch.randn(BATCH, MEM_LEN, D_MODEL)
        x = tgt
        for layer in layers:
            x = layer(x, memory)
        assert x.shape == (BATCH, TGT_LEN, D_MODEL)
        assert not torch.isnan(x).any()

    def test_full_stack_gradient_flow(self):
        """Gradients flow through a full encoder+decoder stack."""
        enc_layers = nn.ModuleList([
            RoPEEncoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
            for _ in range(2)
        ])
        dec_layers = nn.ModuleList([
            RoPEDecoderLayer(D_MODEL, NHEAD, DIM_FF, dropout=0.0)
            for _ in range(2)
        ])

        src = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
        tgt = torch.randn(BATCH, TGT_LEN, D_MODEL, requires_grad=True)

        memory = src
        for layer in enc_layers:
            memory = layer(memory)

        out = tgt
        for layer in dec_layers:
            out = layer(out, memory)

        loss = out.sum()
        loss.backward()

        assert src.grad is not None and src.grad.abs().sum() > 0
        assert tgt.grad is not None and tgt.grad.abs().sum() > 0
