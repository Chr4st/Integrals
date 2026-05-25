"""Tests for positional encoding modules."""

import pytest
import torch

from neurips.models.positional import (
    ALiBiSlopes,
    RotaryEmbedding,
    apply_rotary,
    build_pe,
)


class TestRotaryEmbedding:
    def test_shape(self):
        rope = RotaryEmbedding(dim=64, max_len=128)
        cos, sin = rope(seq_len=32)
        assert cos.shape == (1, 32, 64)
        assert sin.shape == (1, 32, 64)

    def test_deterministic(self):
        rope = RotaryEmbedding(dim=64, max_len=128)
        c1, s1 = rope(seq_len=32)
        c2, s2 = rope(seq_len=32)
        assert torch.allclose(c1, c2)
        assert torch.allclose(s1, s2)


class TestApplyRotary:
    def test_shape_preserved(self):
        B, H, S, D = 2, 8, 16, 64
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos = torch.randn(1, S, D)
        sin = torch.randn(1, S, D)
        q_r, k_r = apply_rotary(q, k, cos, sin)
        assert q_r.shape == (B, H, S, D)
        assert k_r.shape == (B, H, S, D)

    def test_zero_sin_is_identity(self):
        B, H, S, D = 1, 1, 4, 8
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        cos = torch.ones(1, S, D)
        sin = torch.zeros(1, S, D)
        q_r, k_r = apply_rotary(q, k, cos, sin)
        assert torch.allclose(q_r, q, atol=1e-6)
        assert torch.allclose(k_r, k, atol=1e-6)


class TestALiBiSlopes:
    def test_shape(self):
        alibi = ALiBiSlopes(n_heads=8)
        bias = alibi(seq_len=32)
        assert bias.shape == (8, 32, 32)

    def test_causal_bias_is_nonpositive(self):
        alibi = ALiBiSlopes(n_heads=4)
        bias = alibi(seq_len=16)
        assert (bias <= 0).all()

    def test_diagonal_is_zero(self):
        alibi = ALiBiSlopes(n_heads=4)
        bias = alibi(seq_len=16)
        for h in range(4):
            assert torch.allclose(bias[h].diag(), torch.zeros(16))


class TestBuildPE:
    @pytest.mark.parametrize("pe_type", ["sinusoidal", "rope", "alibi", "nope"])
    def test_build_all_types(self, pe_type):
        pe = build_pe(pe_type, d_model=64, max_len=128, dropout=0.1, n_heads=8)
        assert pe is not None

    def test_sinusoidal_output_shape(self):
        pe = build_pe("sinusoidal", d_model=64, max_len=128, dropout=0.0, n_heads=8)
        x = torch.randn(2, 32, 64)
        out = pe(x)
        assert out.shape == (2, 32, 64)

    def test_nope_is_identity(self):
        pe = build_pe("nope", d_model=64, max_len=128, dropout=0.0, n_heads=8)
        x = torch.randn(2, 32, 64)
        out = pe(x)
        assert torch.allclose(out, x)
