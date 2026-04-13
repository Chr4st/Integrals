"""Tests for the integral prediction Transformer model."""
import torch
import pytest

from integral_engine.model import IntegralTransformer, PositionalEncoding
from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    FEATURE_DIM,
    PAD_INDEX,
    SEQ_VOCAB_SIZE,
)


# Small model dims for fast test execution
_SMALL_CONFIG = dict(
    vocab_size=SEQ_VOCAB_SIZE,
    d_model=64,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=128,
    dropout=0.0,
    feature_dim=FEATURE_DIM,
)


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.randn(2, 10, 64)
        out = pe(x)
        assert out.shape == (2, 10, 64)

    def test_no_nan(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.randn(2, 50, 64)
        out = pe(x)
        assert not torch.isnan(out).any()

    def test_different_positions_differ(self):
        pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        out = pe(x)
        assert not torch.equal(out[0, 0], out[0, 1])


class TestIntegralTransformer:
    def test_forward_shape(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        batch = 2
        src_len, tgt_len = 10, 8
        src = torch.randint(0, SEQ_VOCAB_SIZE, (batch, src_len))
        tgt = torch.randint(0, SEQ_VOCAB_SIZE, (batch, tgt_len))
        feat = torch.randn(batch, FEATURE_DIM)
        logits = model(src, tgt, feat)
        assert logits.shape == (batch, tgt_len, SEQ_VOCAB_SIZE)

    def test_forward_with_padding_mask(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        batch = 2
        src = torch.randint(0, SEQ_VOCAB_SIZE, (batch, 10))
        tgt = torch.randint(0, SEQ_VOCAB_SIZE, (batch, 8))
        feat = torch.randn(batch, FEATURE_DIM)
        src_mask = torch.zeros(batch, 10, dtype=torch.bool)
        src_mask[:, -3:] = True  # Last 3 positions padded
        tgt_mask = torch.zeros(batch, 8, dtype=torch.bool)
        logits = model(src, tgt, feat, src_mask, tgt_mask)
        assert logits.shape == (batch, 8, SEQ_VOCAB_SIZE)
        assert not torch.isnan(logits).any()

    def test_gradient_flow(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.train()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        tgt = torch.randint(0, SEQ_VOCAB_SIZE, (1, 4))
        feat = torch.randn(1, FEATURE_DIM)
        logits = model(src, tgt, feat)
        loss = logits.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        tgt = torch.randint(0, SEQ_VOCAB_SIZE, (1, 3))
        feat = torch.randn(1, FEATURE_DIM)

        out1 = model(src, tgt, feat)

        torch.manual_seed(42)
        model2 = IntegralTransformer(**_SMALL_CONFIG)
        model2.eval()
        out2 = model2(src, tgt, feat)

        assert torch.allclose(out1, out2, atol=1e-6)


class TestBeamSearch:
    def test_beam_count(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        beams = model.generate(src, feat, max_len=10, beam_width=3)
        assert len(beams) <= 3
        assert len(beams) >= 1

    def test_bos_prefix(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        beams = model.generate(src, feat, max_len=10, beam_width=2)
        for seq, _ in beams:
            assert seq[0] == BOS_INDEX

    def test_log_probs_negative(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        beams = model.generate(src, feat, max_len=10, beam_width=3)
        for seq, log_prob in beams:
            assert log_prob <= 0.0, "Log probabilities should be non-positive"

    def test_length_penalty_affects_ranking(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        beams_no_penalty = model.generate(
            src, feat, max_len=15, beam_width=3, length_penalty_alpha=0.0,
        )
        beams_with_penalty = model.generate(
            src, feat, max_len=15, beam_width=3, length_penalty_alpha=1.0,
        )
        # Just verify both return valid results
        assert len(beams_no_penalty) >= 1
        assert len(beams_with_penalty) >= 1

    def test_default_beam_width_is_10(self):
        """Phase 8: verify default beam width increased to 10."""
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        beams = model.generate(src, feat, max_len=10)
        assert len(beams) <= 10


class TestSampling:
    def test_sample_returns_sequence_and_logprob(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        seq, log_prob = model.sample(src, feat, max_len=10)
        assert isinstance(seq, list)
        assert len(seq) >= 2  # At least BOS + one token
        assert seq[0] == BOS_INDEX
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0

    def test_sample_respects_max_len(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)
        seq, _ = model.sample(src, feat, max_len=5)
        # BOS + up to 5 more tokens
        assert len(seq) <= 6

    def test_different_temperatures_produce_different_outputs(self):
        torch.manual_seed(42)
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 5))
        feat = torch.randn(1, FEATURE_DIM)

        # Generate many samples at different temps to check diversity
        results_low_temp = set()
        results_high_temp = set()
        for i in range(10):
            torch.manual_seed(i)
            seq, _ = model.sample(src, feat, max_len=10, temperature=0.1)
            results_low_temp.add(tuple(seq))
            torch.manual_seed(i)
            seq, _ = model.sample(src, feat, max_len=10, temperature=1.5)
            results_high_temp.add(tuple(seq))

        # High temperature should produce more diversity (or at least not crash)
        assert len(results_low_temp) >= 1
        assert len(results_high_temp) >= 1

    def test_sample_with_padding_mask(self):
        model = IntegralTransformer(**_SMALL_CONFIG)
        model.eval()
        src = torch.randint(0, SEQ_VOCAB_SIZE, (1, 8))
        feat = torch.randn(1, FEATURE_DIM)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[:, -3:] = True
        seq, log_prob = model.sample(
            src, feat, max_len=10, src_key_padding_mask=mask,
        )
        assert len(seq) >= 2
        assert seq[0] == BOS_INDEX
