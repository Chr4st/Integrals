"""Tests for contrastive pre-training (Spec 6) and method-routing classifier (Spec 9)."""
import numpy as np
import sympy as sp
import torch
import pytest

from integral_engine.contrastive import (
    ContrastiveModel,
    NumericEncoder,
    evaluate_at_points,
    info_nce_loss,
    transfer_encoder_weights,
)
from integral_engine.router import (
    IntegrationRouter,
    TECHNIQUE_CLASSES,
    classify_integrand,
)
from integral_engine.vocabulary import FEATURE_DIM, PAD_INDEX, SEQ_VOCAB_SIZE

x = sp.Symbol("x")

# Small dimensions for fast test execution
_SMALL_D_MODEL = 64
_SMALL_NHEAD = 4
_SMALL_LAYERS = 1
_SMALL_EVAL_POINTS = 64


# ---------------------------------------------------------------------------
# Spec 6: ContrastiveModel tests
# ---------------------------------------------------------------------------


class TestNumericEncoder:
    def test_output_shape(self):
        enc = NumericEncoder(n_points=64, hidden=128, d_model=_SMALL_D_MODEL)
        inp = torch.randn(4, 64)
        out = enc(inp)
        assert out.shape == (4, _SMALL_D_MODEL)

    def test_different_inputs_produce_different_outputs(self):
        enc = NumericEncoder(n_points=64, hidden=128, d_model=_SMALL_D_MODEL)
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)
        out_a = enc(a)
        out_b = enc(b)
        assert not torch.allclose(out_a, out_b, atol=1e-6)


class TestContrastiveModel:
    def _make_model(self):
        return ContrastiveModel(
            vocab_size=SEQ_VOCAB_SIZE,
            d_model=_SMALL_D_MODEL,
            nhead=_SMALL_NHEAD,
            num_layers=_SMALL_LAYERS,
            n_eval_points=_SMALL_EVAL_POINTS,
        )

    def test_forward_shapes(self):
        model = self._make_model()
        model.eval()
        batch, seq_len = 4, 10
        token_ids = torch.randint(0, SEQ_VOCAB_SIZE, (batch, seq_len))
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        numeric_values = torch.randn(batch, _SMALL_EVAL_POINTS)

        sym_emb, num_emb = model(token_ids, padding_mask, numeric_values)

        assert sym_emb.shape == (batch, _SMALL_D_MODEL)
        assert num_emb.shape == (batch, _SMALL_D_MODEL)

    def test_forward_no_mask(self):
        model = self._make_model()
        model.eval()
        batch, seq_len = 2, 8
        token_ids = torch.randint(0, SEQ_VOCAB_SIZE, (batch, seq_len))
        numeric_values = torch.randn(batch, _SMALL_EVAL_POINTS)

        sym_emb, num_emb = model(token_ids, None, numeric_values)

        assert sym_emb.shape == (batch, _SMALL_D_MODEL)
        assert num_emb.shape == (batch, _SMALL_D_MODEL)
        assert not torch.isnan(sym_emb).any()
        assert not torch.isnan(num_emb).any()

    def test_forward_with_padding(self):
        model = self._make_model()
        model.eval()
        batch, seq_len = 3, 12
        token_ids = torch.randint(0, SEQ_VOCAB_SIZE, (batch, seq_len))
        padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
        padding_mask[:, -4:] = True  # last 4 positions padded
        numeric_values = torch.randn(batch, _SMALL_EVAL_POINTS)

        sym_emb, num_emb = model(token_ids, padding_mask, numeric_values)

        assert not torch.isnan(sym_emb).any()
        assert not torch.isnan(num_emb).any()


class TestInfoNCELoss:
    def test_loss_is_finite(self):
        sym_emb = torch.randn(8, _SMALL_D_MODEL)
        num_emb = torch.randn(8, _SMALL_D_MODEL)
        loss = info_nce_loss(sym_emb, num_emb)
        assert torch.isfinite(loss)

    def test_loss_is_differentiable(self):
        sym_emb = torch.randn(8, _SMALL_D_MODEL, requires_grad=True)
        num_emb = torch.randn(8, _SMALL_D_MODEL, requires_grad=True)
        loss = info_nce_loss(sym_emb, num_emb)
        loss.backward()
        assert sym_emb.grad is not None
        assert num_emb.grad is not None
        assert sym_emb.grad.abs().sum() > 0
        assert num_emb.grad.abs().sum() > 0

    def test_matching_pairs_lower_loss(self):
        """Matching (identical) pairs should yield lower loss than random pairs."""
        torch.manual_seed(42)
        shared = torch.randn(8, _SMALL_D_MODEL)
        # Matching: sym_emb == num_emb (perfect alignment)
        loss_matched = info_nce_loss(shared, shared)
        # Random: unrelated embeddings
        loss_random = info_nce_loss(torch.randn(8, _SMALL_D_MODEL), torch.randn(8, _SMALL_D_MODEL))
        assert loss_matched < loss_random

    def test_temperature_scaling(self):
        """Lower temperature should produce higher loss for random pairs."""
        sym_emb = torch.randn(8, _SMALL_D_MODEL)
        num_emb = torch.randn(8, _SMALL_D_MODEL)
        loss_low_temp = info_nce_loss(sym_emb, num_emb, temperature=0.01)
        loss_high_temp = info_nce_loss(sym_emb, num_emb, temperature=1.0)
        # With random embeddings and low temperature, logits are more extreme
        assert loss_low_temp > loss_high_temp


class TestEvaluateAtPoints:
    def test_sin_x_values(self):
        points = np.linspace(-5, 5, 64)
        values = evaluate_at_points(sp.sin(x), x, points)
        expected = np.sin(points)
        np.testing.assert_allclose(values, expected, atol=1e-10)

    def test_default_points_length(self):
        values = evaluate_at_points(sp.sin(x), x)
        assert len(values) == 64

    def test_nan_handling(self):
        """1/x has a singularity at 0; NaN/Inf should be replaced with 0."""
        points = np.array([-1.0, 0.0, 1.0])
        values = evaluate_at_points(1 / x, x, points)
        assert np.all(np.isfinite(values))
        # At x=0, 1/x is Inf -> replaced with 0
        assert values[1] == 0.0

    def test_constant_expression(self):
        """Constant expression should broadcast to all points."""
        points = np.linspace(-5, 5, 10)
        values = evaluate_at_points(sp.Integer(5), x, points)
        assert len(values) == 10
        np.testing.assert_allclose(values, 5.0)

    def test_polynomial(self):
        points = np.linspace(-2, 2, 32)
        values = evaluate_at_points(x**2 + 3 * x + 1, x, points)
        expected = points**2 + 3 * points + 1
        np.testing.assert_allclose(values, expected, atol=1e-10)


class TestTransferEncoderWeights:
    def test_weight_transfer(self):
        from integral_engine.model import IntegralTransformer

        contrastive = ContrastiveModel(
            vocab_size=SEQ_VOCAB_SIZE,
            d_model=_SMALL_D_MODEL,
            nhead=_SMALL_NHEAD,
            num_layers=_SMALL_LAYERS,
            n_eval_points=_SMALL_EVAL_POINTS,
        )
        transformer = IntegralTransformer(
            vocab_size=SEQ_VOCAB_SIZE,
            d_model=_SMALL_D_MODEL,
            nhead=_SMALL_NHEAD,
            num_encoder_layers=_SMALL_LAYERS,
            num_decoder_layers=_SMALL_LAYERS,
            dim_feedforward=_SMALL_D_MODEL * 4,
            dropout=0.1,
            feature_dim=FEATURE_DIM,
        )

        # Weights should differ before transfer
        src_state = contrastive.symbolic_encoder.state_dict()
        tgt_state_before = {
            k: v.clone() for k, v in transformer.encoder.state_dict().items()
        }

        transfer_encoder_weights(contrastive, transformer)

        # After transfer, matching keys should be equal
        tgt_state_after = transformer.encoder.state_dict()
        for key in src_state:
            if key in tgt_state_after and src_state[key].shape == tgt_state_after[key].shape:
                assert torch.equal(src_state[key], tgt_state_after[key])


# ---------------------------------------------------------------------------
# Spec 9: IntegrationRouter tests
# ---------------------------------------------------------------------------


class TestIntegrationRouter:
    def test_output_shape(self):
        router = IntegrationRouter(feature_dim=FEATURE_DIM, hidden_dim=128, n_classes=4)
        features = torch.randn(5, FEATURE_DIM)
        probs = router(features)
        assert probs.shape == (5, 4)

    def test_probabilities_sum_to_one(self):
        router = IntegrationRouter()
        features = torch.randn(8, FEATURE_DIM)
        probs = router(features)
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=0)

    def test_classify_returns_indices(self):
        router = IntegrationRouter()
        features = torch.randn(6, FEATURE_DIM)
        classes = router.classify(features)
        assert classes.shape == (6,)
        assert classes.dtype in (torch.int64, torch.long)
        assert (classes >= 0).all()
        assert (classes < 4).all()

    def test_gradient_flow(self):
        router = IntegrationRouter()
        features = torch.randn(4, FEATURE_DIM, requires_grad=True)
        probs = router(features)
        # Use class 0 probability as loss (not sum, since softmax sums to 1 always)
        loss = probs[:, 0].sum()
        loss.backward()
        assert features.grad is not None
        assert features.grad.abs().sum() > 0

    def test_technique_classes_coverage(self):
        """All four technique classes should be present in the mapping."""
        assert len(TECHNIQUE_CLASSES) == 4
        assert set(TECHNIQUE_CLASSES.keys()) == {0, 1, 2, 3}


class TestClassifyIntegrand:
    def test_trig_expression(self):
        """sin(x) should be classified as trigonometric (class 1)."""
        assert classify_integrand(sp.sin(x), x) == 1

    def test_trig_hyperbolic(self):
        assert classify_integrand(sp.cosh(x), x) == 1

    def test_polynomial(self):
        """x^2 + 3*x should be classified as polynomial/rational (class 0)."""
        assert classify_integrand(x**2 + 3 * x, x) == 0

    def test_constant(self):
        assert classify_integrand(sp.Integer(7), x) == 0

    def test_exp_expression(self):
        assert classify_integrand(sp.exp(x), x) == 2

    def test_log_expression(self):
        assert classify_integrand(sp.log(x), x) == 2

    def test_special_function(self):
        assert classify_integrand(sp.erf(x), x) == 3

    def test_bessel_is_special(self):
        assert classify_integrand(sp.besselj(0, x), x) == 3

    def test_trig_priority_over_exp(self):
        """Mixed trig+exp should still classify as trig (priority ordering)."""
        expr = sp.sin(x) * sp.exp(x)
        assert classify_integrand(expr, x) == 1

    def test_nested_trig(self):
        """Trig inside other operations should still be detected."""
        expr = x**2 + sp.sin(x)
        assert classify_integrand(expr, x) == 1
