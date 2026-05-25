"""Tests for the structural feature extractor (Phase 7b)."""
import numpy as np
import pytest
from neurips.data.features import FEATURE_DIM, extract_features


class TestFeatureShape:
    def test_output_shape(self):
        feats = extract_features("sin x", int_var="x")
        assert feats.shape == (FEATURE_DIM,)

    def test_output_dtype(self):
        feats = extract_features("add x x", int_var="x")
        assert feats.dtype == np.float32 or feats.dtype == np.float64

    def test_no_nan(self):
        for prefix in ["x", "sin x", "add mul x x x", "exp neg pow x 2"]:
            feats = extract_features(prefix, int_var="x")
            assert not np.any(np.isnan(feats))

    def test_no_inf(self):
        for prefix in ["x", "sin x", "add mul x x x"]:
            feats = extract_features(prefix, int_var="x")
            assert not np.any(np.isinf(feats))


class TestFeatureContent:
    def test_values_bounded(self):
        feats = extract_features("sin x", int_var="x")
        assert np.all(feats >= 0.0)
        assert np.all(feats <= 1.0 + 1e-6)  # allow small float error

    def test_single_var(self):
        feats = extract_features("x", int_var="x")
        assert feats.shape == (FEATURE_DIM,)

    def test_complex_expression(self):
        feats = extract_features("add mul sin x cos y pow x 2", int_var="x")
        assert feats.shape == (FEATURE_DIM,)

    def test_different_expressions_different_features(self):
        f1 = extract_features("sin x", int_var="x")
        f2 = extract_features("exp x", int_var="x")
        assert not np.allclose(f1, f2)
