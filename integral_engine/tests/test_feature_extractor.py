"""Tests for depth-encoded feature extractor."""
import numpy as np
import sympy as sp
from integral_engine.feature_extractor import extract_features, extract_features_debug
from integral_engine.vocabulary import FEATURE_DIM, NUM_DEPTH_BINS, depth_to_bin

x = sp.Symbol("x")


def test_sin_log_x():
    debug = extract_features_debug(sp.sin(sp.log(x)), x)
    assert debug[(("sin", depth_to_bin(2)))] == 1
    assert debug[(("log", depth_to_bin(1)))] == 1


def test_log_sin_x():
    debug = extract_features_debug(sp.log(sp.sin(x)), x)
    assert debug[(("log", depth_to_bin(2)))] == 1
    assert debug[(("sin", depth_to_bin(1)))] == 1


def test_sin_log_vs_log_sin_distinct():
    v1 = extract_features(sp.sin(sp.log(x)), x)
    v2 = extract_features(sp.log(sp.sin(x)), x)
    assert not np.array_equal(v1, v2)


def test_sin_plus_log():
    debug = extract_features_debug(sp.sin(x) + sp.log(x), x)
    assert debug[("Add", depth_to_bin(2))] == 1
    assert debug[("sin", depth_to_bin(1))] == 1
    assert debug[("log", depth_to_bin(1))] == 1


def test_sin_plus_log_distinct_from_composed():
    v_sum = extract_features(sp.sin(x) + sp.log(x), x)
    v_sin_log = extract_features(sp.sin(sp.log(x)), x)
    v_log_sin = extract_features(sp.log(sp.sin(x)), x)
    assert not np.array_equal(v_sum, v_sin_log)
    assert not np.array_equal(v_sum, v_log_sin)


def test_exp_x_is_exp():
    """Phase 12: exp(x) should be classified as 'Exp'."""
    debug = extract_features_debug(sp.exp(x), x)
    assert debug.get(("Exp", depth_to_bin(1)), 0) >= 1


def test_2_to_x_is_exp():
    """Phase 12: 2^x should be classified as 'Exp' (variable in exponent only)."""
    debug = extract_features_debug(sp.Integer(2) ** x, x)
    assert debug.get(("Exp", depth_to_bin(1)), 0) >= 1
    assert debug.get(("Pow", depth_to_bin(1)), 0) == 0


def test_x_squared_is_pow_not_exp():
    """Phase 12: x^2 should be 'Pow', not 'Exp' or 'Tower'."""
    debug = extract_features_debug(x**2, x)
    assert debug.get(("Pow", depth_to_bin(1)), 0) >= 1
    assert debug.get(("Exp", depth_to_bin(1)), 0) == 0
    assert debug.get(("Tower", depth_to_bin(1)), 0) == 0


def test_x_to_the_x_is_tower():
    """Phase 12: x^x should be classified as 'Tower' (variable in both)."""
    debug = extract_features_debug(x**x, x)
    assert debug.get(("Tower", depth_to_bin(1)), 0) >= 1
    assert debug.get(("Pow", depth_to_bin(1)), 0) == 0
    assert debug.get(("Exp", depth_to_bin(1)), 0) == 0


def test_logarithmic_depth_binning():
    """Phase 13: Verify logarithmic binning behavior."""
    # Depths 1-4 get their own bin
    assert depth_to_bin(1) == 0
    assert depth_to_bin(2) == 1
    assert depth_to_bin(3) == 2
    assert depth_to_bin(4) == 3
    # Grouped bins
    assert depth_to_bin(5) == 4
    assert depth_to_bin(6) == 4
    assert depth_to_bin(7) == 5
    assert depth_to_bin(9) == 5
    assert depth_to_bin(10) == 6
    assert depth_to_bin(15) == 6
    assert depth_to_bin(16) == 7
    assert depth_to_bin(100) == 7


def test_depth_bins_high_nesting():
    """Phase 13: Deeply nested expressions use higher bins instead of clamping at 4."""
    expr = x
    for _ in range(10):
        expr = sp.sin(expr)
    debug = extract_features_debug(expr, x)
    # Should see entries in bins > 4 (which was the old max)
    for key in debug:
        assert key[1] < NUM_DEPTH_BINS


def test_unknown_head_ignored():
    f = sp.Function("mystery_func")
    expr = f(x) + sp.sin(x)
    vec = extract_features(expr, x)
    debug = extract_features_debug(expr, x)
    assert debug.get(("sin", depth_to_bin(1)), 0) == 1
    assert vec.shape == (FEATURE_DIM,)


def test_output_shape_and_dtype():
    vec = extract_features(sp.sin(x), x)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    # Phase 12+13: 76 heads × 8 bins = 608
    assert FEATURE_DIM == 608
