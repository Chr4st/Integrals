"""Tests for expression rewriting augmentation."""
import random

import pytest
import sympy as sp

from integral_engine.augmentation import (
    _permute_commutative_args,
    _random_expand_or_factor,
    _trig_rewrite,
    augment_expression,
)


class TestPermuteCommutativeArgs:
    def test_returns_valid_expression(self):
        x = sp.Symbol("x")
        expr = x + sp.sin(x) + sp.cos(x)
        rng = random.Random(42)
        result = _permute_commutative_args(expr, rng)
        # Result should be mathematically equivalent
        assert sp.simplify(result - expr) == 0

    def test_mul_returns_valid_expression(self):
        x = sp.Symbol("x")
        expr = x * sp.sin(x) * sp.cos(x)
        rng = random.Random(42)
        result = _permute_commutative_args(expr, rng)
        assert sp.simplify(result - expr) == 0

    def test_mathematically_equivalent(self):
        x = sp.Symbol("x")
        expr = x + 2 * sp.sin(x) + 3
        rng = random.Random(42)
        permuted = _permute_commutative_args(expr, rng)

        # Should be mathematically equal when simplified
        diff = sp.simplify(sp.sympify(str(permuted)) - expr)
        assert diff == 0


class TestExpandOrFactor:
    def test_expand(self):
        x = sp.Symbol("x")
        expr = (x + 1) ** 2
        result = _random_expand_or_factor(expr, random.Random(0))
        # Either expanded or factored — both should simplify to original
        assert sp.simplify(result - expr) == 0

    def test_factor(self):
        x = sp.Symbol("x")
        expr = x**2 + 2*x + 1
        result = _random_expand_or_factor(expr, random.Random(1))
        assert sp.simplify(result - expr) == 0


class TestTrigRewrite:
    def test_trig_simplification(self):
        x = sp.Symbol("x")
        expr = sp.sin(x)**2 + sp.cos(x)**2
        result = _trig_rewrite(expr, random.Random(0))
        # Should simplify to 1 (or at least be equivalent)
        assert sp.simplify(result - 1) == 0 or sp.simplify(result - expr) == 0

    def test_no_crash_on_non_trig(self):
        x = sp.Symbol("x")
        expr = x**2 + 1
        result = _trig_rewrite(expr, random.Random(42))
        # Should return something, even if unchanged
        assert result is not None


class TestAugmentExpression:
    def test_returns_variants(self):
        x = sp.Symbol("x")
        expr = sp.sin(x) + sp.cos(x) * x
        variants = augment_expression(expr, random.Random(42), n_variants=3)
        assert len(variants) >= 1

    def test_variants_are_unique(self):
        x = sp.Symbol("x")
        expr = x**2 + 2*x + sp.sin(x)
        variants = augment_expression(expr, random.Random(42), n_variants=5)
        reprs = [sp.srepr(v) for v in variants]
        assert len(reprs) == len(set(reprs))

    def test_variants_mathematically_equivalent(self):
        x = sp.Symbol("x")
        # Use polynomial expr to avoid trig equivalence checking issues
        expr = x**2 + 2*x + 1
        variants = augment_expression(expr, random.Random(42), n_variants=3)
        for v in variants:
            diff = sp.simplify(v - expr)
            assert diff == 0, f"Variant {v} not equivalent to {expr}"

    def test_empty_for_simple_constant(self):
        # A constant has no commutative args to permute and can't be expanded/factored
        expr = sp.Integer(5)
        variants = augment_expression(expr, random.Random(42), n_variants=3)
        # May return 0 variants since no meaningful rewrites exist
        assert isinstance(variants, list)
