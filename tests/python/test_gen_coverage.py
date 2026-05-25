"""Tests for coverage-guaranteed data generation."""

import sympy as sp
from sympy import Symbol

x = Symbol("x")


def _families():
    from scripts.generate_covered import SKELETON_FAMILIES
    return SKELETON_FAMILIES


def _generate(family):
    from scripts.generate_covered import generate_from_skeleton
    return generate_from_skeleton(family)


class TestSkeletonFamilies:
    def test_all_families_registered(self):
        families = _families()
        assert len(families) >= 15

    def test_expected_families_present(self):
        families = _families()
        expected = [
            "elementary", "linear_sub", "composition", "u_sub",
            "ibp_poly_exp", "ibp_poly_trig", "ibp_poly_log",
            "ibp_exp_trig", "ibp_inverse_trig", "rational",
            "trig_power", "trig_sub", "power_rule", "exp_chain",
            "hyperbolic", "triple_comp", "poly_product",
        ]
        for fam in expected:
            assert fam in families, f"Missing family: {fam}"


class TestSkeletonGeneration:
    def test_elementary_generates(self):
        result = _generate("elementary")
        assert result is not None
        integrand, antideriv = result
        assert integrand != 0

    def test_ibp_generates(self):
        result = _generate("ibp_poly_exp")
        assert result is not None

    def test_rational_generates(self):
        result = _generate("rational")
        assert result is not None

    def test_trig_power_generates(self):
        result = _generate("trig_power")
        assert result is not None


class TestDifferentiationCorrectness:
    def test_elementary_pair_verifies(self):
        result = _generate("elementary")
        assert result is not None
        integrand, antideriv = result
        computed = sp.diff(antideriv, x)
        diff = sp.simplify(computed - integrand)
        assert diff == 0 or abs(float(diff.subs(x, 1.0))) < 1e-10

    def test_usub_pair_verifies(self):
        result = _generate("u_sub")
        assert result is not None
        integrand, antideriv = result
        computed = sp.diff(antideriv, x)
        diff = sp.simplify(computed - integrand)
        assert diff == 0 or abs(float(diff.subs(x, 0.5))) < 1e-10

    def test_power_rule_verifies(self):
        result = _generate("power_rule")
        assert result is not None
        integrand, antideriv = result
        computed = sp.diff(antideriv, x)
        diff = sp.simplify(computed - integrand)
        assert diff == 0 or abs(float(diff.subs(x, 2.0))) < 1e-6
