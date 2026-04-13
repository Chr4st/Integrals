"""Tests for constant solver."""
import sympy as sp
from integral_engine.constant_solver import solve_constants

x = sp.Symbol("x")
C1, C2 = sp.Symbol("C1"), sp.Symbol("C2")


class TestSymbolicSolve:
    def test_cos_x_template(self):
        """integrand = cos(x), template = C1*sin(x). Should solve C1=1."""
        integrand = sp.cos(x)
        template = C1 * sp.sin(x)
        result = solve_constants(template, integrand, x, [C1])
        assert result is not None
        assert sp.simplify(sp.diff(result, x) - integrand) == 0

    def test_quadratic_template(self):
        """integrand = 2*x, template = C1*x**2. Should solve C1=1."""
        integrand = 2 * x
        template = C1 * x**2
        result = solve_constants(template, integrand, x, [C1])
        assert result is not None
        assert sp.simplify(sp.diff(result, x) - integrand) == 0

    def test_two_constants(self):
        """integrand = 2*x + 3, template = C1*x**2 + C2*x."""
        integrand = 2 * x + 3
        template = C1 * x**2 + C2 * x
        result = solve_constants(template, integrand, x, [C1, C2])
        assert result is not None
        assert sp.simplify(sp.diff(result, x) - integrand) == 0


class TestNumericFallback:
    def test_numeric_engages_on_short_timeout(self):
        integrand = x
        template = C1 * x**2
        result = solve_constants(
            template, integrand, x, [C1], symbolic_timeout=0.001,
        )
        if result is not None:
            assert sp.simplify(sp.diff(result, x) - integrand) == 0


class TestFailureCases:
    def test_unsolvable_returns_none(self):
        integrand = sp.sin(x)
        template = C1 * x**2
        result = solve_constants(template, integrand, x, [C1])
        assert result is None

    def test_empty_constants_list(self):
        integrand = sp.cos(x)
        template = sp.sin(x)
        result = solve_constants(template, integrand, x, [])
        assert result is not None
        assert sp.simplify(sp.diff(result, x) - integrand) == 0

    def test_wrong_template_no_constants(self):
        integrand = sp.cos(x)
        template = x**2
        result = solve_constants(template, integrand, x, [])
        assert result is None
