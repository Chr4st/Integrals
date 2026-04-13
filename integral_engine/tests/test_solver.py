"""Tests for the public solve_integral API."""
import time

import pytest

from integral_engine.solver import solve_integral


class TestSolveIntegral:
    def test_sin_x_solves(self):
        result = solve_integral("sin(x)")
        assert result["status"] in ["solved_sympy", "solved_ml"]
        assert result["latex"] is not None
        assert result["latex"].endswith("+ C")

    def test_x_squared(self):
        result = solve_integral("x**2")
        assert result["status"] == "solved_sympy"
        assert result["latex"] is not None

    def test_invalid_expression_raises(self):
        with pytest.raises(ValueError):
            solve_integral("+++invalid(((")

    def test_returns_within_timeout(self):
        start = time.time()
        solve_integral("sin(x)")
        assert time.time() - start < 30

    def test_result_structure(self):
        result = solve_integral("cos(x)")
        assert "status" in result
        assert "latex" in result
        assert "method" in result
        assert "confidence" in result

    def test_custom_variable(self):
        result = solve_integral("sin(t)", var="t")
        assert result["status"] == "solved_sympy"
