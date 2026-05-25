"""Tests for the verification oracle (Phase 13)."""
import pytest

# These tests require sympy, mark as slow
pytestmark = pytest.mark.timeout(10)


class TestOracleImport:
    def test_import(self):
        from neurips.evaluation.verify import VerificationOracle


class TestIndefiniteVerification:
    def test_correct_power_rule(self):
        from neurips.evaluation.verify import VerificationOracle
        oracle = VerificationOracle(timeout=4.0)
        # d/dx(x^3/3) = x^2 ✓
        result = oracle.verify(
            candidate="div pow x 3 3",
            integrand="pow x 2",
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is True

    def test_wrong_answer(self):
        from neurips.evaluation.verify import VerificationOracle
        oracle = VerificationOracle(timeout=4.0)
        # d/dx(x^3) = 3x^2 ≠ x^2
        result = oracle.verify(
            candidate="pow x 3",
            integrand="pow x 2",
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is False

    def test_sin_cos(self):
        from neurips.evaluation.verify import VerificationOracle
        oracle = VerificationOracle(timeout=4.0)
        # d/dx(-cos(x)) = sin(x) ✓
        result = oracle.verify(
            candidate="neg cos x",
            integrand="sin x",
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is True


class TestMultivariateVerification:
    def test_partial_derivative(self):
        from neurips.evaluation.verify import VerificationOracle
        oracle = VerificationOracle(timeout=4.0)
        # ∂/∂x(x²y/2) = xy ✓
        result = oracle.verify(
            candidate="div mul pow x 2 y 2",
            integrand="mul x y",
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is True
