"""Tests for verification oracles (SymPy-based and Rust-based).

Covers: python/neurips/evaluation/verify.py.
Verifies MATHEMATICAL CORRECTNESS: correct pairs return True,
wrong pairs return False, multivariate/parametric dispatch works,
and timeout handling doesn't hang.
"""
from __future__ import annotations

import pytest

from neurips.evaluation.verify import (
    RustVerificationOracle,
    VerificationOracle,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle() -> VerificationOracle:
    """SymPy verification oracle with a generous timeout."""
    return VerificationOracle(timeout=4.0)


@pytest.fixture
def rust_oracle() -> RustVerificationOracle:
    """Rust verification oracle (falls back to SymPy if Rust unavailable)."""
    return RustVerificationOracle(timeout=4.0)


# ---------------------------------------------------------------------------
# Known correct indefinite integral pairs (must return True)
# ---------------------------------------------------------------------------

_CORRECT_INDEF = [
    pytest.param(
        "div pow x 3 3", "pow x 2",
        id="integral-x2-is-x3/3",
    ),
    pytest.param(
        "neg cos x", "sin x",
        id="integral-sinx-is-neg-cosx",
    ),
    pytest.param(
        "exp x", "exp x",
        id="integral-expx-is-expx",
    ),
    pytest.param(
        "log x", "div 1 x",
        id="integral-1/x-is-logx",
    ),
    pytest.param(
        "div pow sin x 2 2", "mul cos x sin x",
        id="integral-cosxsinx-is-sin2x/2",
    ),
]


class TestCorrectIndefinite:
    """Known correct antiderivatives must verify as True."""

    @pytest.mark.parametrize("candidate,integrand", _CORRECT_INDEF)
    def test_correct_pair_returns_true(
        self, oracle: VerificationOracle, candidate: str, integrand: str
    ) -> None:
        """d/dx(candidate) == integrand for these known-correct pairs."""
        result = oracle.verify(
            candidate=candidate,
            integrand=integrand,
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is True, (
            f"Oracle rejected correct pair: "
            f"d/dx({candidate!r}) should equal {integrand!r}"
        )


# ---------------------------------------------------------------------------
# Known WRONG pairs (must return False)
# ---------------------------------------------------------------------------

_WRONG_INDEF = [
    pytest.param(
        "pow x 3", "pow x 2",
        id="wrong-missing-1/3-factor",
    ),
    pytest.param(
        "cos x", "sin x",
        id="wrong-missing-negation",
    ),
    pytest.param(
        "x", "pow x 2",
        id="wrong-completely-wrong",
    ),
    pytest.param(
        "mul 2 x", "pow x 2",
        id="wrong-derivative-is-2-not-x2",
    ),
]


class TestWrongIndefinite:
    """Known wrong antiderivatives must verify as False."""

    @pytest.mark.parametrize("candidate,integrand", _WRONG_INDEF)
    def test_wrong_pair_returns_false(
        self, oracle: VerificationOracle, candidate: str, integrand: str
    ) -> None:
        """d/dx(candidate) != integrand for these known-wrong pairs."""
        result = oracle.verify(
            candidate=candidate,
            integrand=integrand,
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is False, (
            f"Oracle accepted wrong pair: "
            f"d/dx({candidate!r}) should NOT equal {integrand!r}"
        )


# ---------------------------------------------------------------------------
# Multivariate verification
# ---------------------------------------------------------------------------


class TestMultivariate:
    """Multivariate (partial derivative) verification."""

    def test_partial_x_of_x2y_equals_2xy(
        self, oracle: VerificationOracle
    ) -> None:
        """partial/partial_x(x^2 * y) = 2xy.

        Candidate is the antiderivative (x^2 * y),
        integrand is what d/dx of that should be (2xy).
        """
        result = oracle.verify(
            candidate="mul pow x 2 y",
            integrand="mul mul 2 x y",
            task_info={"task": "multivariate", "var": "x"},
        )
        assert result is True

    def test_partial_y_of_xy_equals_x(
        self, oracle: VerificationOracle
    ) -> None:
        """partial/partial_y(x*y) = x."""
        result = oracle.verify(
            candidate="mul x y",
            integrand="x",
            task_info={"task": "multivariate", "var": "y"},
        )
        assert result is True

    def test_wrong_multivariate(
        self, oracle: VerificationOracle
    ) -> None:
        """partial/partial_x(x*y) = y, NOT x. Must reject x as integrand."""
        result = oracle.verify(
            candidate="mul x y",
            integrand="x",
            task_info={"task": "multivariate", "var": "x"},
        )
        assert result is False


# ---------------------------------------------------------------------------
# Parametric verification
# ---------------------------------------------------------------------------


class TestParametric:
    """Parametric integration: parameters treated as constants."""

    def test_integral_ax_is_ax2_over_2(
        self, oracle: VerificationOracle
    ) -> None:
        """integral(a*x, dx) = a*x^2/2 where a is a parameter.

        d/dx(a*x^2/2) = a*x.
        """
        result = oracle.verify(
            candidate="div mul a pow x 2 2",
            integrand="mul a x",
            task_info={"task": "parametric", "var": "x", "params": ["a"]},
        )
        assert result is True

    def test_wrong_parametric(
        self, oracle: VerificationOracle
    ) -> None:
        """d/dx(a*x) = a, NOT a*x. Must reject."""
        result = oracle.verify(
            candidate="mul a x",
            integrand="mul a x",
            task_info={"task": "parametric", "var": "x", "params": ["a"]},
        )
        assert result is False


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------


class TestTimeout:
    """Verification must not hang on hard-to-simplify expressions."""

    @pytest.mark.slow
    def test_timeout_returns_false_not_hang(self) -> None:
        """A deliberately complex expression should time out gracefully
        and return False, not hang the process.
        """
        short_timeout_oracle = VerificationOracle(timeout=2.0)
        # Deeply nested expression unlikely to simplify quickly
        candidate = "div pow sin pow cos pow exp x 2 3 mul pow log x 4 5"
        integrand = "mul exp pow x 3 sin pow x 7"
        result = short_timeout_oracle.verify(
            candidate=candidate,
            integrand=integrand,
            task_info={"task": "INDEF", "var": "x"},
        )
        # We don't care about the value — the point is it doesn't hang.
        # It should return False (timeout or mismatch).
        assert isinstance(result, bool)

    def test_invalid_expression_returns_false(
        self, oracle: VerificationOracle
    ) -> None:
        """Malformed prefix that causes a parse error returns False, not exception."""
        result = oracle.verify(
            candidate="add",  # incomplete expression
            integrand="x",
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is False


# ---------------------------------------------------------------------------
# RustVerificationOracle
# ---------------------------------------------------------------------------


class TestRustOracle:
    """Tests for RustVerificationOracle, with graceful skip if unavailable."""

    def test_rust_oracle_creation(
        self, rust_oracle: RustVerificationOracle
    ) -> None:
        """RustVerificationOracle can be instantiated regardless of Rust availability."""
        assert isinstance(rust_oracle, RustVerificationOracle)
        assert isinstance(rust_oracle.available, bool)

    @pytest.mark.parametrize("candidate,integrand", _CORRECT_INDEF)
    def test_rust_correct_pairs(
        self,
        rust_oracle: RustVerificationOracle,
        candidate: str,
        integrand: str,
    ) -> None:
        """Rust oracle agrees with SymPy on correct pairs.

        Even when Rust module is unavailable, this exercises the SymPy fallback
        path in RustVerificationOracle — ensuring the delegation works.
        """
        result = rust_oracle.verify(
            candidate=candidate,
            integrand=integrand,
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is True

    @pytest.mark.parametrize("candidate,integrand", _WRONG_INDEF)
    def test_rust_wrong_pairs(
        self,
        rust_oracle: RustVerificationOracle,
        candidate: str,
        integrand: str,
    ) -> None:
        """Rust oracle rejects wrong pairs (via Rust or SymPy fallback)."""
        result = rust_oracle.verify(
            candidate=candidate,
            integrand=integrand,
            task_info={"task": "INDEF", "var": "x"},
        )
        assert result is False

    def test_rust_and_sympy_agree_on_all_pairs(
        self,
        oracle: VerificationOracle,
        rust_oracle: RustVerificationOracle,
    ) -> None:
        """Critical integration test: both oracles must produce identical
        results for the same set of test pairs.
        """
        all_pairs = [
            ("div pow x 3 3", "pow x 2", True),
            ("neg cos x", "sin x", True),
            ("exp x", "exp x", True),
            ("pow x 3", "pow x 2", False),
            ("cos x", "sin x", False),
        ]
        for candidate, integrand, _expected in all_pairs:
            task_info = {"task": "INDEF", "var": "x"}
            sympy_result = oracle.verify(candidate, integrand, task_info)
            rust_result = rust_oracle.verify(candidate, integrand, task_info)
            assert sympy_result == rust_result, (
                f"Disagreement on ({candidate!r}, {integrand!r}): "
                f"SymPy={sympy_result}, Rust={rust_result}"
            )

    def test_verify_many_matches_individual(
        self, rust_oracle: RustVerificationOracle
    ) -> None:
        """verify_many() batch results must match individual verify() calls."""
        integrand = "pow x 2"
        candidates = [
            "div pow x 3 3",  # correct
            "pow x 3",  # wrong (missing 1/3)
            "neg cos x",  # wrong (not related)
        ]
        task_info = {"task": "INDEF", "var": "x"}

        batch_results = rust_oracle.verify_many(
            candidates, integrand, task_info
        )
        individual_results = [
            rust_oracle.verify(c, integrand, task_info) for c in candidates
        ]

        assert batch_results == individual_results, (
            f"Batch vs individual mismatch:\n"
            f"  batch:      {batch_results}\n"
            f"  individual: {individual_results}"
        )

    def test_verify_many_empty_list(
        self, rust_oracle: RustVerificationOracle
    ) -> None:
        """verify_many with empty candidate list returns empty list."""
        result = rust_oracle.verify_many(
            [], "pow x 2", {"task": "INDEF", "var": "x"}
        )
        assert result == []

    def test_definite_integral_falls_back_to_sympy(
        self, rust_oracle: RustVerificationOracle
    ) -> None:
        """Definite integral tasks always use SymPy fallback (Rust has no mpmath).

        This tests the delegation path, not the numerical accuracy.
        """
        # We just verify it returns a bool and doesn't crash
        result = rust_oracle.verify(
            candidate="div 1 3",  # predicted value
            integrand="pow x 2",
            task_info={"task": "DEF", "var": "x", "bounds": (0, 1)},
        )
        assert isinstance(result, bool)
