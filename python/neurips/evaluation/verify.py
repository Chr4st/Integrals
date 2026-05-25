"""Verification oracles — SymPy-based and Rust-based."""

from __future__ import annotations

import random
from typing import Any

import mpmath
import sympy

from neurips.data.prefix import python_prefix_to_rust
from neurips.evaluation.oracle import _timeout, prefix_to_sympy


class VerificationOracle:
    """Check whether a candidate antiderivative is correct."""

    def __init__(self, timeout: float = 4.0) -> None:
        self.timeout = timeout

    def verify(
        self,
        candidate: str,
        integrand: str,
        task_info: dict[str, Any],
    ) -> bool:
        """Dispatch verification by task type."""
        try:
            with _timeout(self.timeout):
                return self._dispatch(candidate, integrand, task_info)
        except TimeoutError:
            return False
        except (sympy.SympifyError, ValueError, ZeroDivisionError,
                OverflowError, TypeError, AttributeError):
            return False

    def _dispatch(
        self,
        candidate: str,
        integrand: str,
        task_info: dict[str, Any],
    ) -> bool:
        task = task_info.get("task", "INDEF")
        var = task_info.get("var", "x")

        if task == "DEF":
            return self._verify_definite(
                candidate, integrand, task_info["bounds"]
            )
        if task in ("multivariate",):
            return self._verify_multivariate(candidate, integrand, var)
        if task in ("parametric",):
            params = task_info.get("params", [])
            return self._verify_parametric(
                candidate, integrand, var, params
            )
        # Default: indefinite (univariate, special_fn)
        return self._verify_indefinite(candidate, integrand, var)

    def _verify_indefinite(
        self, F_str: str, f_str: str, var: str
    ) -> bool:
        """Check d/dx F(x) == f(x) symbolically, then numerically."""
        F_sym = prefix_to_sympy(F_str)
        f_sym = prefix_to_sympy(f_str)
        x = sympy.Symbol(var)

        dF = sympy.diff(F_sym, x)
        diff = sympy.simplify(dF - f_sym)
        if diff == 0:
            return True

        return self._numerical_check(dF, f_sym, {x}, 20, 18)

    def _verify_definite(
        self, predicted_str: str, f_str: str, bounds: tuple[float, float]
    ) -> bool:
        """Compare predicted value to mpmath numerical integration."""
        f_sym = prefix_to_sympy(f_str)
        pred_sym = prefix_to_sympy(predicted_str)

        mpmath.mp.dps = 50
        f_func = sympy.lambdify("x", f_sym, modules=["mpmath"])
        try:
            numerical = mpmath.quad(
                f_func, [float(bounds[0]), float(bounds[1])]
            )
        except Exception:
            return False

        try:
            predicted = mpmath.mpf(str(pred_sym.evalf(50)))
        except Exception:
            return False

        return bool(abs(predicted - numerical) < mpmath.mpf("1e-6"))

    def _verify_multivariate(
        self, F_str: str, f_str: str, var: str
    ) -> bool:
        """Check partial derivative dF/d(var) == f."""
        F_sym = prefix_to_sympy(F_str)
        f_sym = prefix_to_sympy(f_str)
        var_sym = sympy.Symbol(var)

        dF = sympy.diff(F_sym, var_sym)
        diff = sympy.simplify(dF - f_sym)
        if diff == 0:
            return True

        all_vars = F_sym.free_symbols | f_sym.free_symbols
        return self._numerical_check(dF, f_sym, all_vars, 20, 18)

    def _verify_parametric(
        self, F_str: str, f_str: str, var: str, params: list[str]
    ) -> bool:
        """Check dF/d(var) == f treating params as constants."""
        F_sym = prefix_to_sympy(F_str)
        f_sym = prefix_to_sympy(f_str)
        var_sym = sympy.Symbol(var)

        dF = sympy.diff(F_sym, var_sym)
        diff = sympy.simplify(dF - f_sym)
        if diff == 0:
            return True

        all_vars = F_sym.free_symbols | f_sym.free_symbols
        return self._numerical_check(dF, f_sym, all_vars, 30, 27)

    @staticmethod
    def _numerical_check(
        expr_a: sympy.Expr,
        expr_b: sympy.Expr,
        variables: set[sympy.Symbol],
        n_points: int,
        threshold: int,
    ) -> bool:
        """Evaluate at random points; accept if enough agree."""
        agreements = 0
        for _ in range(n_points):
            subs = {v: random.uniform(-5, 5) for v in variables}
            try:
                val_a = complex(expr_a.subs(subs))
                val_b = complex(expr_b.subs(subs))
                if abs(val_a - val_b) < 1e-8 * max(1, abs(val_b)):
                    agreements += 1
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
        return agreements >= threshold


# ── Rust-backed verification oracle ─────────────────────────────────


class RustVerificationOracle:
    """Fast verification using the Rust ``py_verify_batch`` backend.

    Falls back to :class:`VerificationOracle` for definite integrals
    (Rust has no ``mpmath.quad`` equivalent).
    """

    VERDICTS = frozenset({
        "correct", "incorrect_symbolic", "incorrect_numerical", "timeout",
    })

    def __init__(self, timeout: float = 4.0) -> None:
        self.timeout = timeout
        self._sympy_fallback = VerificationOracle(timeout=timeout)
        try:
            from neurips._core import verify_batch as _rust_verify  # noqa: F401
            self._rust_verify = _rust_verify
        except ImportError:
            self._rust_verify = None
        try:
            from neurips._core import verify_batch_rich as _rust_verify_rich  # noqa: F401
            self._rust_verify_rich = _rust_verify_rich
        except ImportError:
            self._rust_verify_rich = None

    @property
    def available(self) -> bool:
        return self._rust_verify is not None

    def verify(
        self,
        candidate: str,
        integrand: str,
        task_info: dict[str, Any],
    ) -> bool:
        """Verify a single candidate. Uses Rust for non-definite tasks."""
        task = task_info.get("task", "INDEF")
        if task == "DEF" or self._rust_verify is None:
            return self._sympy_fallback.verify(
                candidate, integrand, task_info
            )

        var = task_info.get("var", "x")
        try:
            f_rust = python_prefix_to_rust(integrand)
            big_f_rust = python_prefix_to_rust(candidate)
            results = self._rust_verify([(f_rust, big_f_rust, var)])
            return bool(results[0])
        except Exception:
            return self._sympy_fallback.verify(
                candidate, integrand, task_info
            )

    def verify_rich(
        self,
        candidate: str,
        integrand: str,
        task_info: dict[str, Any],
    ) -> str:
        """Verify a single candidate and return verdict string.

        Returns one of: "correct", "incorrect_symbolic",
        "incorrect_numerical", "timeout".
        Falls back to boolean SymPy oracle mapped to verdict strings.
        """
        task = task_info.get("task", "INDEF")
        if task == "DEF" or self._rust_verify_rich is None:
            ok = self._sympy_fallback.verify(candidate, integrand, task_info)
            return "correct" if ok else "incorrect_numerical"

        var = task_info.get("var", "x")
        try:
            f_rust = python_prefix_to_rust(integrand)
            big_f_rust = python_prefix_to_rust(candidate)
            results = self._rust_verify_rich([(f_rust, big_f_rust, var)])
            return results[0]
        except Exception:
            ok = self._sympy_fallback.verify(candidate, integrand, task_info)
            return "correct" if ok else "incorrect_numerical"

    def verify_many(
        self,
        candidates: list[str],
        integrand: str,
        task_info: dict[str, Any],
    ) -> list[bool]:
        """Batch-verify multiple candidates for the same integrand.

        Returns a list of bools, one per candidate.
        """
        task = task_info.get("task", "INDEF")
        if task == "DEF" or self._rust_verify is None:
            return [
                self._sympy_fallback.verify(c, integrand, task_info)
                for c in candidates
            ]

        var = task_info.get("var", "x")
        try:
            f_rust = python_prefix_to_rust(integrand)
            pairs = [
                (f_rust, python_prefix_to_rust(c), var)
                for c in candidates
            ]
            return [bool(r) for r in self._rust_verify(pairs)]
        except Exception:
            return [
                self._sympy_fallback.verify(c, integrand, task_info)
                for c in candidates
            ]

    def verify_many_rich(
        self,
        candidates: list[str],
        integrand: str,
        task_info: dict[str, Any],
    ) -> list[str]:
        """Batch-verify with rich verdict strings.

        Returns a list of verdict strings, one per candidate.
        """
        task = task_info.get("task", "INDEF")
        if task == "DEF" or self._rust_verify_rich is None:
            return [
                "correct" if self._sympy_fallback.verify(c, integrand, task_info)
                else "incorrect_numerical"
                for c in candidates
            ]

        var = task_info.get("var", "x")
        try:
            f_rust = python_prefix_to_rust(integrand)
            pairs = [
                (f_rust, python_prefix_to_rust(c), var)
                for c in candidates
            ]
            return list(self._rust_verify_rich(pairs))
        except Exception:
            return [
                "correct" if self._sympy_fallback.verify(c, integrand, task_info)
                else "incorrect_numerical"
                for c in candidates
            ]
