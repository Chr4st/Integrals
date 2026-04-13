"""Symbolic and numeric constant solver for predicted integral templates.

Given a predicted antiderivative template T(x, C1, C2, ...) and the original
integrand f(x), determines the values of C1, C2, ... such that dT/dx = f(x).

Strategy:
  1. Symbolic: differentiate T, solve the system dT/dx = f(x) for the constants.
  2. Numeric fallback: Sobol quasi-random sampling on [-10, 10], split into
     70% train / 30% verify, multi-start L-BFGS-B, cascading nsimplify.
  Both use persistent ProcessPoolExecutor singletons for safe timeouts.
"""
from __future__ import annotations

import atexit
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Tuple

import numpy as np
import structlog
import sympy as sp

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Phase 10: Persistent process pool singleton
# ---------------------------------------------------------------------------
_pool: Optional[ProcessPoolExecutor] = None
_pool_lock = threading.Lock()


def _get_pool() -> ProcessPoolExecutor:
    """Return a module-level singleton ProcessPoolExecutor."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=2)
            atexit.register(_shutdown_pool)
    return _pool


def _shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None


# ---------------------------------------------------------------------------
# Symbolic solver
# ---------------------------------------------------------------------------
def _symbolic_solve(
    template: sp.Basic,
    integrand: sp.Basic,
    var: sp.Symbol,
    constants: List[sp.Symbol],
) -> Optional[sp.Basic]:
    """Try to solve for constants symbolically."""
    if not constants:
        diff = sp.diff(template, var)
        if sp.simplify(diff - integrand) == 0:
            return template
        return None

    dT = sp.diff(template, var)
    eq = sp.Eq(dT, integrand)
    solutions = sp.solve(eq, constants, dict=True)

    if not solutions:
        return None

    sol = solutions[0]
    if len(sol) != len(constants):
        return None

    result = template.subs(sol)
    if sp.simplify(sp.diff(result, var) - integrand) == 0:
        return result
    return None


# ---------------------------------------------------------------------------
# Phase 7: Sobol quasi-random sampling + multi-start L-BFGS-B
# Phase 8: Configurable seed + train/verify split
# Phase 9: Cascading nsimplify tolerance
# ---------------------------------------------------------------------------
_RATIONALIZE_CONSTANTS = [sp.pi, sp.E, sp.sqrt(2), sp.sqrt(3), sp.log(2), sp.GoldenRatio]
_RATIONALIZE_TOLERANCES = (1e-10, 1e-8, 1e-6)


def _rationalize(val: float) -> sp.Basic:
    """Convert a numeric value to a symbolic constant with cascading tolerance.

    Tries nsimplify with common-constant hints at progressively looser
    tolerances. Falls back to Rational.limit_denominator(10000).
    """
    for tol in _RATIONALIZE_TOLERANCES:
        try:
            candidate = sp.nsimplify(
                val, constants=_RATIONALIZE_CONSTANTS, tolerance=tol, rational=True,
            )
            if abs(float(candidate) - val) < tol * 10:
                return candidate
        except (ValueError, OverflowError, ZeroDivisionError):
            continue

    # Fallback: rational approximation
    return sp.Rational(val).limit_denominator(10000)


def _generate_sample_points(
    integrand: sp.Basic,
    var: sp.Symbol,
    n_total: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample points using Sobol quasi-random sequences on [-10, 10].

    Filters to points where the integrand evaluates to a finite value
    with |f(x)| < 1e12.
    """
    from scipy.stats.qmc import Sobol

    sampler = Sobol(d=1, scramble=True, seed=seed)
    # Generate 4x candidates to account for filtering
    raw = sampler.random(n_total * 4)
    candidates = raw[:, 0] * 20.0 - 10.0  # Scale [0,1] → [-10, 10]

    x_points: List[float] = []
    f_vals: List[float] = []

    for xv in candidates:
        try:
            fv = float(integrand.subs(var, float(xv)))
            if np.isfinite(fv) and abs(fv) < 1e12:
                x_points.append(float(xv))
                f_vals.append(fv)
                if len(x_points) >= n_total:
                    break
        except Exception:
            continue

    return np.array(x_points), np.array(f_vals)


def _numeric_solve(
    template: sp.Basic,
    integrand: sp.Basic,
    var: sp.Symbol,
    constants: List[sp.Symbol],
    n_points: int = 40,
    seed: int = 42,
    verify_ratio: float = 0.3,
    n_restarts: int = 3,
) -> Optional[sp.Basic]:
    """Solve for constants numerically via multi-start L-BFGS-B.

    Uses Sobol sampling, splits points into train/verify sets,
    fits on train, validates on held-out verify set.
    """
    from scipy.optimize import minimize

    if not constants:
        return None

    dT = sp.diff(template, var)

    # Phase 7: Sobol sampling on [-10, 10]
    x_arr, f_arr = _generate_sample_points(integrand, var, n_points, seed)

    if len(x_arr) < 5:
        return None

    # Phase 8: Train/verify split
    n_verify = max(2, int(len(x_arr) * verify_ratio))
    n_train = len(x_arr) - n_verify
    x_train, f_train = x_arr[:n_train], f_arr[:n_train]
    x_verify, f_verify = x_arr[n_train:], f_arr[n_train:]

    all_syms = [var] + constants
    dT_func = sp.lambdify(all_syms, dT, modules=["numpy"])

    def residual(c_vals: np.ndarray, x_pts: np.ndarray, f_pts: np.ndarray) -> float:
        try:
            predicted = np.array([float(dT_func(xv, *c_vals)) for xv in x_pts])
            return float(np.sum((predicted - f_pts) ** 2))
        except Exception:
            return 1e10

    # Phase 7: Multi-start L-BFGS-B
    rng = np.random.RandomState(seed)
    best_result = None
    best_fun = float("inf")

    for restart_idx in range(n_restarts):
        if restart_idx == 0:
            x0 = np.zeros(len(constants))
        else:
            x0 = rng.uniform(-5.0, 5.0, size=len(constants))

        opt = minimize(
            residual, x0, args=(x_train, f_train), method="L-BFGS-B",
        )

        if opt.success and opt.fun < best_fun:
            best_fun = opt.fun
            best_result = opt

    if best_result is None:
        return None

    # Phase 8: Verify on held-out points
    train_mse = best_result.fun / max(n_train, 1)
    if train_mse > 1e-6:
        return None

    verify_residual = residual(best_result.x, x_verify, f_verify)
    verify_max_err = float("inf")
    try:
        predicted_verify = np.array(
            [float(dT_func(xv, *best_result.x)) for xv in x_verify],
        )
        verify_max_err = float(np.max(np.abs(predicted_verify - f_verify)))
    except Exception:
        return None

    if verify_max_err > 1e-4:
        return None

    # Phase 9: Cascading nsimplify for each constant
    subs = {}
    for sym, val in zip(constants, best_result.x):
        subs[sym] = _rationalize(float(val))

    result_expr = template.subs(subs)
    if sp.simplify(sp.diff(result_expr, var) - integrand) == 0:
        return result_expr
    return None


# ---------------------------------------------------------------------------
# String-based wrappers for safe pickling across process boundaries
# ---------------------------------------------------------------------------
def _symbolic_solve_from_str(
    template_str: str, integrand_str: str, var_str: str, constant_strs: List[str],
) -> Optional[str]:
    """String-based wrapper for safe pickling across process boundaries."""
    template = sp.sympify(template_str)
    integrand = sp.sympify(integrand_str)
    var = sp.Symbol(var_str)
    constants = [sp.Symbol(c) for c in constant_strs]
    result = _symbolic_solve(template, integrand, var, constants)
    return str(result) if result is not None else None


def _numeric_solve_from_str(
    template_str: str, integrand_str: str, var_str: str, constant_strs: List[str],
) -> Optional[str]:
    """String-based wrapper for safe pickling across process boundaries."""
    template = sp.sympify(template_str)
    integrand = sp.sympify(integrand_str)
    var = sp.Symbol(var_str)
    constants = [sp.Symbol(c) for c in constant_strs]
    result = _numeric_solve(template, integrand, var, constants)
    return str(result) if result is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_constants(
    template: sp.Basic,
    integrand: sp.Basic,
    var: sp.Symbol,
    constants: List[sp.Symbol],
    symbolic_timeout: float = 3.0,
    numeric_timeout: float = 10.0,
) -> Optional[sp.Basic]:
    """Solve for undetermined constants in an integral template.

    Args:
        template: Predicted antiderivative with constants C1, C2, ...
        integrand: Original integrand f(x).
        var: Integration variable.
        constants: List of constant symbols to solve for.
        symbolic_timeout: Timeout for symbolic solve (seconds).
        numeric_timeout: Timeout for numeric solve (seconds).

    Returns:
        The template with constants determined, or None if solving failed.
    """
    template_str = str(template)
    integrand_str = str(integrand)
    var_str = str(var)
    constant_strs = [str(c) for c in constants]

    pool = _get_pool()

    # Step 1: Symbolic solve
    try:
        future = pool.submit(
            _symbolic_solve_from_str, template_str, integrand_str,
            var_str, constant_strs,
        )
        result_str = future.result(timeout=symbolic_timeout)
        if result_str is not None:
            return sp.sympify(result_str)
    except FuturesTimeoutError:
        logger.debug("symbolic_solve_timeout", timeout=symbolic_timeout)
    except Exception:
        logger.debug("symbolic_solve_error", exc_info=True)

    # Step 2: Numeric fallback
    if not constants:
        return None

    try:
        future = pool.submit(
            _numeric_solve_from_str, template_str, integrand_str,
            var_str, constant_strs,
        )
        result_str = future.result(timeout=numeric_timeout)
        if result_str is not None:
            return sp.sympify(result_str)
    except FuturesTimeoutError:
        logger.debug("numeric_solve_timeout", timeout=numeric_timeout)
    except Exception:
        logger.debug("numeric_solve_error", exc_info=True)

    return None
