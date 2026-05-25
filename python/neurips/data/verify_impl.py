"""Verification implementation: validate (integrand, antiderivative) pairs."""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Any

import numpy as np
import sympy
from sympy import Symbol, diff, expand, simplify, trigsimp

from neurips.data.prefix import parse_prefix, tokenize_prefix, tree_depth
from neurips.data.split import _classify_difficulty
from neurips.data.verify import prefix_to_sympy

_DEGENERATE = {sympy.zoo, sympy.nan, sympy.oo, -sympy.oo}


def _symbolic_check(
    f: sympy.Expr, big_f: sympy.Expr, var: Symbol,
) -> tuple[bool, sympy.Expr]:
    """Check if diff(F, var) == f symbolically. Returns (ok, dF/dx)."""
    d = diff(big_f, var)
    residual = simplify(d - f)
    if residual == 0:
        return True, d
    if trigsimp(residual) == 0:
        return True, d
    if expand(residual) == 0:
        return True, d
    return False, d


def _numerical_check(
    f: sympy.Expr, dF: sympy.Expr, var: Symbol, n_points: int = 20,
    min_evaluated: int = 5,
) -> bool:
    """Numerical verification at random points. Expects pre-computed dF/dx."""
    rng = random.Random(42)
    points = [rng.uniform(0.1, 5.0) for _ in range(n_points)]

    evaluated = 0
    for pt in points:
        try:
            val_d = complex(dF.subs(var, pt))
            val_f = complex(f.subs(var, pt))
            if not (np.isfinite(val_d.real) and np.isfinite(val_f.real)):
                continue
            evaluated += 1
            if abs(val_d - val_f) > 1e-8 * max(abs(val_f), 1.0):
                return False
        except (ValueError, TypeError, OverflowError):
            continue
    # Reject if too few points were actually evaluated
    return evaluated >= min_evaluated


def _has_degenerate(expr: sympy.Expr) -> bool:
    """Check for zoo, nan, oo in expression."""
    atoms = expr.atoms()
    return bool(atoms & _DEGENERATE)


def verify_pair(pair: dict[str, Any]) -> dict[str, Any] | None:
    """Verify a single (integrand, antiderivative) pair.

    Returns enriched pair dict or None if invalid.
    """
    try:
        f_prefix = pair["integrand"]
        big_f_prefix = pair["antiderivative"]
        var_name = pair.get("var", "x")

        f = prefix_to_sympy(f_prefix)
        big_f = prefix_to_sympy(big_f_prefix)
        var = Symbol(var_name)

        # Filter degenerate cases
        if f == 0 or _has_degenerate(f) or _has_degenerate(big_f):
            return None
        if diff(big_f, var) == 0 and f != 0:
            return None  # F is constant but f is not zero

        # Symbolic verification (also returns dF/dx for reuse)
        valid, dF = _symbolic_check(f, big_f, var)

        # Numerical fallback (reuses pre-computed dF/dx)
        if not valid:
            valid = _numerical_check(f, dF, var)

        if not valid:
            return None

        # Compute difficulty metadata
        f_tokens = tokenize_prefix(f_prefix)
        f_nodes = parse_prefix(f_tokens)
        depth = tree_depth(f_nodes)
        node_count = len(f_nodes)
        difficulty = _classify_difficulty(depth, node_count)

        return {
            **pair,
            "verified": True,
            "depth": depth,
            "node_count": node_count,
            "difficulty": difficulty,
        }

    except Exception:
        return None


def verify_batch(
    pairs: list[dict[str, Any]],
    n_workers: int = 8,
    timeout: float = 4.0,
) -> list[dict[str, Any]]:
    """Verify pairs in parallel.

    Uses the Rust ``py_verify_batch`` backend when available (~1000x
    faster).  Falls back to SymPy multiprocessing pool otherwise.

    Returns only valid, enriched pairs.
    """
    try:
        return _verify_batch_rust(pairs)
    except ImportError:
        return _verify_batch_sympy(pairs, n_workers, timeout)


def _verify_batch_rust(
    pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fast path: Rust verification + Python-side enrichment."""
    from neurips._core import verify_batch as rust_verify
    from neurips.data.prefix import python_prefix_to_rust

    # Translate prefix format and call Rust.
    rust_triples: list[tuple[str, str, str]] = []
    for p in pairs:
        f_rust = python_prefix_to_rust(p["integrand"])
        big_f_rust = python_prefix_to_rust(p["antiderivative"])
        var = p.get("var", "x")
        rust_triples.append((f_rust, big_f_rust, var))

    valid_flags = rust_verify(rust_triples)

    # Enrich valid pairs with difficulty metadata.
    results: list[dict[str, Any]] = []
    for p, valid in zip(pairs, valid_flags):
        if not valid:
            continue
        try:
            f_tokens = tokenize_prefix(p["integrand"])
            f_nodes = parse_prefix(f_tokens)
            depth = tree_depth(f_nodes)
            node_count = len(f_nodes)
            difficulty = _classify_difficulty(depth, node_count)
            results.append({
                **p,
                "verified": True,
                "depth": depth,
                "node_count": node_count,
                "difficulty": difficulty,
            })
        except Exception:
            continue
    return results


def _verify_batch_sympy(
    pairs: list[dict[str, Any]],
    n_workers: int = 8,
    timeout: float = 4.0,
) -> list[dict[str, Any]]:
    """Fallback: SymPy verification with multiprocessing."""
    results: list[dict[str, Any]] = []

    with mp.Pool(processes=n_workers) as pool:
        async_results = [
            pool.apply_async(verify_pair, (p,)) for p in pairs
        ]
        for ar in async_results:
            try:
                result = ar.get(timeout=timeout)
                if result is not None:
                    results.append(result)
            except mp.TimeoutError:
                continue
            except Exception:
                continue

    return results
