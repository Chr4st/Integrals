"""Depth-encoded feature extractor for SymPy expressions.

Produces a FEATURE_DIM-dimensional feature vector encoding which function heads
appear in the expression and at what minimum depth the integration variable
is reachable from each. Uses logarithmic depth binning (8 bins).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import sympy as sp

from integral_engine.vocabulary import (
    FEATURE_DIM,
    FEATURE_HEAD_TO_INDEX,
    NUM_DEPTH_BINS,
    depth_to_bin,
)


def _contains_var(expr: sp.Basic, var: sp.Symbol) -> bool:
    """Check if expr contains var."""
    return var in expr.free_symbols


def _get_head_name(expr: sp.Basic, var: sp.Symbol) -> Optional[str]:
    """Get the feature vocabulary head name for an expression node.

    Returns None for nodes not in the feature vocabulary.
    Phase 12: Three-way power classification:
      - Pow: variable only in base (e.g., x^2)
      - Exp: variable only in exponent (e.g., 2^x, exp(x))
      - Tower: variable in both base and exponent (e.g., x^x)
    """
    if isinstance(expr, sp.Pow):
        base, exponent = expr.args
        base_has_var = _contains_var(base, var)
        exp_has_var = _contains_var(exponent, var)
        if base_has_var and exp_has_var:
            return "Tower"
        if exp_has_var:
            return "Exp"
        return "Pow"

    if isinstance(expr, sp.Integer):
        return "Integer"
    if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
        return "Rational"
    if isinstance(expr, sp.Float):
        return "Float"

    if isinstance(expr, sp.Function):
        name = type(expr).__name__
        # exp(x) with var in argument → Exp (exponential with variable)
        if name == "exp" and _contains_var(expr, var):
            return "Exp"
        if name in FEATURE_HEAD_TO_INDEX:
            return name
        return None

    type_name = type(expr).__name__
    if type_name in FEATURE_HEAD_TO_INDEX:
        return type_name

    return None


def _walk(
    expr: sp.Basic,
    var: sp.Symbol,
    counts: Dict[Tuple[str, int], int],
) -> int:
    """Recursively walk expression tree, computing min depth to var.

    Returns min edges from this node to a leaf containing var, or -1.
    """
    if expr == var:
        return 0

    if not expr.args:
        return -1

    min_child_depth = -1
    for arg in expr.args:
        child_depth = _walk(arg, var, counts)
        if child_depth >= 0:
            if min_child_depth < 0:
                min_child_depth = child_depth
            else:
                min_child_depth = min(min_child_depth, child_depth)

    if min_child_depth >= 0:
        depth_from_here = min_child_depth + 1
        head = _get_head_name(expr, var)
        if head is not None and head in FEATURE_HEAD_TO_INDEX:
            # Phase 13: Logarithmic depth binning
            binned = depth_to_bin(depth_from_here)
            key = (head, binned)
            counts[key] = counts.get(key, 0) + 1

    return min_child_depth + 1 if min_child_depth >= 0 else -1


def extract_features_debug(
    expr: sp.Basic, var: sp.Symbol
) -> Dict[Tuple[str, int], int]:
    """Extract depth features as {(function_head, depth_bin): count} dict."""
    counts: Dict[Tuple[str, int], int] = {}
    _walk(expr, var, counts)
    return counts


def extract_features(expr: sp.Basic, var: sp.Symbol) -> np.ndarray:
    """Extract FEATURE_DIM-dim depth-encoded feature vector from a SymPy expression."""
    counts = extract_features_debug(expr, var)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    for (head, depth_bin), count in counts.items():
        idx = FEATURE_HEAD_TO_INDEX[head]
        vec[idx * NUM_DEPTH_BINS + depth_bin] = count
    return vec
