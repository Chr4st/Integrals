"""Structural feature extraction from prefix-notation expressions.

Produces a 344-dimensional feature vector:
  264  depth-encoded function heads (33 types x 8 bins)
   16  variable role features
   40  special function signatures (8 bins x 5 types)
   24  task + complexity features
"""

from __future__ import annotations

import numpy as np

from neurips.data.prefix import PrefixNode, parse_prefix, tokenize_prefix
from neurips.data.vocab import FUNCTION_NAMES, TASK_TOKENS

_N_FUNC_TYPES: int = len(FUNCTION_NAMES)  # 33
_N_DEPTH_BINS: int = 8
FEATURE_DIM: int = _N_FUNC_TYPES * _N_DEPTH_BINS + 16 + 40 + 24  # 344

_FUNC_INDEX: dict[str, int] = {name: i for i, name in enumerate(FUNCTION_NAMES)}

_SIGNATURE_TYPES: list[str] = [
    "gaussian_decay", "oscillatory", "rational",
    "algebraic_singularity", "exponential_growth",
]

# Heuristic: which functions signal each signature type
_SIG_INDICATORS: dict[str, frozenset[str]] = {
    "gaussian_decay": frozenset(["exp", "erf"]),
    "oscillatory": frozenset(["sin", "cos", "tan", "FresnelS", "FresnelC"]),
    "rational": frozenset(["div", "log", "polylog"]),
    "algebraic_singularity": frozenset(["sqrt", "pow"]),
    "exponential_growth": frozenset(["exp", "sinh", "cosh"]),
}

_TASK_TYPES: list[str] = sorted(TASK_TOKENS.keys())


def _depth_bin(depth: int) -> int:
    """Map depth to one of 8 bins."""
    return min(depth // 2, _N_DEPTH_BINS - 1)


def _func_head_features(nodes: list[PrefixNode]) -> np.ndarray:
    """264-dim: 33 function types x 8 depth bins."""
    out = np.zeros(_N_FUNC_TYPES * _N_DEPTH_BINS, dtype=np.float32)
    for node in nodes:
        idx = _FUNC_INDEX.get(node.token)
        if idx is not None:
            b = _depth_bin(node.depth)
            out[idx * _N_DEPTH_BINS + b] += 1.0
    return out


def _variable_role_features(
    nodes: list[PrefixNode], int_var: str,
) -> np.ndarray:
    """16-dim: variable presence, integration target, depth stats."""
    out = np.zeros(16, dtype=np.float32)
    all_vars = ("x", "y", "z", "w", "t")

    # Single pass: collect present vars and int_var depths
    present_vars: set[str] = set()
    int_var_depths: list[int] = []
    for n in nodes:
        if n.token in all_vars:
            present_vars.add(n.token)
        if n.token == int_var:
            int_var_depths.append(n.depth)

    for i, v in enumerate(all_vars):
        if v in present_vars:
            out[i] = 1.0
        if v == int_var:
            out[5 + i] = 1.0

    total = len(nodes) if nodes else 1
    out[10] = len(int_var_depths) / total

    if int_var_depths:
        out[11] = min(int_var_depths) / 20.0
        out[12] = max(int_var_depths) / 20.0
        out[13] = (sum(int_var_depths) / len(int_var_depths)) / 20.0
    # out[14], out[15] reserved (padding)
    return out


def _signature_features(nodes: list[PrefixNode]) -> np.ndarray:
    """40-dim: 8 bins x 5 signature types."""
    out = np.zeros(40, dtype=np.float32)
    for sig_idx, sig_name in enumerate(_SIGNATURE_TYPES):
        indicators = _SIG_INDICATORS[sig_name]
        for node in nodes:
            if node.token in indicators:
                b = _depth_bin(node.depth)
                out[sig_idx * _N_DEPTH_BINS + b] += 1.0
    return out


def _task_complexity_features(
    nodes: list[PrefixNode],
    task: str = "indef",
    bound_type: str = "none",
    param_count: int = 0,
) -> np.ndarray:
    """24-dim: task one-hot, complexity scalars, bound type, reserved."""
    out = np.zeros(24, dtype=np.float32)
    # Task one-hot (4)
    for i, t in enumerate(_TASK_TYPES):
        if t == task:
            out[i] = 1.0
    # Complexity scalars
    out[4] = min(len(nodes) / 50.0, 1.0)  # node count normalized
    max_d = max((n.depth for n in nodes), default=0)
    out[5] = min(max_d / 15.0, 1.0)  # max depth normalized
    out[6] = min(param_count / 5.0, 1.0)

    # Bound type one-hot (4): none, finite, semi_inf, inf
    bound_types = ["none", "finite", "semi_inf", "inf"]
    for i, bt in enumerate(bound_types):
        if bt == bound_type:
            out[7 + i] = 1.0
    # out[11:24] reserved
    return out


def extract_features(
    prefix_str: str,
    int_var: str,
    task: str = "indef",
    bound_type: str = "none",
    param_count: int = 0,
) -> np.ndarray:
    """Extract a 344-dimensional feature vector from a prefix expression.

    Returns shape (FEATURE_DIM,) float32 array.
    """
    tokens = tokenize_prefix(prefix_str)
    nodes = parse_prefix(tokens)

    parts = [
        _func_head_features(nodes),        # 264
        _variable_role_features(nodes, int_var),  # 16
        _signature_features(nodes),         # 40
        _task_complexity_features(nodes, task, bound_type, param_count),  # 24
    ]
    result = np.concatenate(parts)
    if result.shape != (FEATURE_DIM,):
        raise ValueError(f"Expected {FEATURE_DIM}-dim, got {result.shape}")
    return result
