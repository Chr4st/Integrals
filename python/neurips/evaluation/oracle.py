"""Verification oracle for checking candidate antiderivatives."""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Any

import sympy

from neurips.data.prefix import tokenize_prefix
from neurips.data.vocab import ARITY

# ── Timeout helper ──────────────────────────────────────────────────


@contextmanager
def _timeout(seconds: float):
    """Raise TimeoutError after *seconds* (Unix only, no-op on Windows)."""

    def _handler(signum, frame):
        raise TimeoutError("verification timed out")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ── Prefix string -> SymPy ──────────────────────────────────────────

_SYMPY_MAP: dict[str, Any] = {
    "add": sympy.Add,
    "mul": sympy.Mul,
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
    "pow": sympy.Pow,
    "neg": lambda a: -a,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "erf": sympy.erf,
}


def prefix_to_sympy(prefix_str: str) -> sympy.Expr:
    """Convert a prefix-notation string to a SymPy expression."""
    tokens = tokenize_prefix(prefix_str)
    if not tokens:
        raise ValueError("empty prefix string")
    expr, _ = _parse_recursive(tokens, 0)
    return expr


def _parse_recursive(
    tokens: list[str], idx: int
) -> tuple[sympy.Expr, int]:
    """Recursively build a SymPy expression from prefix tokens."""
    if idx >= len(tokens):
        raise ValueError("unexpected end of tokens")

    tok = tokens[idx]

    # Leaf: variable
    if tok in ("x", "y", "z", "w", "t"):
        return sympy.Symbol(tok), idx + 1

    # Leaf: parameter
    if tok in ("a", "b", "c", "n", "k", "alpha", "beta"):
        return sympy.Symbol(tok), idx + 1

    # Leaf: constant
    if tok == "pi":
        return sympy.pi, idx + 1
    if tok == "euler_gamma":
        return sympy.EulerGamma, idx + 1
    if tok == "inf":
        return sympy.oo, idx + 1

    # Leaf: sign prefix for numbers
    if tok in ("+", "-"):
        num_expr, next_idx = _parse_recursive(tokens, idx + 1)
        if tok == "-":
            return -num_expr, next_idx
        return num_expr, next_idx

    # Leaf: integer
    try:
        return sympy.Integer(int(tok)), idx + 1
    except ValueError:
        pass

    # Operator
    arity = ARITY.get(tok, 0)
    if arity == 0:
        return sympy.Symbol(tok), idx + 1

    args: list[sympy.Expr] = []
    pos = idx + 1
    for _ in range(arity):
        arg, pos = _parse_recursive(tokens, pos)
        args.append(arg)

    fn = _SYMPY_MAP.get(tok)
    if fn is not None:
        return fn(*args), pos

    # Fallback: treat as a SymPy Function
    return sympy.Function(tok)(*args), pos
