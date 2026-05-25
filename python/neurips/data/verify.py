"""Prefix notation <-> SymPy expression conversion."""

from __future__ import annotations

from typing import Any

import sympy
from sympy import S, Symbol, oo

from neurips.data.prefix import tokenize_prefix
from neurips.data.vocab import ARITY

# ── Prefix <-> SymPy conversion ─────────────────────────────────────

_SYMPY_MAP: dict[str, Any] = {
    "add": sympy.Add, "sub": lambda a, b: a - b,
    "mul": sympy.Mul, "div": lambda a, b: a / b,
    "pow": sympy.Pow, "neg": lambda a: -a,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "exp": sympy.exp, "log": sympy.log, "sqrt": sympy.sqrt,
    "asin": sympy.asin, "acos": sympy.acos, "atan": sympy.atan,
    "sinh": sympy.sinh, "cosh": sympy.cosh, "tanh": sympy.tanh,
    "erf": sympy.erf, "Ei": sympy.Ei, "Si": sympy.Si, "Ci": sympy.Ci,
    "Li": sympy.li,
    "BesselJ": sympy.besselj, "BesselY": sympy.bessely,
    "Gamma": sympy.gamma, "digamma": sympy.digamma,
    "FresnelS": sympy.fresnels, "FresnelC": sympy.fresnelc,
    "EllipticK": sympy.elliptic_k, "EllipticE": sympy.elliptic_e,
    "Hyp2F1": sympy.hyper,
    "polylog": sympy.polylog,
    "pi": sympy.pi, "euler_gamma": sympy.EulerGamma,
    "catalan": sympy.Catalan, "inf": oo,
}

_VAR_NAMES: set[str] = {"x", "y", "z", "w", "t"}
_PARAM_NAMES: set[str] = {"a", "b", "c", "n", "k", "alpha", "beta"}
_SYMBOLS: dict[str, Symbol] = {}


def _sym(name: str) -> Symbol:
    """Get or create a SymPy symbol by name."""
    if name not in _SYMBOLS:
        _SYMBOLS[name] = Symbol(name)
    return _SYMBOLS[name]


def prefix_to_sympy(prefix_str: str) -> sympy.Expr:
    """Parse prefix notation string to SymPy expression."""
    tokens = tokenize_prefix(prefix_str)
    expr, _ = _parse_expr(tokens, 0)
    return expr


def _parse_expr(tokens: list[str], pos: int) -> tuple[sympy.Expr, int]:
    """Recursive descent parser for prefix tokens."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of token sequence")

    tok = tokens[pos]

    if tok in _VAR_NAMES or tok in _PARAM_NAMES:
        return _sym(tok), pos + 1

    if tok in ("pi", "euler_gamma", "catalan", "inf"):
        return _SYMPY_MAP[tok], pos + 1

    # Try numeric literal
    try:
        return sympy.Integer(int(tok)), pos + 1
    except ValueError:
        pass

    arity = ARITY.get(tok)
    if arity is None:
        raise ValueError(f"Unknown token: {tok}")

    func = _SYMPY_MAP[tok]
    args = []
    p = pos + 1
    for _ in range(arity):
        arg, p = _parse_expr(tokens, p)
        args.append(arg)

    if tok == "Hyp2F1":
        return sympy.hyper([args[0], args[1]], [args[2]], args[3]), p

    return func(*args), p


# ── SymPy -> prefix notation ────────────────────────────────────────

_SYMPY_TO_PREFIX: dict[type, str] = {
    sympy.Add: "add", sympy.Mul: "mul", sympy.Pow: "pow",
    sympy.sin: "sin", sympy.cos: "cos", sympy.tan: "tan",
    sympy.exp: "exp", sympy.log: "log", sympy.sqrt: "sqrt",
    sympy.asin: "asin", sympy.acos: "acos", sympy.atan: "atan",
    sympy.sinh: "sinh", sympy.cosh: "cosh", sympy.tanh: "tanh",
    sympy.erf: "erf", sympy.Ei: "Ei", sympy.Si: "Si", sympy.Ci: "Ci",
    sympy.li: "Li", sympy.besselj: "BesselJ", sympy.bessely: "BesselY",
    sympy.gamma: "Gamma", sympy.digamma: "digamma",
    sympy.fresnels: "FresnelS", sympy.fresnelc: "FresnelC",
    sympy.elliptic_k: "EllipticK", sympy.elliptic_e: "EllipticE",
    sympy.polylog: "polylog",
}


def sympy_to_prefix(expr: sympy.Expr) -> str:
    """Convert a SymPy expression to prefix notation string."""
    return " ".join(_expr_to_tokens(expr))


def _expr_to_tokens(expr: sympy.Expr) -> list[str]:
    """Recursively convert SymPy expr to prefix token list."""
    if isinstance(expr, sympy.Symbol):
        return [str(expr)]
    if isinstance(expr, sympy.Integer):
        return [str(int(expr))]
    if isinstance(expr, sympy.Rational) and not isinstance(expr, sympy.Integer):
        return ["div", str(int(expr.p)), str(int(expr.q))]
    if expr is sympy.pi:
        return ["pi"]
    if expr is sympy.EulerGamma:
        return ["euler_gamma"]
    if expr is oo:
        return ["inf"]
    if expr.is_Number:
        return [str(expr)]

    # Negation: -1 * x -> neg x
    if isinstance(expr, sympy.Mul) and len(expr.args) == 2:
        if expr.args[0] == S.NegativeOne:
            return ["neg", *_expr_to_tokens(expr.args[1])]

    prefix_name = _SYMPY_TO_PREFIX.get(type(expr))

    if prefix_name is None:
        # Fallback: use string representation
        return [str(expr)]

    args = expr.args
    # Binary ops with >2 args: left-fold
    if prefix_name in ("add", "mul") and len(args) > 2:
        result = _expr_to_tokens(args[0])
        for arg in args[1:]:
            result = [prefix_name, *result, *_expr_to_tokens(arg)]
        return result

    tokens = [prefix_name]
    for arg in args:
        tokens.extend(_expr_to_tokens(arg))
    return tokens
