"""Bidirectional prefix-notation tokenizer for SymPy expressions.

Integers encoded as sign+digits: 42 → ["INT+", "4", "2"], -3 → ["INT-", "3"]
"""
from __future__ import annotations

from typing import Dict, List

import sympy as sp

from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    INT_NEG,
    INT_POS,
    MAX_INPUT_LEN,
    NUMBER_TOKENS,
    SEQ_TOKEN_TO_INDEX,
    SEQ_TOKENS,
    UNK_INDEX,
)

# Set of valid number tokens for fast lookup during decoding
_NUMBER_TOKEN_SET = frozenset(NUMBER_TOKENS)

# Unary functions (arity 1)
_UNARY_FUNC_MAP = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "acot": sp.acot,
    "asec": sp.asec, "acsc": sp.acsc,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "coth": sp.coth,
    "sech": sp.sech, "csch": sp.csch,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh, "acoth": sp.acoth,
    "asech": sp.asech, "acsch": sp.acsch,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "cbrt": sp.cbrt,
    "Abs": sp.Abs, "sign": sp.sign, "sinc": sp.sinc,
    "erf": sp.erf, "erfc": sp.erfc, "erfi": sp.erfi, "erfinv": sp.erfinv,
    "gamma": sp.gamma, "loggamma": sp.loggamma,
    "Ei": sp.Ei, "Li": sp.Li,
    "DiracDelta": sp.DiracDelta,
    "re": sp.re, "im": sp.im,
    "SinIntegral": sp.Si, "CosIntegral": sp.Ci,
    "SinhIntegral": sp.Shi, "CoshIntegral": sp.Chi,
    "fresnelc": sp.fresnelc, "fresnels": sp.fresnels,
    "zeta": sp.zeta,
}

# Binary functions (arity 2)
_BINARY_FUNC_MAP = {
    "lowergamma": sp.lowergamma, "uppergamma": sp.uppergamma,
    "besselj": sp.besselj, "bessely": sp.bessely,
    "besseli": sp.besseli, "besselk": sp.besselk,
    "polygamma": sp.polygamma, "polylog": sp.polylog,
    "expint": sp.expint,
    "Heaviside": sp.Heaviside,
}

_FUNC_MAP = {**_UNARY_FUNC_MAP, **_BINARY_FUNC_MAP}
_NARY_OPS = {"Add", "Mul"}


def _int_to_base100(val: int) -> List[str]:
    """Encode a non-negative integer as base-100 digit-pair tokens.

    Right-aligned chunking: 42 → ["42"], 123456 → ["12", "34", "56"].
    """
    s = str(val)
    # Pad to even length for right-aligned pairing
    if len(s) % 2 != 0:
        s = "0" + s
    return [s[i:i + 2].lstrip("0") or "0" for i in range(0, len(s), 2)]


def _renumber_constants(tokens: List[str]) -> List[str]:
    """Renumber constant placeholders by order of first appearance.

    Ensures canonical numbering: first constant seen becomes C1, second C2, etc.
    """
    mapping: Dict[str, str] = {}
    counter = 1
    result: List[str] = []
    for t in tokens:
        if t.startswith("C") and t[1:].isdigit():
            if t not in mapping:
                mapping[t] = f"C{counter}"
                counter += 1
            result.append(mapping[t])
        else:
            result.append(t)
    return result


def _to_prefix_raw(expr: sp.Basic) -> List[str]:
    """Convert a SymPy expression to prefix notation token list (unnumbered)."""
    if isinstance(expr, sp.Symbol):
        return [expr.name]

    if isinstance(expr, sp.Integer):
        val = int(expr)
        if val >= 0:
            return [INT_POS] + _int_to_base100(val)
        else:
            return [INT_NEG] + _int_to_base100(abs(val))

    if isinstance(expr, sp.Rational) and not isinstance(expr, sp.Integer):
        p, q = expr.p, expr.q
        tokens = ["Rational"]
        tokens += [INT_POS] + _int_to_base100(p) if p >= 0 else [INT_NEG] + _int_to_base100(abs(p))
        tokens += [INT_POS] + _int_to_base100(q) if q >= 0 else [INT_NEG] + _int_to_base100(abs(q))
        return tokens

    if isinstance(expr, sp.Float):
        r = sp.nsimplify(expr, rational=True)
        if isinstance(r, sp.Rational):
            return _to_prefix_raw(r)
        return [INT_POS] + _int_to_base100(abs(int(round(float(expr)))))

    if expr is sp.pi:
        return ["pi"]
    if expr is sp.E:
        return ["E"]
    if expr is sp.I:
        return ["I"]

    # Negation: Mul(-1, expr) → Neg(expr)
    if isinstance(expr, sp.Mul) and len(expr.args) == 2:
        a, b = expr.args
        if a == sp.Integer(-1):
            return ["Neg"] + _to_prefix_raw(b)

    # N-ary operations: serialize as nested binary
    type_name = type(expr).__name__
    if type_name in _NARY_OPS:
        args = list(expr.args)
        if len(args) == 1:
            return _to_prefix_raw(args[0])
        result = _to_prefix_raw(args[-1])
        for arg in reversed(args[:-1]):
            result = [type_name] + _to_prefix_raw(arg) + result
        return result

    # Pow
    if isinstance(expr, sp.Pow):
        base, exp = expr.args
        return ["Pow"] + _to_prefix_raw(base) + _to_prefix_raw(exp)

    # Known functions
    func_name = type(expr).__name__
    if func_name in _FUNC_MAP and hasattr(expr, "args"):
        tokens = [func_name]
        for arg in expr.args:
            tokens += _to_prefix_raw(arg)
        return tokens

    # Fallback
    if hasattr(expr, "args") and expr.args:
        tokens = [func_name] if func_name in SEQ_TOKEN_TO_INDEX else []
        for arg in expr.args:
            tokens += _to_prefix_raw(arg)
        return tokens if tokens else [str(expr)]

    return [str(expr)]


def to_prefix(expr: sp.Basic) -> List[str]:
    """Convert a SymPy expression to prefix notation with canonical constant numbering."""
    return _renumber_constants(_to_prefix_raw(expr))


def from_prefix(tokens: List[str]) -> sp.Basic:
    """Convert prefix notation token list back to a SymPy expression."""
    pos = [0]

    def _parse() -> sp.Basic:
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of token sequence")

        token = tokens[pos[0]]
        pos[0] += 1

        # Integer tokens (base-100 digit pairs)
        if token in (INT_POS, INT_NEG):
            chunks: List[int] = []
            while pos[0] < len(tokens) and tokens[pos[0]] in _NUMBER_TOKEN_SET:
                chunks.append(int(tokens[pos[0]]))
                pos[0] += 1
            if not chunks:
                raise ValueError("INT+/INT- not followed by number tokens")
            val = 0
            for chunk in chunks:
                val = val * 100 + chunk
            return sp.Integer(val if token == INT_POS else -val)

        if token == "x":
            return sp.Symbol("x")
        if token == "pi":
            return sp.pi
        if token == "E":
            return sp.E
        if token == "I":
            return sp.I

        # Constant placeholders
        if token.startswith("C") and token[1:].isdigit():
            return sp.Symbol(token)

        if token == "Neg":
            return -_parse()

        if token == "Rational":
            p = _parse()
            q = _parse()
            return sp.Rational(int(p), int(q))

        # N-ary: Add, Mul (serialized as binary)
        if token in _NARY_OPS:
            left = _parse()
            right = _parse()
            return left + right if token == "Add" else left * right

        if token == "Pow":
            base = _parse()
            exp = _parse()
            return base**exp

        # Binary functions
        if token in _BINARY_FUNC_MAP:
            func = _BINARY_FUNC_MAP[token]
            arg1 = _parse()
            arg2 = _parse()
            return func(arg1, arg2)

        # Unary functions
        if token in _UNARY_FUNC_MAP:
            func = _UNARY_FUNC_MAP[token]
            return func(_parse())

        try:
            return sp.Integer(int(token))
        except (ValueError, TypeError):
            pass

        raise ValueError(f"Unknown token: {token}")

    return _parse()


def encode(expr: sp.Basic, max_len: int = MAX_INPUT_LEN) -> List[int]:
    """Convert SymPy expression to index sequence with BOS/EOS."""
    tokens = to_prefix(expr)
    indices = [SEQ_TOKEN_TO_INDEX.get(t, UNK_INDEX) for t in tokens]
    indices = indices[:max_len - 2]
    return [BOS_INDEX] + indices + [EOS_INDEX]


def decode(indices: List[int]) -> sp.Basic:
    """Convert index sequence back to SymPy expression. Strips BOS/EOS."""
    tokens = []
    for idx in indices:
        if idx in (BOS_INDEX, EOS_INDEX):
            continue
        if 0 <= idx < len(SEQ_TOKENS):
            tokens.append(SEQ_TOKENS[idx])
    return from_prefix(tokens)
