"""Token vocabulary for prefix-notation integral sequences.

Two vocabularies:
1. FEATURE_VOCAB: 76 function heads used by the depth feature extractor.
   Index order defines the feature vector layout.
2. SEQ_VOCAB: Full sequence vocabulary for tokenizer encode/decode.
   Includes special tokens (PAD, BOS, EOS, UNK), operators, INT+/INT-,
   constant placeholders C1..C20, variable 'x', and number tokens.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List

# --- Feature extractor vocabulary (76 entries, order matters) ---
# Phase 12: Replaced "powerExpVar" with "Exp" (variable in exponent) and "Tower" (var in both)
FEATURE_HEADS: List[str] = [
    "Abs", "acos", "acosh", "acot", "acoth", "acsc", "acsch",
    "asec", "asech", "asin", "asinh", "atan", "atanh",
    "cos", "cosh", "CoshIntegral", "CosIntegral", "cot", "coth",
    "csc", "csch", "cbrt", "sin", "sinh", "SinhIntegral", "SinIntegral",
    "exp", "expint", "Ei",
    "fresnelc", "fresnels", "gamma", "lowergamma", "uppergamma",
    "hyper", "appellf1", "log", "loggamma", "Li",
    "Mul", "Add", "Pow", "polygamma", "polylog",
    "re", "im", "sign", "sec", "sech", "sinc", "sqrt", "tan", "tanh",
    "besselj", "bessely", "besseli", "besselk",
    "legendre", "hermite", "laguerre", "chebyshevt", "chebyshevu",
    "zeta", "dirichlet_eta", "Piecewise", "Heaviside", "DiracDelta",
    "erf", "erfc", "erfi", "erfinv",
    "Integer", "Rational", "Float", "Exp", "Tower",
]

FEATURE_HEAD_TO_INDEX: Dict[str, int] = {h: i for i, h in enumerate(FEATURE_HEADS)}
NUM_FEATURE_HEADS: int = len(FEATURE_HEADS)

# Phase 13: Logarithmic depth binning (8 bins instead of linear 5)
# Bin mapping: 0→d1, 1→d2, 2→d3, 3→d4, 4→d5-6, 5→d7-9, 6→d10-15, 7→d16+
NUM_DEPTH_BINS: int = 8

# Upper bound (inclusive) for each bin; last bin captures everything above
_DEPTH_BIN_UPPER = (1, 2, 3, 4, 6, 9, 15)  # 7 boundaries → 8 bins


def depth_to_bin(depth: int) -> int:
    """Map raw depth to logarithmic bin index.

    Depths 1-4 get their own bin (preserving resolution at shallow depths).
    Deeper levels are grouped: 5-6, 7-9, 10-15, 16+.
    """
    for i, upper in enumerate(_DEPTH_BIN_UPPER):
        if depth <= upper:
            return i
    return NUM_DEPTH_BINS - 1


FEATURE_DIM: int = NUM_FEATURE_HEADS * NUM_DEPTH_BINS

# --- Sequence vocabulary ---
# Special tokens (indices 0-3)
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Integer sign tokens
INT_POS = "INT+"
INT_NEG = "INT-"

# Constant placeholders
CONSTANT_TOKENS: List[str] = [f"C{i}" for i in range(1, 21)]  # C1..C20

# Operator/function tokens (used in prefix sequences)
OPERATOR_TOKENS: List[str] = [
    "Add", "Mul", "Pow", "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "exp", "log", "sqrt", "cbrt", "Abs", "sign", "sinc",
    "erf", "erfc", "erfi", "erfinv",
    "gamma", "lowergamma", "uppergamma", "loggamma", "polygamma",
    "besselj", "bessely", "besseli", "besselk",
    "legendre", "hermite", "laguerre", "chebyshevt", "chebyshevu",
    "zeta", "dirichlet_eta", "polylog",
    "Ei", "expint", "Li",
    "SinIntegral", "CosIntegral", "SinhIntegral", "CoshIntegral",
    "fresnelc", "fresnels",
    "hyper", "appellf1",
    "Piecewise", "Heaviside", "DiracDelta",
    "re", "im",
    "Rational", "Float",
    "x",  # integration variable
    "pi", "E", "I",  # symbolic constants
    "Neg",  # unary negation shorthand
]

# Phase 14: Base-100 number tokens (used after INT+/INT-)
# Tokens "0"-"99" for digit-pair encoding. E.g., 42→["INT+","42"], 123456→["INT+","12","34","56"]
NUMBER_TOKENS: List[str] = [str(i) for i in range(100)]

# Build full vocabulary list (order defines indices)
SEQ_TOKENS: List[str] = (
    SPECIAL_TOKENS
    + [INT_POS, INT_NEG]
    + CONSTANT_TOKENS
    + OPERATOR_TOKENS
    + NUMBER_TOKENS
)

SEQ_TOKEN_TO_INDEX: Dict[str, int] = {t: i for i, t in enumerate(SEQ_TOKENS)}
SEQ_VOCAB_SIZE: int = len(SEQ_TOKENS)

# Convenience index lookups
PAD_INDEX: int = SEQ_TOKEN_TO_INDEX[PAD_TOKEN]
BOS_INDEX: int = SEQ_TOKEN_TO_INDEX[BOS_TOKEN]
EOS_INDEX: int = SEQ_TOKEN_TO_INDEX[EOS_TOKEN]
UNK_INDEX: int = SEQ_TOKEN_TO_INDEX[UNK_TOKEN]

MAX_INPUT_LEN: int = 384
MAX_OUTPUT_LEN: int = 256


def feature_schema_hash() -> str:
    """SHA256 hash of the feature schema (head names + indices + depth config).

    Changes if FEATURE_HEADS order, contents, or NUM_DEPTH_BINS change.
    Used to detect feature schema mismatches between checkpoints and code.
    """
    schema_str = "|".join(
        f"{i}:{h}" for i, h in enumerate(FEATURE_HEADS)
    ) + f"|NUM_DEPTH_BINS={NUM_DEPTH_BINS}"
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
