"""Token vocabulary for prefix-notation expression encoding."""

from __future__ import annotations

from typing import Final

# ── Special tokens (0-9) ────────────────────────────────────────────
PAD: Final[int] = 0
SOS: Final[int] = 1
EOS: Final[int] = 2
UNK: Final[int] = 3
FEAT: Final[int] = 4
INDEF: Final[int] = 5
DEF: Final[int] = 6
VAR: Final[int] = 7
PARAM: Final[int] = 8
LOWER: Final[int] = 9

# ── Binary ops (10-15) ─────────────────────────────────────────────
# ── Elementary functions (20-31) ────────────────────────────────────
# ── Special functions (40-54) ───────────────────────────────────────
# ── Variables (70-74), Parameters (80-86), Constants (90-93) ────────
# ── Number encoding (100-201) ───────────────────────────────────────

VOCAB: dict[str, int] = {
    "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3,
    "[FEAT]": 4, "[INDEF]": 5, "[DEF]": 6, "[VAR]": 7,
    "[PARAM]": 8, "[LOWER]": 9,
    "add": 10, "sub": 11, "mul": 12, "div": 13, "pow": 14, "neg": 15,
    "sin": 20, "cos": 21, "tan": 22, "exp": 23, "log": 24, "sqrt": 25,
    "asin": 26, "acos": 27, "atan": 28, "sinh": 29, "cosh": 30, "tanh": 31,
    "erf": 40, "Ei": 41, "Si": 42, "Ci": 43, "Li": 44,
    "BesselJ": 45, "BesselY": 46, "EllipticK": 47, "EllipticE": 48,
    "Gamma": 49, "digamma": 50, "FresnelS": 51, "FresnelC": 52,
    "Hyp2F1": 53, "polylog": 54,
    "x": 70, "y": 71, "z": 72, "w": 73, "t": 74,
    "a": 80, "b": 81, "c": 82, "n": 83, "k": 84, "alpha": 85, "beta": 86,
    "pi": 90, "euler_gamma": 91, "catalan": 92, "inf": 93,
    "+": 100, "-": 101,
}

# Digit tokens: 0-99 map to IDs 102-201
for _d in range(100):
    VOCAB[str(_d)] = 102 + _d

# Reverse lookup: token_id -> token_string
ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}

# Token name -> arity (number of children in prefix tree)
ARITY: dict[str, int] = {
    "add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2,
    "neg": 1,
    "sin": 1, "cos": 1, "tan": 1, "exp": 1, "log": 1, "sqrt": 1,
    "asin": 1, "acos": 1, "atan": 1, "sinh": 1, "cosh": 1, "tanh": 1,
    "erf": 1, "Ei": 1, "Si": 1, "Ci": 1, "Li": 1,
    "BesselJ": 2, "BesselY": 2, "EllipticK": 1, "EllipticE": 1,
    "Gamma": 1, "digamma": 1, "FresnelS": 1, "FresnelC": 1,
    "Hyp2F1": 4, "polylog": 2,
}

# All function/operator names (for feature extraction)
FUNCTION_NAMES: list[str] = sorted(ARITY.keys())

# Task tokens
TASK_TOKENS: dict[str, int] = {
    "indef": INDEF, "def": DEF, "var": VAR, "lower": LOWER,
}

VOCAB_SIZE: Final[int] = 256
