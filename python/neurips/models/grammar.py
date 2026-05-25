"""Grammar-constrained decoding for prefix-notation expressions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import torch

# ── Arity table ──────────────────────────────────────────────────────
# Maps token id → number of children.
#   0 = leaf (numbers, variables, constants)
#   1 = unary operator (sin, cos, exp, log, sqrt, neg, …)
#   2 = binary operator (add, mul, pow, sub, div, …)
#   4 = special (Hyp2F1)
# Tokens not in the table are treated as leaves (arity 0).

# Reserve low ids for structural tokens.
PAD_ID: Final[int] = 0
BOS_ID: Final[int] = 1
EOS_ID: Final[int] = 2
UNK_ID: Final[int] = 3   # [UNK] token
FEAT_ID: Final[int] = 4  # [FEAT] injection token

ARITY_TABLE: dict[int, int] = {
    PAD_ID: -1,   # never valid mid-expression
    BOS_ID: -1,
    EOS_ID: -1,
    UNK_ID: -1,
    FEAT_ID: -1,
    # Binary (arity 2) — matches vocab.py IDs
    10: 2, 11: 2, 12: 2, 13: 2, 14: 2,  # add sub mul div pow
    # Unary (arity 1) — neg=15
    15: 1,  # neg
    # Elementary functions (arity 1) — IDs 20-31
    20: 1, 21: 1, 22: 1, 23: 1, 24: 1, 25: 1,  # sin cos tan exp log sqrt
    26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1,  # asin acos atan sinh cosh tanh
    # Special functions (unary, arity 1) — IDs 40-52
    40: 1, 41: 1, 42: 1, 43: 1, 44: 1,  # erf Ei Si Ci Li
    47: 1, 48: 1,  # EllipticK EllipticE
    49: 1, 50: 1, 51: 1, 52: 1,  # Gamma digamma FresnelS FresnelC
    # Special functions (binary, arity 2) — IDs 45-46, 54
    45: 2, 46: 2, 54: 2,  # BesselJ BesselY polylog
    # Special (arity 4) — ID 53
    53: 4,  # Hyp2F1
    # Variables (70-74), params (80-86), constants (90-93): arity 0
    # Number tokens (100-201): arity 0
    # Anything not listed defaults to 0 (leaf).
}

# Pre-computed sets for fast lookup.
_STRUCTURAL_IDS: Final[frozenset[int]] = frozenset({PAD_ID, BOS_ID, EOS_ID, UNK_ID, FEAT_ID})
_EXPRESSION_IDS_SET: Final[frozenset[int]] = frozenset(
    i for i in range(256) if i not in _STRUCTURAL_IDS
)
_LEAF_IDS_SET: Final[frozenset[int]] = frozenset(
    i for i in _EXPRESSION_IDS_SET if ARITY_TABLE.get(i, 0) == 0
)

# Pre-computed boolean tensors for O(1) mask construction.
_EXPRESSION_MASK: torch.BoolTensor = torch.zeros(256, dtype=torch.bool)
_LEAF_MASK: torch.BoolTensor = torch.zeros(256, dtype=torch.bool)
_EXPRESSION_MASK[list(_EXPRESSION_IDS_SET)] = True
_LEAF_MASK[list(_LEAF_IDS_SET)] = True


def arity_of(token_id: int) -> int:
    """Return the arity of *token_id*, defaulting to 0 (leaf)."""
    return ARITY_TABLE.get(token_id, 0)


# ── Arity stack ──────────────────────────────────────────────────────

@dataclass
class ArityStack:
    """Track remaining argument slots during prefix-notation parsing."""

    _slots: list[int] = field(default_factory=lambda: [1])

    def push_op(self, token_id: int) -> None:
        """Consume one slot, then push *arity_of(token_id)* new slots."""
        if self._slots:
            self._slots[-1] -= 1
            # Pop completed frames.
            while self._slots and self._slots[-1] == 0:
                self._slots.pop()
        a = arity_of(token_id)
        if a > 0:
            self._slots.append(a)

    def is_complete(self) -> bool:
        return len(self._slots) == 0

    def remaining_slots(self) -> int:
        return sum(self._slots)


# ── Grammar mask ─────────────────────────────────────────────────────

_MAX_STACK_DEPTH: Final[int] = 20


_EOS_MASK: torch.BoolTensor = torch.zeros(256, dtype=torch.bool)
_EOS_MASK[EOS_ID] = True


def move_grammar_masks_to(device: torch.device) -> None:
    """Move pre-computed grammar masks to *device*. Call once at train start."""
    global _EXPRESSION_MASK, _LEAF_MASK, _EOS_MASK
    _EXPRESSION_MASK = _EXPRESSION_MASK.to(device)
    _LEAF_MASK = _LEAF_MASK.to(device)
    _EOS_MASK = _EOS_MASK.to(device)


class PrefixGrammarMask:
    """Produce a boolean mask over the vocabulary at each decoding step."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size
        self._stack = ArityStack()

    def get_mask(self, generated_ids: list[int]) -> torch.BoolTensor:
        """Return a *vocab_size* bool tensor (True = allowed)."""
        stack = ArityStack()
        for tid in generated_ids:
            if tid in _STRUCTURAL_IDS:
                continue
            stack.push_op(tid)

        if stack.is_complete():
            return _EOS_MASK
        if stack.remaining_slots() > _MAX_STACK_DEPTH:
            return _LEAF_MASK
        return _EXPRESSION_MASK

    def step(self, token_id: int) -> torch.BoolTensor:
        """Incremental mask: push one token, return next mask. O(1)."""
        if token_id not in _STRUCTURAL_IDS:
            self._stack.push_op(token_id)

        if self._stack.is_complete():
            return _EOS_MASK
        if self._stack.remaining_slots() > _MAX_STACK_DEPTH:
            return _LEAF_MASK
        return _EXPRESSION_MASK

    def reset(self) -> None:
        """Reset internal stack for a new candidate."""
        self._stack = ArityStack()
