"""Prefix-notation grammar mask for constrained decoding.

Tracks arity stack during autoregressive generation and returns a boolean
mask of valid next tokens at each step. Prevents syntactically invalid
sequences (wrong arity, dangling operators) from wasting beam/sample slots.

Grammar rules for prefix notation:
  - Binary ops (Add, Mul, Pow, Rational, ...): need 2 operands
  - Unary ops (sin, cos, Neg, ...): need 1 operand
  - INT+/INT- must be followed by at least one number token
  - EOS is valid only when the arity stack is empty (expression complete)
"""
from __future__ import annotations

from typing import FrozenSet, Optional, Set

import torch

from integral_engine.vocabulary import (
    EOS_INDEX,
    INT_NEG,
    INT_POS,
    NUMBER_TOKENS,
    SEQ_TOKEN_TO_INDEX,
    SEQ_VOCAB_SIZE,
)

# Operator arities (how many operands each consumes)
_BINARY_OPS: FrozenSet[str] = frozenset({
    "Add", "Mul", "Pow", "Rational",
    "lowergamma", "uppergamma", "besselj", "bessely",
    "besseli", "besselk", "polygamma", "polylog",
    "expint", "Heaviside",
})

_UNARY_OPS: FrozenSet[str] = frozenset({
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "exp", "log", "sqrt", "cbrt", "Abs", "sign", "sinc",
    "erf", "erfc", "erfi", "erfinv",
    "gamma", "loggamma",
    "Ei", "Li", "DiracDelta",
    "re", "im",
    "SinIntegral", "CosIntegral", "SinhIntegral", "CoshIntegral",
    "fresnelc", "fresnels",
    "zeta",
    "Neg",
})

# Terminals (arity 0): tokens that complete an operand
_TERMINALS: FrozenSet[str] = frozenset({
    "x", "pi", "E", "I",
} | {f"C{i}" for i in range(1, 21)})

# Number tokens (valid after INT+/INT-)
_NUMBER_TOKEN_INDICES: FrozenSet[int] = frozenset(
    SEQ_TOKEN_TO_INDEX[t] for t in NUMBER_TOKENS if t in SEQ_TOKEN_TO_INDEX
)

# Index sets for fast lookup
_BINARY_OP_INDICES: FrozenSet[int] = frozenset(
    SEQ_TOKEN_TO_INDEX[t] for t in _BINARY_OPS if t in SEQ_TOKEN_TO_INDEX
)
_UNARY_OP_INDICES: FrozenSet[int] = frozenset(
    SEQ_TOKEN_TO_INDEX[t] for t in _UNARY_OPS if t in SEQ_TOKEN_TO_INDEX
)
_TERMINAL_INDICES: FrozenSet[int] = frozenset(
    SEQ_TOKEN_TO_INDEX[t] for t in _TERMINALS if t in SEQ_TOKEN_TO_INDEX
)
_INT_POS_INDEX: int = SEQ_TOKEN_TO_INDEX[INT_POS]
_INT_NEG_INDEX: int = SEQ_TOKEN_TO_INDEX[INT_NEG]
_INT_SIGN_INDICES: FrozenSet[int] = frozenset({_INT_POS_INDEX, _INT_NEG_INDEX})

# All valid expression-start tokens (operators, terminals, INT signs)
_EXPR_START_INDICES: FrozenSet[int] = (
    _BINARY_OP_INDICES | _UNARY_OP_INDICES | _TERMINAL_INDICES | _INT_SIGN_INDICES
)


class PrefixGrammarMask:
    """Tracks prefix-notation arity constraints during decoding.

    Usage:
        mask = PrefixGrammarMask(device)
        # After BOS, call step() for each generated token:
        valid_mask = mask.valid_token_mask()
        # ... sample/pick token ...
        mask.step(chosen_token_index)
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Stack tracks how many operands are still needed.
        # Starts at 1: we need one complete expression.
        self._need: int = 1
        # Whether the last token was INT+/INT- (next must be a number)
        self._in_integer: bool = False
        # Whether we've seen at least one number after INT+/INT-
        self._has_number: bool = False

    def step(self, token_idx: int) -> None:
        """Update state after choosing a token."""
        if self._in_integer:
            if token_idx in _NUMBER_TOKEN_INDICES:
                self._has_number = True
                return  # Stay in integer mode, more digits may follow
            else:
                # Exiting integer mode — the integer is complete (a terminal)
                self._in_integer = False
                self._has_number = False
                self._need -= 1
                # Now process the current token normally (fall through)

        if token_idx in _INT_SIGN_INDICES:
            self._in_integer = True
            self._has_number = False
        elif token_idx in _BINARY_OP_INDICES:
            # Binary op: consumes 1 operand slot, creates 2 new ones → net +1
            self._need += 1
        elif token_idx in _UNARY_OP_INDICES:
            # Unary op: consumes 1 operand slot, creates 1 new one → net 0
            pass
        elif token_idx in _TERMINAL_INDICES:
            # Terminal: completes one operand
            self._need -= 1
        # EOS and other tokens don't change the stack

    def valid_token_mask(self) -> torch.Tensor:
        """Return a boolean mask of shape (vocab_size,) where True = valid."""
        mask = torch.zeros(SEQ_VOCAB_SIZE, dtype=torch.bool, device=self.device)

        if self._in_integer:
            # Must have at least one number; after that, numbers or exit
            for idx in _NUMBER_TOKEN_INDICES:
                mask[idx] = True
            if self._has_number:
                # Can also exit integer mode with any valid continuation
                # The step() will handle finishing the integer
                for idx in _EXPR_START_INDICES:
                    mask[idx] = True
                if self._need <= 1:
                    # This integer completing would make need=0
                    mask[EOS_INDEX] = True
            return mask

        if self._need <= 0:
            # Expression is complete — only EOS is valid
            mask[EOS_INDEX] = True
            return mask

        # Need operands: allow any expression-starting token
        for idx in _EXPR_START_INDICES:
            mask[idx] = True

        return mask

    @property
    def is_complete(self) -> bool:
        """Whether the expression is syntactically complete."""
        return self._need <= 0 and not self._in_integer
