"""Tests for prefix-notation grammar mask."""
import torch
import pytest

from integral_engine.grammar_mask import (
    PrefixGrammarMask,
    _BINARY_OP_INDICES,
    _EXPR_START_INDICES,
    _INT_POS_INDEX,
    _NUMBER_TOKEN_INDICES,
    _TERMINAL_INDICES,
    _UNARY_OP_INDICES,
)
from integral_engine.vocabulary import (
    EOS_INDEX,
    SEQ_TOKEN_TO_INDEX,
    SEQ_VOCAB_SIZE,
)


class TestPrefixGrammarMask:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_initial_state_allows_expression_start(self, device):
        mask = PrefixGrammarMask(device)
        valid = mask.valid_token_mask()
        # Should allow operators, terminals, INT+/-
        for idx in _EXPR_START_INDICES:
            assert valid[idx], f"Token index {idx} should be valid at start"
        # Should not allow EOS (expression not complete)
        assert not valid[EOS_INDEX]

    def test_terminal_completes_expression(self, device):
        mask = PrefixGrammarMask(device)
        x_idx = SEQ_TOKEN_TO_INDEX["x"]
        mask.step(x_idx)
        valid = mask.valid_token_mask()
        # After a single terminal, expression is complete → only EOS valid
        assert valid[EOS_INDEX]

    def test_unary_op_needs_one_operand(self, device):
        mask = PrefixGrammarMask(device)
        sin_idx = SEQ_TOKEN_TO_INDEX["sin"]
        mask.step(sin_idx)
        # Still need 1 operand
        assert not mask.is_complete
        valid = mask.valid_token_mask()
        assert not valid[EOS_INDEX]
        # Should allow expression-starting tokens
        for idx in _EXPR_START_INDICES:
            assert valid[idx]

    def test_binary_op_needs_two_operands(self, device):
        mask = PrefixGrammarMask(device)
        add_idx = SEQ_TOKEN_TO_INDEX["Add"]
        x_idx = SEQ_TOKEN_TO_INDEX["x"]

        mask.step(add_idx)  # Need 2 operands
        assert not mask.is_complete

        mask.step(x_idx)  # First operand done, need 1 more
        assert not mask.is_complete

        mask.step(x_idx)  # Second operand done
        assert mask.is_complete

    def test_nested_expression(self, device):
        """sin(Add(x, x)) → sin needs 1, Add needs 2, x completes."""
        mask = PrefixGrammarMask(device)
        sin_idx = SEQ_TOKEN_TO_INDEX["sin"]
        add_idx = SEQ_TOKEN_TO_INDEX["Add"]
        x_idx = SEQ_TOKEN_TO_INDEX["x"]

        mask.step(sin_idx)   # need=1
        mask.step(add_idx)   # need=2 (sin's 1 slot replaced by add's 2)
        mask.step(x_idx)     # need=1
        mask.step(x_idx)     # need=0
        assert mask.is_complete

    def test_integer_mode(self, device):
        """INT+ followed by number tokens."""
        mask = PrefixGrammarMask(device)
        mask.step(_INT_POS_INDEX)

        valid = mask.valid_token_mask()
        # Must have at least one number
        for idx in _NUMBER_TOKEN_INDICES:
            assert valid[idx]
        # Can't have EOS yet (no number seen)
        assert not valid[EOS_INDEX]

    def test_integer_with_digits_allows_exit(self, device):
        """INT+ 4 2 → valid integer, can exit."""
        mask = PrefixGrammarMask(device)
        idx_42 = SEQ_TOKEN_TO_INDEX["42"]
        mask.step(_INT_POS_INDEX)
        mask.step(idx_42)  # has_number = True

        valid = mask.valid_token_mask()
        # Should allow more digits or exit
        for idx in _NUMBER_TOKEN_INDICES:
            assert valid[idx]
        # Should allow EOS since integer completes the expression
        assert valid[EOS_INDEX]

    def test_complete_prefix_sequence(self, device):
        """Add(sin(x), INT+ 42) → complete expression."""
        mask = PrefixGrammarMask(device)
        add_idx = SEQ_TOKEN_TO_INDEX["Add"]
        sin_idx = SEQ_TOKEN_TO_INDEX["sin"]
        x_idx = SEQ_TOKEN_TO_INDEX["x"]
        idx_42 = SEQ_TOKEN_TO_INDEX["42"]

        mask.step(add_idx)     # need=2
        mask.step(sin_idx)     # need=2 (sin takes 1 slot, adds 1)
        mask.step(x_idx)       # need=1 (sin's operand done)
        mask.step(_INT_POS_INDEX)  # entering integer
        mask.step(idx_42)      # integer complete

        # Now step with a non-number to exit integer mode
        # Actually, the next valid_token_mask should allow EOS
        valid = mask.valid_token_mask()
        assert valid[EOS_INDEX]

    def test_eos_only_when_complete(self, device):
        """EOS should never be valid when expression is incomplete."""
        mask = PrefixGrammarMask(device)
        add_idx = SEQ_TOKEN_TO_INDEX["Add"]
        x_idx = SEQ_TOKEN_TO_INDEX["x"]

        mask.step(add_idx)  # need=2
        assert not mask.valid_token_mask()[EOS_INDEX]

        mask.step(x_idx)    # need=1
        assert not mask.valid_token_mask()[EOS_INDEX]

        mask.step(x_idx)    # need=0
        assert mask.valid_token_mask()[EOS_INDEX]
