"""Tests for grammar-constrained decoding (Phase 9a)."""
import pytest
from neurips.models.grammar import ArityStack, PrefixGrammarMask, ARITY_TABLE


class TestArityStack:
    def test_empty_is_not_complete(self):
        stack = ArityStack()
        # Stack starts expecting one root expression
        assert not stack.is_complete()

    def test_single_leaf(self):
        stack = ArityStack()
        stack.push_op(70)  # x (arity 0)
        assert stack.is_complete()

    def test_unary_incomplete(self):
        stack = ArityStack()
        stack.push_op(20)  # sin (arity 1)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 1

    def test_unary_complete(self):
        stack = ArityStack()
        stack.push_op(20)  # sin
        stack.push_op(70)  # x
        assert stack.is_complete()

    def test_binary_complete(self):
        stack = ArityStack()
        stack.push_op(10)  # add (arity 2)
        stack.push_op(70)  # x
        stack.push_op(71)  # y
        assert stack.is_complete()

    def test_nested(self):
        # sin(add(x, y)) → sin, add, x, y
        stack = ArityStack()
        stack.push_op(20)  # sin
        assert not stack.is_complete()
        stack.push_op(10)  # add
        assert not stack.is_complete()
        stack.push_op(70)  # x
        assert not stack.is_complete()
        stack.push_op(71)  # y
        assert stack.is_complete()


class TestGrammarMask:
    def test_empty_allows_operators_and_leaves(self):
        mask_gen = PrefixGrammarMask()
        mask = mask_gen.get_mask([])
        assert mask[70].item()  # x allowed
        assert mask[10].item()  # add allowed
        assert mask[20].item()  # sin allowed

    def test_complete_only_eos(self):
        mask_gen = PrefixGrammarMask()
        mask = mask_gen.get_mask([70])  # just "x" — complete
        assert mask[2].item()   # EOS allowed
        assert not mask[70].item()  # no more tokens

    def test_never_all_blocked(self):
        mask_gen = PrefixGrammarMask()
        for seq in [[], [10], [10, 70], [20, 10, 70]]:
            mask = mask_gen.get_mask(seq)
            assert mask.any(), f"All blocked for {seq}"
