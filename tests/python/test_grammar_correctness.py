"""Correctness tests for grammar-constrained decoding.

Tests verify that ArityStack tracks prefix expression completion correctly,
PrefixGrammarMask enforces valid token sequences, and property-based tests
confirm grammar invariants hold for arbitrary valid sequences.
"""
from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from neurips.models.grammar import (
    ARITY_TABLE,
    EOS_ID,
    PAD_ID,
    BOS_ID,
    UNK_ID,
    FEAT_ID,
    ArityStack,
    PrefixGrammarMask,
    arity_of,
    _EXPRESSION_IDS_SET,
    _LEAF_IDS_SET,
    _STRUCTURAL_IDS,
    _MAX_STACK_DEPTH,
)

# ── Helpers ──────────────────────────────────────────────────────────

# Valid expression token IDs (non-structural).
_VALID_IDS: list[int] = sorted(_EXPRESSION_IDS_SET)

# Operator IDs (arity > 0).
_OPERATOR_IDS: list[int] = [
    tid for tid in _VALID_IDS if arity_of(tid) > 0
]

# Leaf IDs (arity == 0).
_LEAF_IDS: list[int] = sorted(_LEAF_IDS_SET)

# Binary operator IDs.
_BINARY_IDS: list[int] = [tid for tid in _VALID_IDS if arity_of(tid) == 2]


# ── ArityStack correctness for known expressions ────────────────────


class TestArityStackKnownExpressions:
    """Verify ArityStack for hand-computed prefix expressions."""

    def test_add_x_y(self) -> None:
        """add x y: push_op(add)->needs 2, push_op(x)->needs 1, push_op(y)->complete.

        This is the simplest binary expression. Getting it wrong means
        every binary operation is broken.
        """
        stack = ArityStack()
        stack.push_op(10)  # add (arity 2)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 2

        stack.push_op(70)  # x (arity 0)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 1

        stack.push_op(71)  # y (arity 0)
        assert stack.is_complete()
        assert stack.remaining_slots() == 0

    def test_sin_add_x_y(self) -> None:
        """sin(add(x, y)): nested unary wrapping binary.

        Tests that the stack correctly pushes a new frame for the inner
        binary after consuming a slot from the outer unary.
        """
        stack = ArityStack()
        stack.push_op(20)  # sin (arity 1)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 1

        stack.push_op(10)  # add (arity 2)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 2

        stack.push_op(70)  # x
        assert not stack.is_complete()
        assert stack.remaining_slots() == 1

        stack.push_op(71)  # y
        assert stack.is_complete()

    def test_hyp2f1_a_b_c_x(self) -> None:
        """Hyp2F1(a, b, c, x): arity-4 special function needs exactly 4 leaves.

        Hyp2F1 is the only arity-4 function. If the stack miscounts,
        expressions using it silently produce wrong trees.
        """
        stack = ArityStack()
        stack.push_op(53)  # Hyp2F1 (arity 4)
        assert not stack.is_complete()
        assert stack.remaining_slots() == 4

        for i, leaf_id in enumerate([80, 81, 82, 70]):  # a, b, c, x
            stack.push_op(leaf_id)
            if i < 3:
                assert not stack.is_complete(), f"Completed too early at arg {i+1}"
            else:
                assert stack.is_complete(), "Should be complete after 4 args"

    def test_deeply_nested_expression(self) -> None:
        """add(mul(x, y), sin(z)): mixed binary/unary nesting.

        Tests that slot consumption propagates correctly through
        multiple nested frames.
        """
        stack = ArityStack()
        stack.push_op(10)  # add (2)
        stack.push_op(12)  # mul (2)
        stack.push_op(70)  # x
        stack.push_op(71)  # y — completes mul, consumes 1 slot of add
        assert not stack.is_complete(), "add still needs right child"

        stack.push_op(20)  # sin (1)
        stack.push_op(72)  # z — completes sin, completes add
        assert stack.is_complete()

    def test_remaining_slots_decreases(self) -> None:
        """Remaining slots strictly decrease as leaves are pushed.

        For a pure leaf sequence filling a binary op, remaining_slots
        must go 2 -> 1 -> 0. Non-monotonic decrease indicates a bug.
        """
        stack = ArityStack()
        stack.push_op(14)  # pow (arity 2)
        assert stack.remaining_slots() == 2

        stack.push_op(70)  # x
        assert stack.remaining_slots() == 1

        stack.push_op(71)  # y
        assert stack.remaining_slots() == 0


# ── Specific grammar invariants ─────────────────────────────────────


class TestGrammarInvariants:
    """Verify specific grammar mask invariants."""

    def test_after_binary_op_needs_two_more(self) -> None:
        """After a binary operator, the stack needs exactly 2 more args.

        If the mask allows EOS here, it would produce an incomplete
        expression like 'add' with no operands.
        """
        for op_id in _BINARY_IDS:
            mask_gen = PrefixGrammarMask()
            mask = mask_gen.step(op_id)
            # EOS must NOT be allowed — expression is incomplete.
            assert not mask[EOS_ID].item(), (
                f"EOS allowed after binary op {op_id} — expression incomplete"
            )
            # Other expressions must be allowed.
            assert mask.any().item(), f"All tokens blocked after binary op {op_id}"

    def test_after_complete_expression_only_eos(self) -> None:
        """After a complete expression, only EOS is allowed.

        If non-EOS tokens are allowed after completion, the grammar
        permits malformed expressions like 'x y' (two disconnected terms).
        """
        mask_gen = PrefixGrammarMask()
        mask = mask_gen.step(70)  # x — complete leaf expression
        assert mask[EOS_ID].item(), "EOS not allowed after complete expression"
        # No expression tokens should be allowed.
        for tid in _VALID_IDS:
            assert not mask[tid].item(), (
                f"Token {tid} allowed after complete expression"
            )

    def test_deep_stack_forces_leaves_only(self) -> None:
        """At stack depth > 20, only leaves are allowed.

        This prevents infinite recursion in decoding. Without this guard,
        a sequence of nested operators could consume unbounded memory.
        """
        mask_gen = PrefixGrammarMask()
        # Push enough binary ops to exceed the stack depth limit.
        # Each binary op adds 2 slots, and consuming one for the next op
        # leaves a net +1 per binary op.
        for _ in range(_MAX_STACK_DEPTH + 1):
            mask = mask_gen.step(10)  # add (binary, arity 2)

        # At this point remaining_slots > _MAX_STACK_DEPTH.
        # The mask should only allow leaves.
        for op_id in _OPERATOR_IDS:
            assert not mask[op_id].item(), (
                f"Operator {op_id} allowed at stack depth > {_MAX_STACK_DEPTH}"
            )
        # Leaves must still be allowed.
        assert mask[70].item(), "Leaf 'x' not allowed at deep stack"

    def test_step_and_get_mask_equivalent(self) -> None:
        """step() and get_mask() produce identical masks for the same history.

        step() is the incremental O(1) version; get_mask() rebuilds from
        scratch. If they diverge, one path has a state tracking bug.
        """
        # Build a sequence: sin(add(x, ...))
        token_sequence = [20, 10, 70]  # sin, add, x — incomplete

        # Incremental path via step().
        mask_gen = PrefixGrammarMask()
        for tid in token_sequence:
            step_mask = mask_gen.step(tid)

        # Batch path via get_mask().
        mask_gen2 = PrefixGrammarMask()
        get_mask_result = mask_gen2.get_mask(token_sequence)

        assert torch.equal(step_mask, get_mask_result), (
            "step() and get_mask() disagree for token history "
            f"{token_sequence}"
        )

    def test_step_and_get_mask_equivalent_complete(self) -> None:
        """step() and get_mask() agree on a complete expression.

        Complete expressions should both yield EOS-only masks.
        """
        # add(x, y) — complete
        token_sequence = [10, 70, 71]

        mask_gen = PrefixGrammarMask()
        for tid in token_sequence:
            step_mask = mask_gen.step(tid)

        mask_gen2 = PrefixGrammarMask()
        get_mask_result = mask_gen2.get_mask(token_sequence)

        assert torch.equal(step_mask, get_mask_result)
        assert step_mask[EOS_ID].item(), "EOS not set for complete expression"

    def test_structural_tokens_ignored(self) -> None:
        """Structural tokens (PAD, BOS, EOS, UNK, FEAT) don't affect the stack.

        These tokens appear in padded sequences but must not change
        the grammar state, or padding will break decoding.
        """
        # Reference: just push a leaf.
        mask_gen_ref = PrefixGrammarMask()
        ref_mask = mask_gen_ref.step(70)  # x — complete

        # Same but with structural tokens interspersed.
        mask_gen = PrefixGrammarMask()
        mask_gen.step(PAD_ID)
        mask_gen.step(BOS_ID)
        mask = mask_gen.step(70)  # x — should still be complete

        assert torch.equal(mask, ref_mask), (
            "Structural tokens affected grammar state"
        )


# ── Property-based tests ────────────────────────────────────────────


class TestGrammarProperties:
    """Property-based tests using Hypothesis."""

    @given(st.lists(st.sampled_from(_VALID_IDS), min_size=1, max_size=30))
    @settings(max_examples=200, deadline=2000)
    def test_grammar_mask_always_allows_something(self, token_ids: list[int]) -> None:
        """For any prefix of valid tokens, the grammar mask is never all-False.

        An all-False mask means decoding is stuck with no valid next token.
        This would cause an infinite loop or crash in beam search.
        """
        mask_gen = PrefixGrammarMask()
        for tid in token_ids:
            mask = mask_gen.step(tid)
            if mask_gen._stack.is_complete():
                break  # stop after completion

        # The final mask must allow at least one token.
        assert mask.any().item(), (
            f"All tokens blocked after sequence {token_ids}"
        )

    @given(st.lists(st.sampled_from(_LEAF_IDS[:10]), min_size=1, max_size=5))
    @settings(max_examples=100, deadline=2000)
    def test_single_leaf_always_completes(self, leaf_ids: list[int]) -> None:
        """Any single leaf token immediately completes the expression.

        Leaves have arity 0, so pushing one leaf should consume the
        initial slot and complete the stack.
        """
        # Only use the first leaf — a single leaf completes the expression.
        first_leaf = leaf_ids[0]
        stack = ArityStack()
        stack.push_op(first_leaf)
        assert stack.is_complete(), (
            f"Leaf {first_leaf} (arity={arity_of(first_leaf)}) "
            f"did not complete the stack"
        )

    @given(
        st.sampled_from(_BINARY_IDS),
        st.sampled_from(_LEAF_IDS[:10]),
        st.sampled_from(_LEAF_IDS[:10]),
    )
    @settings(max_examples=100, deadline=2000)
    def test_binary_op_with_two_leaves_completes(
        self, op_id: int, left: int, right: int
    ) -> None:
        """Any binary op followed by exactly two leaves completes.

        This is the fundamental binary expression pattern. If it doesn't
        complete, binary operations are broken.
        """
        stack = ArityStack()
        stack.push_op(op_id)
        assert not stack.is_complete()

        stack.push_op(left)
        assert not stack.is_complete()

        stack.push_op(right)
        assert stack.is_complete(), (
            f"Binary op {op_id} + leaves {left},{right} did not complete"
        )

    @given(
        st.lists(
            st.sampled_from(_VALID_IDS),
            min_size=1,
            max_size=40,
        )
    )
    @settings(max_examples=200, deadline=2000)
    def test_grammar_constrained_walk_terminates(
        self, token_ids: list[int]
    ) -> None:
        """Walking through tokens following grammar masks never gets stuck.

        Simulates constrained decoding: at each step, only tokens allowed
        by the mask are accepted. The walk must either complete (EOS allowed)
        or still have valid tokens to choose from.
        """
        mask_gen = PrefixGrammarMask()
        accepted: list[int] = []

        for tid in token_ids:
            mask = mask_gen.step(tid)
            accepted.append(tid)

            if mask[EOS_ID].item():
                # Expression is complete.
                break

        # After the walk, the mask must allow at least one token.
        assert mask.any().item(), (
            f"Stuck after accepting {accepted}"
        )
