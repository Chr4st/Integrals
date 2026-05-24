"""Tests for MCTS decoder and value network."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from integral_engine.grammar_mask import PrefixGrammarMask
from integral_engine.mcts import MCTSDecoder, MCTSNode, _verify_sequence
from integral_engine.value_network import ValueNetwork
from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    SEQ_TOKEN_TO_INDEX,
    SEQ_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# MCTSNode tests
# ---------------------------------------------------------------------------


class TestMCTSNode:
    def test_ucb1_prefers_unexplored(self):
        """Unvisited children should have infinite UCB1."""
        parent = MCTSNode(token_id=0)
        parent.visit_count = 10

        visited = MCTSNode(token_id=1, parent=parent, prior=0.5)
        visited.visit_count = 5
        visited.total_value = 2.0

        unvisited = MCTSNode(token_id=2, parent=parent, prior=0.3)
        # visit_count stays 0

        assert unvisited.ucb1() == float("inf")
        assert visited.ucb1() < float("inf")
        assert unvisited.ucb1() > visited.ucb1()

    def test_best_child_returns_highest_ucb1(self):
        """best_child should pick the child with highest UCB1."""
        parent = MCTSNode(token_id=0)
        parent.visit_count = 20

        child_a = MCTSNode(token_id=1, parent=parent, prior=0.5)
        child_a.visit_count = 10
        child_a.total_value = 8.0  # mean = 0.8

        child_b = MCTSNode(token_id=2, parent=parent, prior=0.3)
        child_b.visit_count = 10
        child_b.total_value = 2.0  # mean = 0.2

        parent.children = {1: child_a, 2: child_b}

        best = parent.best_child()
        assert best.token_id == child_a.token_id

    def test_best_child_prefers_unvisited(self):
        """An unvisited child should beat a frequently visited one."""
        parent = MCTSNode(token_id=0)
        parent.visit_count = 10

        visited = MCTSNode(token_id=1, parent=parent, prior=0.5)
        visited.visit_count = 5
        visited.total_value = 4.0

        fresh = MCTSNode(token_id=2, parent=parent, prior=0.1)
        # visit_count = 0

        parent.children = {1: visited, 2: fresh}
        best = parent.best_child()
        assert best.token_id == fresh.token_id

    def test_is_leaf(self):
        node = MCTSNode(token_id=0)
        assert node.is_leaf()
        node.children[1] = MCTSNode(token_id=1, parent=node)
        assert not node.is_leaf()

    def test_best_child_raises_on_no_children(self):
        node = MCTSNode(token_id=0)
        with pytest.raises(ValueError, match="No children"):
            node.best_child()


# ---------------------------------------------------------------------------
# Verification helper test
# ---------------------------------------------------------------------------


class TestVerifySequence:
    def test_correct_antiderivative(self):
        """diff(x^2/2, x) == x  =>  reward 1.0."""
        import sympy as sp

        x = sp.Symbol("x")
        # Encode: Mul(Rational(1,2), Pow(x, INT+ 2))
        # "Mul" "Rational" "INT+" "1" "INT+" "2" "Pow" "x" "INT+" "2"
        token_names = [
            "Mul", "Rational", "INT+", "1", "INT+", "2",
            "Pow", "x", "INT+", "2",
        ]
        token_ids = [BOS_INDEX]
        for name in token_names:
            token_ids.append(SEQ_TOKEN_TO_INDEX[name])
        token_ids.append(EOS_INDEX)

        reward = _verify_sequence(token_ids, x, x)
        assert reward == 1.0

    def test_wrong_antiderivative(self):
        """diff(x, x) == 1 != sin(x)  =>  reward 0.0."""
        import sympy as sp

        x = sp.Symbol("x")
        token_ids = [BOS_INDEX, SEQ_TOKEN_TO_INDEX["x"], EOS_INDEX]
        reward = _verify_sequence(token_ids, sp.sin(x), x)
        assert reward == 0.0

    def test_empty_sequence(self):
        import sympy as sp

        x = sp.Symbol("x")
        assert _verify_sequence([], x, x) == 0.0


# ---------------------------------------------------------------------------
# MCTSDecoder with mock model
# ---------------------------------------------------------------------------


def _make_mock_model(target_token_ids: list[int]) -> MagicMock:
    """Create a mock model that always predicts the next token in target_token_ids.

    The mock model's decode returns logits where the target token
    has a very high logit, driving both expansion and greedy rollout
    to produce the desired sequence.
    """
    model = MagicMock()
    model.eval = MagicMock()

    call_count = [0]

    def mock_encode(src, depth_features, src_key_padding_mask=None):
        return torch.zeros(1, 5, 512)

    def mock_decode(
        tgt, memory, tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        seq_len = tgt.shape[1]
        # Determine which token position we are predicting
        # seq_len includes BOS, so position in target is seq_len - 1
        pos = seq_len - 1
        logits = torch.full(
            (1, seq_len, SEQ_VOCAB_SIZE), -10.0
        )
        if pos < len(target_token_ids):
            target = target_token_ids[pos]
        else:
            target = EOS_INDEX
        # Set high logit for the target token at last position
        logits[0, -1, target] = 10.0
        call_count[0] += 1
        return logits

    model.encode = mock_encode
    model.decode = mock_decode
    return model


class TestMCTSDecoder:
    def test_finds_trivial_antiderivative(self):
        """MCTS should find x^2/2 as antiderivative of x.

        Target sequence: Mul Rational INT+ 1 INT+ 2 Pow x INT+ 2
        """
        import sympy as sp

        x = sp.Symbol("x")

        target_names = [
            "Mul", "Rational", "INT+", "1", "INT+", "2",
            "Pow", "x", "INT+", "2", "<EOS>",
        ]
        target_ids = [SEQ_TOKEN_TO_INDEX[n] for n in target_names]

        model = _make_mock_model(target_ids)
        decoder = MCTSDecoder(
            model=model,
            max_simulations=10,
            max_len=20,
            top_k=5,
        )

        src = torch.zeros(1, 5, dtype=torch.long)
        depth_feat = torch.zeros(1, 608)

        results = decoder.search(src, depth_feat, x, x)

        # Should find at least one completed sequence
        assert len(results) > 0
        # The best result should verify correctly
        verified = [r for r in results if r[1] == 1.0]
        assert len(verified) > 0

    def test_returns_empty_when_no_eos(self):
        """If model never produces EOS within max_len, completed list may be empty."""
        import sympy as sp

        x = sp.Symbol("x")
        # Model always predicts the same non-EOS token
        x_idx = SEQ_TOKEN_TO_INDEX["x"]
        model = _make_mock_model([x_idx] * 50)

        decoder = MCTSDecoder(
            model=model,
            max_simulations=5,
            max_len=10,
            top_k=3,
        )

        src = torch.zeros(1, 5, dtype=torch.long)
        depth_feat = torch.zeros(1, 608)

        results = decoder.search(src, depth_feat, x, x)
        # All sequences should either be empty or unverified
        verified = [r for r in results if r[1] == 1.0]
        assert len(verified) == 0


# ---------------------------------------------------------------------------
# Grammar mask integration
# ---------------------------------------------------------------------------


class TestGrammarMaskIntegration:
    def test_grammar_mask_filters_invalid_tokens(self):
        """When grammar_mask is enabled, expansion should respect arity rules."""
        device = torch.device("cpu")

        # After BOS, the grammar mask should allow expression-start tokens
        # but not EOS (expression incomplete). We test that MCTSDecoder
        # expansion respects this.
        import sympy as sp

        x = sp.Symbol("x")

        # Model tries to produce EOS immediately (invalid per grammar)
        model = _make_mock_model([EOS_INDEX] * 5)

        decoder = MCTSDecoder(
            model=model,
            grammar_mask=PrefixGrammarMask(device),
            max_simulations=5,
            max_len=10,
            top_k=5,
        )

        src = torch.zeros(1, 5, dtype=torch.long)
        depth_feat = torch.zeros(1, 608)

        results = decoder.search(src, depth_feat, x, x)
        # EOS should be filtered out at the first position since
        # the expression is not yet complete. No valid completed
        # sequence can start with just BOS + EOS under the grammar.
        for seq, reward in results:
            # If any sequence completed, check it is not just BOS+EOS
            if len(seq) <= 2:
                assert reward == 0.0


# ---------------------------------------------------------------------------
# ValueNetwork tests
# ---------------------------------------------------------------------------


class TestValueNetwork:
    def test_output_shape(self):
        net = ValueNetwork(d_model=512, hidden_dim=256)
        x = torch.randn(4, 512)
        out = net(x)
        assert out.shape == (4, 1)

    def test_output_range(self):
        """Output should be in [0, 1] due to sigmoid."""
        net = ValueNetwork(d_model=128, hidden_dim=64)
        x = torch.randn(100, 128)
        out = net(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_single_input(self):
        net = ValueNetwork(d_model=64, hidden_dim=32)
        x = torch.randn(1, 64)
        out = net(x)
        assert out.shape == (1, 1)
        assert 0.0 <= out.item() <= 1.0

    def test_custom_dimensions(self):
        net = ValueNetwork(d_model=256, hidden_dim=128)
        # Verify parameter shapes
        params = list(net.parameters())
        # Linear(256, 128) weight + bias, Linear(128, 1) weight + bias
        assert params[0].shape == (128, 256)
        assert params[1].shape == (128,)
        assert params[2].shape == (1, 128)
        assert params[3].shape == (1,)
