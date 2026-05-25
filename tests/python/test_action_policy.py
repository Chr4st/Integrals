"""Tests for action-space policy head (Sprint 3)."""

import pytest
import torch

from neurips.models.action_policy import (
    NUM_ACTIONS,
    ActionPolicy,
    IBPParamHead,
    SubstitutionParamHead,
)


class TestActionPolicy:
    def test_forward_shape(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        x = torch.randn(4, 64)
        logits = policy(x)
        assert logits.shape == (4, NUM_ACTIONS)

    def test_num_actions(self):
        assert NUM_ACTIONS == 4

    def test_select_action_greedy(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        x = torch.randn(4, 64)
        action_ids, log_probs = policy.select_action(x, temperature=0.0)
        assert action_ids.shape == (4,)
        assert log_probs.shape == (4,)
        # Greedy: log_probs are valid (negative) probabilities
        assert (log_probs <= 0.0).all()
        assert torch.isfinite(log_probs).all()

    def test_select_action_stochastic(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        x = torch.randn(4, 64)
        action_ids, log_probs = policy.select_action(x, temperature=1.0)
        assert action_ids.shape == (4,)
        assert (action_ids >= 0).all() and (action_ids < NUM_ACTIONS).all()

    def test_gradient_flows(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        x = torch.randn(4, 64, requires_grad=True)
        logits = policy(x)
        logits.sum().backward()
        assert x.grad is not None


class TestSubstitutionParamHead:
    def test_forward_shape(self):
        head = SubstitutionParamHead(embed_dim=64)
        pooled = torch.randn(2, 64)
        nodes = torch.randn(2, 10, 64)
        scores = head(pooled, nodes)
        assert scores.shape == (2, 10)

    def test_mask(self):
        head = SubstitutionParamHead(embed_dim=64)
        pooled = torch.randn(2, 64)
        nodes = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, :5] = True
        scores = head(pooled, nodes, mask)
        # Masked positions should be -inf
        assert (scores[:, 5:] == float("-inf")).all()


class TestIBPParamHead:
    def test_forward_shape(self):
        head = IBPParamHead(embed_dim=64)
        pooled = torch.randn(2, 64)
        nodes = torch.randn(2, 10, 64)
        u_logits, dv_logits = head(pooled, nodes)
        assert u_logits.shape == (2, 10)
        assert dv_logits.shape == (2, 10)
