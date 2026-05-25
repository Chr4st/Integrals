"""Tests for MCTS search."""

import torch

from neurips.inference.mcts import MCTSConfig, MCTSNode, mcts_search
from neurips.models.action_policy import NUM_ACTIONS


def _dummy_policy_fn(state: str) -> torch.Tensor:
    priors = torch.ones(NUM_ACTIONS) / NUM_ACTIONS
    return priors


def _dummy_value_fn(state: str) -> float:
    return 0.0


def _dummy_env_step(state: str, action: int) -> tuple[str, bool, bool]:
    if action == 3:
        return state + "_closed", True, True
    return state + f"_a{action}", False, False


class TestMCTSNode:
    def test_q_value_zero_visits(self):
        node = MCTSNode(state="x")
        assert node.q_value == 0.0

    def test_q_value_after_visits(self):
        node = MCTSNode(state="x")
        node.visit_count = 4
        node.total_value = 2.0
        assert node.q_value == 0.5


class TestMCTSSearch:
    def test_basic_search_returns_result(self):
        config = MCTSConfig(n_simulations=10, max_depth=3)
        result = mcts_search(
            root_state="(add (var x) (num 1))",
            policy_fn=_dummy_policy_fn,
            value_fn=_dummy_value_fn,
            env_step_fn=_dummy_env_step,
            config=config,
        )
        assert len(result.visit_counts) == NUM_ACTIONS
        assert len(result.policy_target) == NUM_ACTIONS
        assert abs(sum(result.policy_target) - 1.0) < 1e-5

    def test_solved_detection(self):
        config = MCTSConfig(n_simulations=50, max_depth=3)

        def close_biased_policy(state: str) -> torch.Tensor:
            priors = torch.tensor([0.05, 0.05, 0.05, 0.85])
            return priors

        result = mcts_search(
            root_state="(var x)",
            policy_fn=close_biased_policy,
            value_fn=_dummy_value_fn,
            env_step_fn=_dummy_env_step,
            config=config,
        )
        assert result.solved
        assert result.antiderivative is not None

    def test_greedy_temperature(self):
        config = MCTSConfig(n_simulations=20, max_depth=3, temperature=0.0)
        result = mcts_search(
            root_state="(var x)",
            policy_fn=_dummy_policy_fn,
            value_fn=_dummy_value_fn,
            env_step_fn=_dummy_env_step,
            config=config,
        )
        assert max(result.policy_target) == 1.0
        assert result.policy_target.count(1.0) == 1

    def test_visit_counts_sum_to_simulations(self):
        config = MCTSConfig(n_simulations=30, max_depth=5)
        result = mcts_search(
            root_state="(var x)",
            policy_fn=_dummy_policy_fn,
            value_fn=_dummy_value_fn,
            env_step_fn=_dummy_env_step,
            config=config,
        )
        assert sum(result.visit_counts) <= config.n_simulations
