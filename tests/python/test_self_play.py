"""Tests for self-play curriculum (Sprint 3)."""

import pytest
import torch

from neurips.training.self_play import (
    CurriculumState,
    SelfPlayConfig,
    Trajectory,
    compute_policy_loss,
    generate_trajectories,
)
from neurips.models.action_policy import ActionPolicy


class _MockEnv:
    def __init__(self):
        self._step = 0
        self._done = False

    def is_done(self) -> bool:
        return self._done

    def state_features(self) -> list[float]:
        return [0.0] * 64

    def step_close(self, prefix: str):
        self._done = True
        return 1.0, True, True

    def step_substitute(self, prefix: str):
        self._step += 1
        return 0.0, False, True

    def step_ibp(self, u: str, dv: str):
        self._step += 1
        return 0.0, False, True


def _mock_env_factory(prefix: str, var: str, max_steps: int):
    return _MockEnv()


class TestTrajectory:
    def test_frozen(self):
        t = Trajectory(
            integrand_prefix="var:x",
            antiderivative_prefix="div pow var:x 2 2",
            action_ids=[0, 3],
            rewards=[0.0, 1.0],
            total_reward=1.0,
            chain_length=2,
        )
        assert t.chain_length == 2
        with pytest.raises(AttributeError):
            t.chain_length = 3


class TestSelfPlayConfig:
    def test_defaults(self):
        cfg = SelfPlayConfig()
        assert cfg.min_chain_length == 1
        assert cfg.max_chain_length == 3
        assert cfg.episodes_per_batch == 256


class TestCurriculumState:
    def test_initial_state(self):
        state = CurriculumState()
        assert state.current_max_length == 1
        assert state.episodes_completed == 0

    def test_update_tracks_solve_rate(self):
        cfg = SelfPlayConfig(complexity_threshold=0.8, escalation_step=5)
        state = CurriculumState()
        for _ in range(5):
            state.update(True, cfg)
        assert state.recent_solve_rate == 1.0
        assert state.episodes_completed == 5

    def test_escalation(self):
        cfg = SelfPlayConfig(
            complexity_threshold=0.5,
            escalation_step=10,
            max_chain_length=3,
        )
        state = CurriculumState()
        for _ in range(10):
            state.update(True, cfg)
        # After 10 successes with threshold 0.5, should escalate
        assert state.current_max_length == 2

    def test_no_escalation_below_threshold(self):
        cfg = SelfPlayConfig(
            complexity_threshold=0.9,
            escalation_step=10,
        )
        state = CurriculumState()
        # 5 solved, 5 unsolved = 50% < 90% threshold
        for i in range(10):
            state.update(i < 5, cfg)
        assert state.current_max_length == 1


class TestGenerateTrajectories:
    def test_produces_trajectories(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        cfg = SelfPlayConfig(episodes_per_batch=4, max_steps_per_episode=3)
        curriculum = CurriculumState()
        trajs = generate_trajectories(
            policy=policy,
            env_factory=_mock_env_factory,
            integrands=["var:x", "pow var:x 2"],
            config=cfg,
            curriculum=curriculum,
        )
        assert isinstance(trajs, list)
        assert curriculum.episodes_completed == 4


class TestComputePolicyLoss:
    def test_empty_trajectories(self):
        loss = compute_policy_loss(ActionPolicy(embed_dim=64), [])
        assert loss.item() == 0.0

    def test_returns_scalar(self):
        traj = Trajectory(
            integrand_prefix="var:x",
            antiderivative_prefix="div pow var:x 2 2",
            action_ids=[0, 3],
            rewards=[0.0, 1.0],
            total_reward=1.0,
            chain_length=2,
        )
        loss = compute_policy_loss(
            ActionPolicy(embed_dim=64), [traj], gamma=0.99
        )
        assert loss.shape == ()
        assert loss.requires_grad

    def test_multiple_trajectories(self):
        trajs = [
            Trajectory("a", "b", [0], [1.0], 1.0, 1),
            Trajectory("c", "d", [1, 3], [0.0, -1.0], -1.0, 2),
        ]
        loss = compute_policy_loss(ActionPolicy(embed_dim=64), trajs)
        assert loss.shape == ()
