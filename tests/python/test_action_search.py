"""Tests for action-space search strategies (Sprint 3)."""

import pytest
import torch

from neurips.inference.action_search import (
    ACTION_NAMES,
    ActionTrace,
    SearchResult,
    greedy_search,
    beam_search,
)
from neurips.models.action_policy import ActionPolicy


class _MockEnv:
    """Minimal mock environment for search tests."""

    def __init__(self, solve_on_step: int = 2):
        self._step = 0
        self._done = False
        self._solve_on = solve_on_step

    def is_done(self) -> bool:
        return self._done

    def state_features(self) -> list[float]:
        return [0.0] * 64

    def step_close(self, prefix: str):
        self._step += 1
        if self._step >= self._solve_on:
            self._done = True
            return 1.0, True, True
        return -1.0, True, False

    def step_substitute(self, prefix: str):
        self._step += 1
        return 0.0, False, True

    def step_ibp(self, u: str, dv: str):
        self._step += 1
        return 0.0, False, True


class _MockEncoder:
    pass


def _mock_env_factory(prefix: str, var: str, max_steps: int):
    return _MockEnv(solve_on_step=2)


class TestActionTrace:
    def test_frozen(self):
        t = ActionTrace(
            action_id=0,
            action_name="substitute",
            state_before="x",
            state_after="u",
            verify_ok=True,
        )
        assert t.action_id == 0
        assert t.verify_ok is True


class TestSearchResult:
    def test_defaults(self):
        r = SearchResult(solved=False)
        assert r.antiderivative is None
        assert r.trace == []
        assert r.total_steps == 0


class TestActionNames:
    def test_count(self):
        assert len(ACTION_NAMES) == 4

    def test_contents(self):
        assert "substitute" in ACTION_NAMES
        assert "close" in ACTION_NAMES


class TestGreedySearch:
    def test_returns_search_result(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        encoder = _MockEncoder()
        result = greedy_search(
            policy=policy,
            encoder=encoder,
            env_factory=_mock_env_factory,
            integrand_prefix="var:x",
            max_steps=5,
        )
        assert isinstance(result, SearchResult)

    def test_respects_max_steps(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)

        def never_solve_factory(prefix, var, max_steps):
            return _MockEnv(solve_on_step=999)

        result = greedy_search(
            policy=policy,
            encoder=_MockEncoder(),
            env_factory=never_solve_factory,
            integrand_prefix="var:x",
            max_steps=3,
        )
        assert not result.solved
        assert result.total_steps <= 3


class TestBeamSearch:
    def test_returns_search_result(self):
        policy = ActionPolicy(embed_dim=64, hidden_dim=32)
        result = beam_search(
            policy=policy,
            encoder=_MockEncoder(),
            env_factory=_mock_env_factory,
            integrand_prefix="var:x",
            beam_width=2,
            max_steps=5,
        )
        assert isinstance(result, SearchResult)
