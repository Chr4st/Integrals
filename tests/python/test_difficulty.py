"""Tests for difficulty scoring and competence curriculum."""

import numpy as np
import pytest

from neurips.training.curriculum import CompetenceCurriculum
from neurips.training.difficulty import batch_difficulty, compute_difficulty


class TestComputeDifficulty:
    def test_easy_example_low_score(self):
        score = compute_difficulty(tree_depth=2, n_ops=3, seq_length=30, tier=0)
        assert score.score < 0.2

    def test_hard_example_high_score(self):
        score = compute_difficulty(tree_depth=18, n_ops=25, seq_length=450, tier=3)
        assert score.score > 0.8

    def test_score_bounds(self):
        score = compute_difficulty(tree_depth=0, n_ops=0, seq_length=0, tier=0)
        assert 0.0 <= score.score <= 1.0

        score = compute_difficulty(tree_depth=100, n_ops=100, seq_length=1000, tier=3)
        assert 0.0 <= score.score <= 1.0

    def test_monotonicity_with_depth(self):
        s1 = compute_difficulty(tree_depth=5, n_ops=10, seq_length=100, tier=1)
        s2 = compute_difficulty(tree_depth=15, n_ops=10, seq_length=100, tier=1)
        assert s2.score > s1.score


class TestBatchDifficulty:
    def test_batch_output_shape(self):
        scores = batch_difficulty(
            depths=[2, 5, 10],
            ops=[3, 8, 20],
            lengths=[30, 100, 300],
            tiers=[0, 1, 3],
        )
        assert scores.shape == (3,)
        assert scores.dtype == np.float32

    def test_ordering(self):
        scores = batch_difficulty(
            depths=[2, 10, 18],
            ops=[3, 15, 28],
            lengths=[30, 200, 480],
            tiers=[0, 2, 3],
        )
        assert scores[0] < scores[1] < scores[2]


class TestCompetenceCurriculum:
    def test_initial_competence(self):
        cc = CompetenceCurriculum(total_steps=1000, c_0=0.1)
        assert abs(cc.competence(0) - 0.1) < 1e-5

    def test_competence_increases(self):
        cc = CompetenceCurriculum(total_steps=1000)
        c1 = cc.competence(100)
        c2 = cc.competence(500)
        c3 = cc.competence(1000)
        assert c1 < c2 < c3

    def test_competence_caps_at_one(self):
        cc = CompetenceCurriculum(total_steps=100)
        assert cc.competence(100) <= 1.0
        assert cc.competence(200) <= 1.0

    def test_step_advances(self):
        cc = CompetenceCurriculum(total_steps=1000)
        cc.step()
        cc.step()
        assert cc.competence() == cc.competence(2)

    def test_sampling_weights_shape(self):
        cc = CompetenceCurriculum(total_steps=1000)
        indices = list(range(10))
        difficulties = np.linspace(0.0, 1.0, 10).astype(np.float32)
        weights = cc.get_sampling_weights(indices, difficulties)
        assert weights.shape == (10,)

    def test_hard_examples_excluded_early(self):
        cc = CompetenceCurriculum(total_steps=1000, c_0=0.1)
        indices = list(range(5))
        difficulties = np.array([0.05, 0.2, 0.5, 0.8, 0.99], dtype=np.float32)
        weights = cc.get_sampling_weights(indices, difficulties)
        assert weights[0] > 0
        assert weights[4] == 0

    def test_state_dict_roundtrip(self):
        cc = CompetenceCurriculum(total_steps=500, c_0=0.2)
        for _ in range(50):
            cc.step()
        state = cc.state_dict()

        cc2 = CompetenceCurriculum(total_steps=100)
        cc2.load_state_dict(state)
        assert cc2.competence() == cc.competence()
