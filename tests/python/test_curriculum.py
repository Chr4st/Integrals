"""Tests for curriculum learning (Phase 12b)."""
import pytest
from neurips.training.curriculum import (
    CurriculumScheduler,
    filter_data,
    weighted_sample,
)


@pytest.fixture
def scheduler():
    return CurriculumScheduler()


@pytest.fixture
def sample_data():
    tasks = ["univariate", "multivariate", "definite", "parametric", "special_fn"]
    tiers = ["easy", "medium", "hard", "very_hard"]
    data = []
    for task in tasks:
        for tier in tiers:
            for i in range(10):
                data.append({
                    "task": task,
                    "difficulty_tier": tier,
                    "integrand": f"dummy_{task}_{tier}_{i}",
                })
    return data


class TestCurriculumScheduler:
    def test_phase1_univariate_only(self, scheduler):
        tasks = scheduler.get_active_tasks(1)
        assert list(tasks.keys()) == ["univariate"]
        assert abs(sum(tasks.values()) - 1.0) < 1e-6

    def test_phase2_adds_multivariate(self, scheduler):
        tasks = scheduler.get_active_tasks(20)
        assert "multivariate" in tasks
        assert abs(sum(tasks.values()) - 1.0) < 1e-6

    def test_phase3_all_tasks(self, scheduler):
        tasks = scheduler.get_active_tasks(40)
        assert len(tasks) == 5
        assert abs(sum(tasks.values()) - 1.0) < 1e-6

    def test_phase4_uniform(self, scheduler):
        tasks = scheduler.get_active_tasks(50)
        assert len(tasks) == 5

    def test_difficulty_cap_early(self, scheduler):
        assert scheduler.get_max_difficulty(5) == "medium"

    def test_difficulty_cap_mid(self, scheduler):
        assert scheduler.get_max_difficulty(20) == "hard"

    def test_difficulty_cap_late(self, scheduler):
        assert scheduler.get_max_difficulty(31) == "very_hard"


class TestFiltering:
    def test_filter_univariate_only(self, sample_data):
        filtered = filter_data(
            sample_data, {"univariate": 1.0}, "very_hard"
        )
        assert all(d["task"] == "univariate" for d in filtered)

    def test_filter_medium_cap(self, sample_data):
        filtered = filter_data(
            sample_data, {"univariate": 1.0}, "medium"
        )
        assert all(d["difficulty_tier"] in ["easy", "medium"] for d in filtered)

    def test_weighted_sample_count(self, sample_data):
        sampled = weighted_sample(
            sample_data, {"univariate": 0.5, "multivariate": 0.5}, 100
        )
        assert len(sampled) == 100
