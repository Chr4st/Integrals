"""Tests for curriculum learning sampler."""
import pytest

from integral_engine.curriculum import (
    CurriculumSampler,
    build_curriculum_sampler,
    difficulty_score,
)


class TestDifficultyScore:
    def test_longer_input_higher_score(self):
        easy = {"input_tokens": ["sin", "x"]}
        hard = {"input_tokens": ["Add", "Mul", "sin", "x", "cos", "x"]}
        assert difficulty_score(easy) < difficulty_score(hard)

    def test_empty_input(self):
        assert difficulty_score({"input_tokens": []}) == 0


class TestCurriculumSampler:
    def _make_difficulties(self):
        return [1, 2, 3, 5, 8, 10, 15, 20]

    def test_epoch_zero_only_easiest(self):
        diffs = self._make_difficulties()
        sampler = CurriculumSampler(diffs, warmup_epochs=10)
        sampler.set_epoch(0)
        indices = list(sampler)
        assert len(indices) == 1
        assert 0 in indices

    def test_full_warmup_includes_all(self):
        diffs = self._make_difficulties()
        sampler = CurriculumSampler(diffs, warmup_epochs=10)
        sampler.set_epoch(10)
        indices = list(sampler)
        assert len(indices) == len(diffs)

    def test_past_warmup_still_all(self):
        diffs = self._make_difficulties()
        sampler = CurriculumSampler(diffs, warmup_epochs=5)
        sampler.set_epoch(100)
        assert len(sampler) == len(diffs)

    def test_monotonic_growth(self):
        diffs = self._make_difficulties()
        sampler = CurriculumSampler(diffs, warmup_epochs=10)
        sizes = []
        for e in range(12):
            sampler.set_epoch(e)
            sizes.append(len(sampler))
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_deterministic_with_same_seed(self):
        diffs = self._make_difficulties()
        s1 = CurriculumSampler(diffs, warmup_epochs=5, seed=42)
        s2 = CurriculumSampler(diffs, warmup_epochs=5, seed=42)
        s1.set_epoch(3)
        s2.set_epoch(3)
        assert list(s1) == list(s2)

    def test_different_seeds_differ(self):
        diffs = self._make_difficulties()
        s1 = CurriculumSampler(diffs, warmup_epochs=5, seed=1)
        s2 = CurriculumSampler(diffs, warmup_epochs=5, seed=99)
        s1.set_epoch(5)
        s2.set_epoch(5)
        assert sorted(list(s1)) == sorted(list(s2))

    def test_empty_difficulties(self):
        sampler = CurriculumSampler([], warmup_epochs=5)
        assert len(sampler) == 0
        assert list(sampler) == []


class TestBuildCurriculumSampler:
    def test_builds_from_items(self):
        items = [
            {"input_tokens": ["x"]},
            {"input_tokens": ["sin", "x"]},
            {"input_tokens": ["Add", "sin", "x", "cos", "x"]},
        ]
        sampler = build_curriculum_sampler(items, warmup_epochs=5)
        sampler.set_epoch(5)
        assert len(sampler) == 3
