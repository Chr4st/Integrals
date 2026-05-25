"""Tests for dataset caching and curriculum index maps.

Covers: python/neurips/data/dataset.py (precompute + load roundtrip),
        python/neurips/training/curriculum.py (index maps, schedule boundaries).
Verifies actual tensor contents and mathematical invariants,
not just shapes.
"""
from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings, strategies as st

from neurips.data.dataset import (
    IntegralDataset,
    _DIFFICULTY_IDS,
    _TASK_IDS,
    _TASK_TO_TOKENIZER,
    precompute_dataset,
)
from neurips.data.features import FEATURE_DIM
from neurips.data.tokenizer import IntegralTokenizer
from neurips.data.vocab import EOS, FEAT, VOCAB
from neurips.training.curriculum import (
    CurriculumScheduler,
    build_index_maps,
    filter_data,
    filter_indices,
    weighted_sample_indices,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer() -> IntegralTokenizer:
    return IntegralTokenizer()


@pytest.fixture
def sample_raw_data() -> list[dict]:
    """Minimal raw dataset with known values for verifiable assertions."""
    return [
        {
            "integrand": "pow x 2",
            "antiderivative": "div pow x 3 3",
            "task": "univariate",
            "var": "x",
            "difficulty_tier": "easy",
        },
        {
            "integrand": "sin x",
            "antiderivative": "neg cos x",
            "task": "univariate",
            "var": "x",
            "difficulty_tier": "medium",
        },
        {
            "integrand": "mul cos x sin x",
            "antiderivative": "div pow sin x 2 2",
            "task": "univariate",
            "var": "x",
            "difficulty_tier": "hard",
        },
    ]


@pytest.fixture
def multi_task_data() -> list[dict]:
    """Dataset spanning all task types and difficulties for curriculum tests."""
    tasks = ["univariate", "multivariate", "parametric", "definite", "special_fn"]
    tiers = ["easy", "medium", "hard", "very_hard"]
    data = []
    for task in tasks:
        for tier in tiers:
            for i in range(5):
                data.append({
                    "integrand": f"pow x {i + 2}",
                    "antiderivative": f"div pow x {i + 3} {i + 3}",
                    "task": task,
                    "var": "x",
                    "difficulty_tier": tier,
                })
    return data


# ---------------------------------------------------------------------------
# Dataset precompute + load roundtrip
# ---------------------------------------------------------------------------


class TestDatasetRoundtrip:
    """Precompute raw data, load from cache, verify tensor contents."""

    def test_roundtrip_token_ids_match_manual_encoding(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Token IDs in the cached dataset must match manual encoding
        of the same expressions with the same tokenizer.
        """
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
            max_src_len=64, max_tgt_len=64,
        )

        for i, ex in enumerate(sample_raw_data):
            tok_task = _TASK_TO_TOKENIZER.get(ex["task"], "indef")
            expected_src = tokenizer.encode(
                ex["integrand"], tok_task, ex["var"],
            )
            expected_tgt = tokenizer.encode(
                ex["antiderivative"], tok_task, ex["var"],
            )

            actual_src = ds.src_ids[i].tolist()
            actual_tgt = ds.tgt_ids[i].tolist()

            # Compare the non-padded prefix
            src_len = min(len(expected_src), 64)
            tgt_len = min(len(expected_tgt), 64)

            assert actual_src[:src_len] == expected_src[:src_len], (
                f"Source mismatch at index {i}:\n"
                f"  expected: {expected_src[:src_len]}\n"
                f"  actual:   {actual_src[:src_len]}"
            )
            assert actual_tgt[:tgt_len] == expected_tgt[:tgt_len], (
                f"Target mismatch at index {i}:\n"
                f"  expected: {expected_tgt[:tgt_len]}\n"
                f"  actual:   {actual_tgt[:tgt_len]}"
            )

    def test_padding_is_zeros(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Positions beyond the encoded length must be zero (PAD)."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
            max_src_len=64, max_tgt_len=64,
        )

        for i, ex in enumerate(sample_raw_data):
            tok_task = _TASK_TO_TOKENIZER.get(ex["task"], "indef")
            src_ids = tokenizer.encode(ex["integrand"], tok_task, ex["var"])
            src_len = min(len(src_ids), 64)
            padding = ds.src_ids[i, src_len:]
            assert (padding == 0).all(), (
                f"Non-zero padding in src_ids[{i}] starting at position {src_len}"
            )

    def test_features_have_correct_dim(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Feature vectors must be exactly FEATURE_DIM-dimensional."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        assert ds.features.shape == (len(sample_raw_data), FEATURE_DIM)

    def test_features_are_not_all_zero(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Feature vectors for non-trivial expressions should have nonzero entries."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        for i in range(len(sample_raw_data)):
            assert ds.features[i].abs().sum() > 0, (
                f"Feature vector at index {i} is all zeros for "
                f"integrand={sample_raw_data[i]['integrand']!r}"
            )

    def test_task_ids_match_expected(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """task_id tensor values must match _TASK_IDS mapping."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        for i, ex in enumerate(sample_raw_data):
            expected = _TASK_IDS[ex["task"]]
            actual = ds.task_ids[i].item()
            assert actual == expected, (
                f"task_id mismatch at {i}: expected {expected} "
                f"({ex['task']}), got {actual}"
            )

    def test_diff_ids_match_expected(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """diff_id tensor values must match _DIFFICULTY_IDS mapping."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        for i, ex in enumerate(sample_raw_data):
            expected = _DIFFICULTY_IDS[ex["difficulty_tier"]]
            actual = ds.diff_ids[i].item()
            assert actual == expected, (
                f"diff_id mismatch at {i}: expected {expected} "
                f"({ex['difficulty_tier']}), got {actual}"
            )

    def test_getitem_returns_correct_keys(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """__getitem__ returns a dict with all required keys."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        item = ds[0]
        assert set(item.keys()) == {"src_ids", "tgt_ids", "features", "task_id", "diff_id"}

    def test_len_matches_input(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Dataset length matches the number of input examples."""
        cache_path = tmp_path / "test_cache.pt"
        ds = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        assert len(ds) == len(sample_raw_data)

    def test_reload_from_cache_matches_original(
        self, sample_raw_data: list[dict], tokenizer: IntegralTokenizer, tmp_path
    ) -> None:
        """Loading a second IntegralDataset from the same cache file
        produces identical tensors.
        """
        cache_path = tmp_path / "test_cache.pt"
        ds1 = precompute_dataset(
            sample_raw_data, tokenizer, cache_path,
        )
        ds2 = IntegralDataset(cache_path)

        assert torch.equal(ds1.src_ids, ds2.src_ids)
        assert torch.equal(ds1.tgt_ids, ds2.tgt_ids)
        assert torch.equal(ds1.features, ds2.features)
        assert torch.equal(ds1.task_ids, ds2.task_ids)
        assert torch.equal(ds1.diff_ids, ds2.diff_ids)


# ---------------------------------------------------------------------------
# Index map correctness (curriculum.py)
# ---------------------------------------------------------------------------


class TestIndexMaps:
    """Verify that index-map based filtering produces the same results
    as direct data filtering.
    """

    def test_filter_indices_matches_filter_data(
        self, multi_task_data: list[dict]
    ) -> None:
        """filter_indices must return exactly the same set of examples
        as filter_data, for several curriculum configurations.
        """
        index_maps = build_index_maps(multi_task_data)
        configs = [
            ({"univariate": 1.0}, "medium"),
            ({"univariate": 0.5, "multivariate": 0.3, "parametric": 0.2}, "hard"),
            ({
                "univariate": 0.25, "multivariate": 0.20,
                "parametric": 0.15, "definite": 0.20, "special_fn": 0.20,
            }, "very_hard"),
        ]
        for active_tasks, max_diff in configs:
            expected_examples = filter_data(multi_task_data, active_tasks, max_diff)
            indices = filter_indices(index_maps, active_tasks, max_diff)
            actual_examples = [multi_task_data[i] for i in indices]

            # Compare as sets (order may differ)
            expected_set = {
                (e["task"], e["difficulty_tier"], e["integrand"])
                for e in expected_examples
            }
            actual_set = {
                (e["task"], e["difficulty_tier"], e["integrand"])
                for e in actual_examples
            }
            assert expected_set == actual_set, (
                f"Mismatch for config ({active_tasks}, {max_diff}):\n"
                f"  expected {len(expected_set)} examples, got {len(actual_set)}"
            )

    def test_weighted_sample_indices_stay_in_pool(
        self, multi_task_data: list[dict]
    ) -> None:
        """weighted_sample_indices must only return indices that belong
        to the allowed (task, difficulty) combinations.
        """
        index_maps = build_index_maps(multi_task_data)
        active_tasks = {"univariate": 0.5, "multivariate": 0.5}
        max_diff = "medium"

        allowed_indices = set(
            filter_indices(index_maps, active_tasks, max_diff)
        )

        sampled = weighted_sample_indices(
            index_maps, active_tasks, max_diff, n_samples=200,
        )
        for idx in sampled:
            assert idx in allowed_indices, (
                f"Sampled index {idx} not in allowed pool. "
                f"Example: {multi_task_data[idx]}"
            )

    @given(
        task_combo=st.sampled_from([
            {"univariate": 1.0},
            {"univariate": 0.5, "multivariate": 0.5},
            {"univariate": 0.3, "multivariate": 0.3, "parametric": 0.4},
        ]),
        max_diff=st.sampled_from(["easy", "medium", "hard", "very_hard"]),
    )
    @settings(max_examples=30, deadline=5000)
    def test_property_filtered_indices_match_task_and_difficulty(
        self, task_combo: dict[str, float], max_diff: str
    ) -> None:
        """Property: for any valid (active_tasks, max_difficulty), every
        index returned by filter_indices references an example with a
        matching task AND an allowed difficulty.
        """
        # Build data inside the test (hypothesis requires deterministic data)
        tasks = ["univariate", "multivariate", "parametric", "definite", "special_fn"]
        tiers = ["easy", "medium", "hard", "very_hard"]
        data = []
        for task in tasks:
            for tier in tiers:
                data.append({
                    "task": task,
                    "difficulty_tier": tier,
                    "integrand": f"dummy_{task}_{tier}",
                })

        diff_order = ("easy", "medium", "hard", "very_hard")
        max_idx = diff_order.index(max_diff)
        allowed_diffs = set(diff_order[: max_idx + 1])

        index_maps = build_index_maps(data)
        indices = filter_indices(index_maps, task_combo, max_diff)

        for idx in indices:
            ex = data[idx]
            assert ex["task"] in task_combo, (
                f"Index {idx} has task={ex['task']!r} not in {list(task_combo)}"
            )
            assert ex["difficulty_tier"] in allowed_diffs, (
                f"Index {idx} has diff={ex['difficulty_tier']!r} "
                f"not in {allowed_diffs}"
            )


# ---------------------------------------------------------------------------
# Curriculum schedule boundary tests
# ---------------------------------------------------------------------------


class TestCurriculumBoundaries:
    """Verify that task sets change at documented epoch boundaries."""

    @pytest.fixture
    def scheduler(self) -> CurriculumScheduler:
        return CurriculumScheduler()

    # Difficulty tier boundaries
    @pytest.mark.parametrize(
        "epoch,expected_max_diff",
        [
            (1, "medium"),
            (10, "medium"),
            (11, "hard"),
            (25, "hard"),
            (26, "very_hard"),
            (100, "very_hard"),
        ],
        ids=[
            "epoch-1-medium",
            "epoch-10-boundary-medium",
            "epoch-11-crosses-to-hard",
            "epoch-25-boundary-hard",
            "epoch-26-crosses-to-very_hard",
            "epoch-100-still-very_hard",
        ],
    )
    def test_difficulty_boundary(
        self, scheduler: CurriculumScheduler, epoch: int, expected_max_diff: str
    ) -> None:
        """At each epoch boundary, the max difficulty tier must change correctly."""
        assert scheduler.get_max_difficulty(epoch) == expected_max_diff

    # Task set boundaries
    @pytest.mark.parametrize(
        "epoch,expected_tasks",
        [
            (1, {"univariate"}),
            (15, {"univariate"}),
            (16, {"univariate", "multivariate", "parametric"}),
            (30, {"univariate", "multivariate", "parametric"}),
            (31, {"univariate", "multivariate", "parametric", "definite", "special_fn"}),
            (45, {"univariate", "multivariate", "parametric", "definite", "special_fn"}),
            (46, {"univariate", "multivariate", "parametric", "definite", "special_fn"}),
        ],
        ids=[
            "epoch-1-univariate-only",
            "epoch-15-boundary-univariate",
            "epoch-16-adds-multi-param",
            "epoch-30-boundary-3-tasks",
            "epoch-31-all-5-tasks",
            "epoch-45-boundary-5-tasks",
            "epoch-46-final-stage",
        ],
    )
    def test_task_set_boundary(
        self,
        scheduler: CurriculumScheduler,
        epoch: int,
        expected_tasks: set[str],
    ) -> None:
        """At each epoch boundary, the active task set must match exactly."""
        active = scheduler.get_active_tasks(epoch)
        assert set(active.keys()) == expected_tasks, (
            f"Epoch {epoch}: expected tasks {expected_tasks}, "
            f"got {set(active.keys())}"
        )

    @pytest.mark.parametrize(
        "epoch",
        [1, 5, 10, 15, 16, 20, 25, 30, 31, 40, 45, 46, 50, 100],
    )
    def test_weights_sum_to_one(
        self, scheduler: CurriculumScheduler, epoch: int
    ) -> None:
        """Task weights must always sum to 1.0 (within floating point tolerance)."""
        active = scheduler.get_active_tasks(epoch)
        total = sum(active.values())
        assert abs(total - 1.0) < 1e-10, (
            f"Epoch {epoch}: weights sum to {total}, not 1.0. "
            f"Tasks: {active}"
        )

    @pytest.mark.parametrize(
        "epoch",
        [1, 5, 10, 15, 16, 20, 25, 30, 31, 40, 45, 46, 50, 100],
    )
    def test_all_weights_positive(
        self, scheduler: CurriculumScheduler, epoch: int
    ) -> None:
        """Every task weight must be strictly positive."""
        active = scheduler.get_active_tasks(epoch)
        for task, weight in active.items():
            assert weight > 0, (
                f"Epoch {epoch}: task {task!r} has non-positive weight {weight}"
            )
