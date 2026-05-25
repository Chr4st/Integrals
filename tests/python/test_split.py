"""Tests for data splitting (Phase 6)."""
import pytest
from neurips.data.split import (
    deduplicate,
    assign_tier,
    stratified_skeleton_split,
)


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        pairs = [
            {"integrand": "add x x", "task": "INDEF"},
            {"integrand": "add x x", "task": "INDEF"},
            {"integrand": "mul x x", "task": "INDEF"},
        ]
        result = deduplicate(pairs)
        assert len(result) == 2

    def test_keeps_unique(self):
        pairs = [
            {"integrand": "sin x", "task": "INDEF"},
            {"integrand": "cos x", "task": "INDEF"},
        ]
        result = deduplicate(pairs)
        assert len(result) == 2


class TestAssignTier:
    def test_easy(self):
        pair = {"integrand": "x"}  # depth=0, 1 node, score=1
        assert assign_tier(pair) == "easy"

    def test_medium(self):
        pair = {"integrand": "add sin x mul x pow x 2"}  # deeper
        tier = assign_tier(pair)
        assert tier in ("medium", "hard")

    def test_hard(self):
        # Build a deep expression: sin(sin(sin(...sin(x)...)))
        # 8 nested sins + x = depth 8, 9 nodes, score=25 → medium boundary
        # Use 13 nested sins for score=13*2+14=40 → hard
        prefix = "sin " * 13 + "x"
        pair = {"integrand": prefix.strip()}
        tier = assign_tier(pair)
        assert tier in ("hard", "very_hard")

    def test_very_hard(self):
        # Very deep nested expression
        prefix = "add " * 15 + " ".join(["x"] * 16)
        pair = {"integrand": prefix.strip()}
        tier = assign_tier(pair)
        assert tier == "very_hard"


class TestStratifiedSplit:
    def test_no_skeleton_overlap(self):
        pairs = []
        for i in range(100):
            pairs.append({
                "integrand": f"add mul {i} x x",
                "integrand_skeleton": f"(add (mul C x) x)" if i % 2 == 0 else f"(add (mul C x) x)_{i}",
                "task": "univariate",
                "difficulty_tier": "easy",
                "integrand_depth": 3,
                "integrand_nodes": 5,
            })
        train, test = stratified_skeleton_split(pairs, train_ratio=0.8, seed=42)
        train_skels = {p["integrand_skeleton"] for p in train}
        test_skels = {p["integrand_skeleton"] for p in test}
        assert train_skels & test_skels == set()

    def test_total_preserved(self):
        pairs = []
        for i in range(50):
            pairs.append({
                "integrand": f"expr_{i}",
                "integrand_skeleton": f"skel_{i}",
                "task": "univariate",
                "difficulty_tier": "easy",
                "integrand_depth": 2,
                "integrand_nodes": 3,
            })
        train, test = stratified_skeleton_split(pairs, train_ratio=0.8, seed=42)
        assert len(train) + len(test) == len(pairs)
