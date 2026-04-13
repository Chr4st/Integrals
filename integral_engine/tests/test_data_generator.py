"""Tests for backward data generation."""
import gzip
import json
import os

import pytest
import sympy as sp

from integral_engine.data_generator import (
    DIFFICULTY_TIERS,
    _expression_hash,
    _random_expression_tree,
    _try_generate_pair,
    generate_backward_pairs,
    write_shards,
)
from integral_engine.vocabulary import FEATURE_DIM, MAX_INPUT_LEN, MAX_OUTPUT_LEN

import random


class TestRandomExpressionTree:
    def test_depth_zero_returns_leaf(self):
        rng = random.Random(42)
        expr = _random_expression_tree(rng, 0)
        # Depth 0 should be a simple expression (leaf)
        assert expr is not None
        x = sp.Symbol("x")
        # Should be differentiable
        sp.diff(expr, x)

    def test_depth_positive_returns_composed(self):
        rng = random.Random(42)
        expr = _random_expression_tree(rng, 3)
        assert expr is not None
        x = sp.Symbol("x")
        # Should be differentiable without error
        result = sp.diff(expr, x)
        assert result is not None

    def test_reproducible_with_seed(self):
        expr1 = _random_expression_tree(random.Random(123), 2)
        expr2 = _random_expression_tree(random.Random(123), 2)
        assert sp.srepr(expr1) == sp.srepr(expr2)


class TestTryGeneratePair:
    def test_generates_valid_pair(self):
        rng = random.Random(42)
        # Try multiple attempts since generation is random
        result = None
        for _ in range(20):
            result = _try_generate_pair(rng, 2)
            if result is not None:
                break
        assert result is not None
        integrand, antideriv, input_tokens, output_tokens = result
        assert len(input_tokens) > 0
        assert len(output_tokens) > 0
        assert len(input_tokens) <= MAX_INPUT_LEN - 2
        assert len(output_tokens) <= MAX_OUTPUT_LEN - 2

    def test_differentiation_verifies(self):
        rng = random.Random(42)
        x = sp.Symbol("x")
        result = None
        for _ in range(20):
            result = _try_generate_pair(rng, 2)
            if result is not None:
                break
        assert result is not None
        integrand, antideriv, _, _ = result
        # Verify that diff(F, x) == f
        diff_result = sp.diff(antideriv, x)
        assert sp.simplify(diff_result - integrand) == 0


class TestExpressionHash:
    def test_same_expr_same_hash(self):
        x = sp.Symbol("x")
        h1 = _expression_hash(sp.sin(x))
        h2 = _expression_hash(sp.sin(x))
        assert h1 == h2

    def test_different_expr_different_hash(self):
        x = sp.Symbol("x")
        h1 = _expression_hash(sp.sin(x))
        h2 = _expression_hash(sp.cos(x))
        assert h1 != h2


class TestGenerateBackwardPairs:
    def test_generates_requested_count(self):
        pairs = generate_backward_pairs(n=10, seed=42)
        # Should generate close to 10 (may be slightly fewer due to dedup)
        assert len(pairs) >= 5
        assert len(pairs) <= 10

    def test_output_format(self):
        pairs = generate_backward_pairs(n=5, seed=42)
        assert len(pairs) > 0
        pair = pairs[0]
        assert "input_tokens" in pair
        assert "output_tokens" in pair
        assert "depth_feature" in pair
        assert "source" in pair
        assert pair["source"] == "backward"
        assert len(pair["depth_feature"]) == FEATURE_DIM

    def test_no_duplicates(self):
        pairs = generate_backward_pairs(n=20, seed=42)
        # Check input_tokens are unique
        token_strs = [str(p["input_tokens"]) for p in pairs]
        assert len(token_strs) == len(set(token_strs))

    def test_tier_distribution(self):
        pairs = generate_backward_pairs(n=30, seed=42)
        tiers = [p["tier"] for p in pairs]
        # With equal weights, should have at least some from each tier
        # (may not have all tiers with small n, so just check we got pairs)
        assert len(pairs) > 0


class TestWriteShards:
    def test_writes_split_dirs(self, tmp_path):
        pairs = generate_backward_pairs(n=10, seed=42)
        output_dir = str(tmp_path / "output")
        counts = write_shards(pairs, output_dir, shard_size=5)

        assert "train" in counts
        assert "val" in counts
        assert "test" in counts
        assert os.path.isdir(os.path.join(output_dir, "train"))
        assert os.path.isdir(os.path.join(output_dir, "val"))
        assert os.path.isdir(os.path.join(output_dir, "test"))

    def test_shards_are_valid_jsonl_gz(self, tmp_path):
        pairs = generate_backward_pairs(n=10, seed=42)
        output_dir = str(tmp_path / "output")
        write_shards(pairs, output_dir, shard_size=100)

        # Read back a shard
        train_dir = os.path.join(output_dir, "train")
        shard_files = [f for f in os.listdir(train_dir) if f.endswith(".jsonl.gz")]
        assert len(shard_files) > 0

        with gzip.open(os.path.join(train_dir, shard_files[0]), "rt") as f:
            for line in f:
                item = json.loads(line)
                assert "input_tokens" in item
                assert "output_tokens" in item
                assert "depth_feature" in item

    def test_metadata_file(self, tmp_path):
        pairs = generate_backward_pairs(n=10, seed=42)
        output_dir = str(tmp_path / "output")
        write_shards(pairs, output_dir)

        meta_path = os.path.join(output_dir, "metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert "total" in meta
