"""Tests for self-training with verification loop."""
import gzip
import json
import os
import tempfile

import pytest
import sympy as sp
import torch

from integral_engine.self_train import (
    generate_unlabeled_pool,
    merge_data_dirs,
    run_inference_round,
    verify_candidate,
    write_verified_shards,
)
from integral_engine.tokenizer import encode, to_prefix
from integral_engine.vocabulary import BOS_INDEX, EOS_INDEX, SEQ_TOKEN_TO_INDEX

_x = sp.Symbol("x")


class TestVerifyCandidate:
    def test_correct_antiderivative_accepted(self):
        cos_x = sp.cos(_x)
        tokens = to_prefix(cos_x)
        indices = [BOS_INDEX]
        for t in tokens:
            indices.append(SEQ_TOKEN_TO_INDEX.get(t, 0))
        indices.append(EOS_INDEX)

        integrand = sp.sin(_x)  # d/dx[-cos(x)] = sin(x)... wait
        # cos(x) differentiates to -sin(x), so integrand should be -sin(x)
        integrand = -sp.sin(_x)
        result = verify_candidate(indices, integrand)
        assert result is not None

    def test_wrong_antiderivative_rejected(self):
        sin_x = sp.sin(_x)
        tokens = to_prefix(sin_x)
        indices = [BOS_INDEX]
        for t in tokens:
            indices.append(SEQ_TOKEN_TO_INDEX.get(t, 0))
        indices.append(EOS_INDEX)

        integrand = sp.exp(_x)
        result = verify_candidate(indices, integrand)
        assert result is None

    def test_empty_sequence_rejected(self):
        result = verify_candidate([BOS_INDEX, EOS_INDEX], sp.sin(_x))
        assert result is None

    def test_malformed_tokens_rejected(self):
        result = verify_candidate([BOS_INDEX, 9999, EOS_INDEX], sp.sin(_x))
        assert result is None


class TestGenerateUnlabeledPool:
    def test_generates_requested_count(self):
        pool = generate_unlabeled_pool(20, seed=42)
        assert len(pool) >= 15  # Allow some generation failures

    def test_all_contain_x(self):
        pool = generate_unlabeled_pool(10, seed=42)
        for expr in pool:
            assert _x in expr.free_symbols

    def test_no_duplicates(self):
        pool = generate_unlabeled_pool(30, seed=42)
        reprs = [str(sp.srepr(e)) for e in pool]
        assert len(reprs) == len(set(reprs))

    def test_deterministic_with_seed(self):
        p1 = generate_unlabeled_pool(10, seed=99)
        p2 = generate_unlabeled_pool(10, seed=99)
        assert [str(e) for e in p1] == [str(e) for e in p2]


class TestWriteVerifiedShards:
    def test_writes_readable_shards(self):
        pairs = [{
            "input_tokens": ["sin", "x"],
            "output_tokens": ["Neg", "cos", "x"],
            "depth_feature": [0.0] * 608,
            "source": "self_train",
            "log_prob": -1.5,
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            write_verified_shards(pairs, tmpdir)
            shard_files = list(sorted(
                f for f in os.listdir(tmpdir) if f.endswith(".jsonl.gz")
            ))
            assert len(shard_files) == 1

            with gzip.open(os.path.join(tmpdir, shard_files[0]), "rt") as f:
                items = [json.loads(line) for line in f]
            assert len(items) == 1
            assert items[0]["input_tokens"] == ["sin", "x"]
            assert "source" not in items[0]  # source stripped for training


class TestMergeDataDirs:
    def _write_shard(self, directory, items):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "shard_0000.jsonl.gz")
        with gzip.open(path, "wt") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def test_merges_with_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.path.join(tmpdir, "orig")
            new_dir = os.path.join(tmpdir, "new")
            out_dir = os.path.join(tmpdir, "merged")

            orig_items = [{"input_tokens": [f"t{i}"], "output_tokens": ["x"],
                           "depth_feature": [0.0]} for i in range(100)]
            new_items = [{"input_tokens": [f"n{i}"], "output_tokens": ["x"],
                          "depth_feature": [0.0]} for i in range(50)]

            self._write_shard(orig_dir, orig_items)
            self._write_shard(new_dir, new_items)

            total = merge_data_dirs(orig_dir, new_dir, out_dir, new_ratio=0.1)
            # 100 original + ~11 new (10% ratio)
            assert total > 100
            assert total <= 150

    def test_handles_fewer_new_than_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.path.join(tmpdir, "orig")
            new_dir = os.path.join(tmpdir, "new")
            out_dir = os.path.join(tmpdir, "merged")

            orig_items = [{"input_tokens": ["x"], "output_tokens": ["x"],
                           "depth_feature": [0.0]} for _ in range(100)]
            new_items = [{"input_tokens": ["y"], "output_tokens": ["y"],
                          "depth_feature": [0.0]} for _ in range(3)]

            self._write_shard(orig_dir, orig_items)
            self._write_shard(new_dir, new_items)

            total = merge_data_dirs(orig_dir, new_dir, out_dir, new_ratio=0.5)
            assert total == 103
