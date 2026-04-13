"""Tests for dataset processing and loading."""
import gzip
import json
import os
import tempfile

import torch
import pytest

from integral_engine.dataset import (
    _normalize_antiderivative,
    extract_unique_integrands,
)
from integral_engine.train import IntegralDataset, collate_fn
from integral_engine.vocabulary import BOS_INDEX, EOS_INDEX, FEATURE_DIM, PAD_INDEX
import sympy as sp


class TestNormalizeAntiderivative:
    def test_normalizes_at_zero(self):
        x = sp.Symbol("x")
        # F(x) = x^2 + 5, should become x^2 (since F(0)=5, subtract 5)
        result = _normalize_antiderivative(x**2 + 5, x)
        assert sp.simplify(result.subs(x, 0)) == 0

    def test_falls_back_to_one(self):
        x = sp.Symbol("x")
        # log(x) is undefined at x=0, should try x=1 → log(1)=0 → no change
        result = _normalize_antiderivative(sp.log(x), x)
        assert sp.simplify(result.subs(x, 1)) == 0

    def test_returns_as_is_if_no_eval_point_works(self):
        x = sp.Symbol("x")
        # Expression that might not simplify well - just verify no crash
        expr = sp.log(x) + sp.log(x - 2)
        result = _normalize_antiderivative(expr, x)
        assert result is not None


class TestExtractUniqueIntegrands:
    def test_extracts_unique(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("expression,rule\n")
            f.write("sin(x),rule1\n")
            f.write("cos(x),rule2\n")
            f.write("sin(x),rule3\n")
            path = f.name
        try:
            integrands = extract_unique_integrands(path)
            assert len(integrands) == 2
            assert "sin(x)" in integrands
            assert "cos(x)" in integrands
        finally:
            os.unlink(path)

    def test_skips_substitution_vars(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("expression,rule\n")
            f.write("sin(x),rule1\n")
            f.write("sin(_u),rule2\n")
            f.write("cos(_v),rule3\n")
            path = f.name
        try:
            integrands = extract_unique_integrands(path)
            assert len(integrands) == 1
            assert "sin(x)" in integrands
        finally:
            os.unlink(path)


class TestIntegralDataset:
    @pytest.fixture
    def shard_dir(self, tmp_path):
        """Create a minimal JSONL.gz shard for testing."""
        shard_dir = tmp_path / "train"
        shard_dir.mkdir()

        items = [
            {
                "input_tokens": ["sin", "x"],
                "output_tokens": ["Neg", "cos", "x"],
                "depth_feature": [0.0] * FEATURE_DIM,
            },
            {
                "input_tokens": ["cos", "x"],
                "output_tokens": ["sin", "x"],
                "depth_feature": [0.0] * FEATURE_DIM,
            },
        ]

        shard_path = shard_dir / "shard_0000.jsonl.gz"
        with gzip.open(str(shard_path), "wt") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        return str(shard_dir)

    def test_loads_items(self, shard_dir):
        dataset = IntegralDataset(shard_dir)
        assert len(dataset) == 2

    def test_bos_eos_framing(self, shard_dir):
        dataset = IntegralDataset(shard_dir)
        item = dataset[0]
        assert item["src"][0].item() == BOS_INDEX
        assert item["src"][-1].item() == EOS_INDEX
        assert item["tgt"][0].item() == BOS_INDEX
        assert item["tgt"][-1].item() == EOS_INDEX

    def test_feature_shape(self, shard_dir):
        dataset = IntegralDataset(shard_dir)
        item = dataset[0]
        assert item["depth_feature"].shape == (FEATURE_DIM,)
        assert item["depth_feature"].dtype == torch.float32


class TestCollation:
    def test_pads_to_max_length(self):
        batch = [
            {
                "src": torch.tensor([BOS_INDEX, 10, EOS_INDEX]),
                "tgt": torch.tensor([BOS_INDEX, 20, 21, EOS_INDEX]),
                "depth_feature": torch.zeros(FEATURE_DIM),
            },
            {
                "src": torch.tensor([BOS_INDEX, 10, 11, 12, EOS_INDEX]),
                "tgt": torch.tensor([BOS_INDEX, 20, EOS_INDEX]),
                "depth_feature": torch.zeros(FEATURE_DIM),
            },
        ]
        collated = collate_fn(batch)
        assert collated["src"].shape == (2, 5)  # max src length
        assert collated["tgt"].shape == (2, 4)  # max tgt length
        # Shorter sequences should be padded
        assert collated["src"][0, -1].item() == PAD_INDEX
        assert collated["src"][0, -2].item() == PAD_INDEX

    def test_mask_shape(self):
        batch = [
            {
                "src": torch.tensor([BOS_INDEX, 10, EOS_INDEX]),
                "tgt": torch.tensor([BOS_INDEX, 20, EOS_INDEX]),
                "depth_feature": torch.zeros(FEATURE_DIM),
            },
        ]
        collated = collate_fn(batch)
        assert collated["src_mask"].shape == collated["src"].shape
        assert collated["tgt_mask"].shape == collated["tgt"].shape
