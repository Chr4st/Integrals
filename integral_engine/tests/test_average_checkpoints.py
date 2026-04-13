"""Tests for checkpoint weight averaging."""
import os

import torch
import pytest

from integral_engine.average_checkpoints import (
    average_checkpoints,
    find_recent_checkpoints,
)


class TestFindRecentCheckpoints:
    def test_finds_epoch_files(self, tmp_path):
        for i in range(7):
            path = tmp_path / f"epoch_{i:04d}.pt"
            path.write_bytes(b"fake")
        paths = find_recent_checkpoints(str(tmp_path), k=5)
        assert len(paths) == 5
        # Should be the 5 most recent
        assert "epoch_0006.pt" in paths[-1]
        assert "epoch_0002.pt" in paths[0]

    def test_skips_best_and_averaged(self, tmp_path):
        for name in ["best.pt", "averaged.pt", "epoch_0001.pt", "epoch_0002.pt"]:
            (tmp_path / name).write_bytes(b"fake")
        paths = find_recent_checkpoints(str(tmp_path), k=5)
        basenames = [os.path.basename(p) for p in paths]
        assert "best.pt" not in basenames
        assert "averaged.pt" not in basenames

    def test_returns_empty_for_no_checkpoints(self, tmp_path):
        paths = find_recent_checkpoints(str(tmp_path), k=5)
        assert paths == []


class TestAverageCheckpoints:
    def test_averages_state_dicts(self, tmp_path):
        # Create two fake checkpoints with known weights
        sd1 = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        sd2 = {"weight": torch.tensor([3.0, 4.0, 5.0])}

        p1 = str(tmp_path / "ckpt1.pt")
        p2 = str(tmp_path / "ckpt2.pt")
        torch.save({"model_state_dict": sd1}, p1)
        torch.save({"model_state_dict": sd2}, p2)

        output = str(tmp_path / "averaged.pt")
        average_checkpoints([p1, p2], output)

        result = torch.load(output, weights_only=True)
        avg = result["model_state_dict"]["weight"]
        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(avg, expected)

    def test_preserves_metadata(self, tmp_path):
        sd = {"weight": torch.tensor([1.0])}
        p1 = str(tmp_path / "ckpt1.pt")
        torch.save({
            "model_state_dict": sd,
            "feature_schema_hash": "abc123",
        }, p1)

        output = str(tmp_path / "averaged.pt")
        average_checkpoints([p1], output)

        result = torch.load(output, weights_only=True)
        assert result["feature_schema_hash"] == "abc123"
        assert result["num_averaged"] == 1

    def test_raises_on_empty_list(self, tmp_path):
        with pytest.raises(ValueError, match="No checkpoint paths"):
            average_checkpoints([], str(tmp_path / "out.pt"))

    def test_three_checkpoint_average(self, tmp_path):
        sds = [
            {"w": torch.tensor([0.0, 0.0])},
            {"w": torch.tensor([3.0, 6.0])},
            {"w": torch.tensor([6.0, 12.0])},
        ]
        paths = []
        for i, sd in enumerate(sds):
            p = str(tmp_path / f"ckpt{i}.pt")
            torch.save({"model_state_dict": sd}, p)
            paths.append(p)

        output = str(tmp_path / "averaged.pt")
        average_checkpoints(paths, output)

        result = torch.load(output, weights_only=True)
        avg = result["model_state_dict"]["w"]
        expected = torch.tensor([3.0, 6.0])
        assert torch.allclose(avg, expected)
