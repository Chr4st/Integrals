"""Dispatch helpers for evaluation, ablation, and error analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from neurips.evaluation.verify import VerificationOracle
from neurips.training.trainer import TrainConfig


def run_evaluation(
    model_type: str,
    cfg: dict,
    test_data: list[dict],
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """Load a model and evaluate it on test data."""
    from neurips.evaluation.benchmark import evaluate

    oracle = VerificationOracle(timeout=4.0)
    training_cfg = cfg.get("training", {})
    config = TrainConfig(
        n_samples=training_cfg.get("n_samples", 25),
        temperature=training_cfg.get("temperature", 0.7),
        top_p=training_cfg.get("top_p", 0.95),
    )

    # TODO: load model from checkpoint_path based on model_type
    model: Any = None
    tokenizer: Any = None

    if model is None:
        print(
            f"  [skip] {model_type} model loading not yet wired",
            file=sys.stderr,
        )
        return {}

    return evaluate(model, test_data, oracle, tokenizer, config)


def run_ablation(
    ablation: str,
    cfg: dict,
    train_data: list[dict],
    test_data: list[dict],
) -> None:
    """Dispatch and print results for a single ablation study."""
    from neurips.evaluation.ablations import (
        ablate_curriculum,
        ablate_message_rounds,
        ablate_n_samples,
        ablate_variable_attention,
    )

    oracle = VerificationOracle(timeout=4.0)
    config = TrainConfig()

    dispatch = {
        "rounds": lambda: ablate_message_rounds(
            config, train_data, test_data, oracle
        ),
        "var_attn": lambda: ablate_variable_attention(
            config, train_data, test_data, oracle
        ),
        "curriculum": lambda: ablate_curriculum(
            config, train_data, test_data, oracle
        ),
        "n_samples": lambda: ablate_n_samples(
            None, test_data, oracle
        ),
    }
    results = dispatch[ablation]()
    print(json.dumps(results, indent=2, default=str))


def run_error_analysis(
    cfg: dict, test_data: list[dict]
) -> None:
    """Run error analysis on failures."""
    from neurips.evaluation.analysis import analyze_failures

    oracle = VerificationOracle(timeout=4.0)
    model: Any = None
    tokenizer: Any = None

    if model is None:
        print(
            "[skip] model loading not yet wired for error analysis",
            file=sys.stderr,
        )
        return

    results = analyze_failures(model, test_data, oracle, tokenizer)
    print("Failure counts:")
    for cat, count in sorted(results["counts"].items()):
        print(f"  {cat}: {count}")
