"""Ablation studies for controlled experiments on tree GNN."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from neurips.evaluation.benchmark import evaluate
from neurips.evaluation.verify import VerificationOracle
from neurips.models.tree_gnn import TreeIntegrator, TreeMessagePassing
from neurips.training.train import train
from neurips.training.trainer import TrainConfig

logger = logging.getLogger(__name__)

# Ablation uses a reduced config: fewer epochs on a small data subset.
_ABLATION_EPOCHS = 20


def _make_ablation_config(base: TrainConfig) -> TrainConfig:
    return replace(base, epochs=_ABLATION_EPOCHS, patience=_ABLATION_EPOCHS)


def _train_and_eval(
    model: torch.nn.Module,
    train_data: list[dict],
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any,
    config: TrainConfig,
    device: str,
    collate_fn: Any = None,
) -> dict:
    """Train a model variant and evaluate it."""
    dev = torch.device(device)
    ablation_cfg = _make_ablation_config(config)

    train(
        model, train_data, train_data[:1000], ablation_cfg,
        model_type="tree", device=dev,
        output_dir=Path("/tmp/ablation_ckpts"),
        collate_fn=collate_fn,
    )

    @dataclass
    class _EvalCfg:
        n_samples: int = 25
        temperature: float = 0.7
        top_p: float = 0.95

    return evaluate(model, test_data, oracle, tokenizer, _EvalCfg())


def ablate_message_rounds(
    config: TrainConfig,
    train_data: list[dict],
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any = None,
    device: str = "cpu",
    collate_fn: Any = None,
) -> dict[int, dict]:
    """Test tree GNN with 2, 4, 6, 8, 12 message-passing rounds."""
    results: dict[int, dict] = {}
    for n_rounds in (2, 4, 6, 8, 12):
        logger.info("Ablation: message rounds = %d", n_rounds)
        model = TreeIntegrator()
        # Replace message passing module with different round count.
        model.message_pass = TreeMessagePassing(n_rounds=n_rounds)
        results[n_rounds] = _train_and_eval(
            model, train_data, test_data, oracle, tokenizer,
            config, device, collate_fn,
        )
    return results


def ablate_variable_attention(
    config: TrainConfig,
    train_data: list[dict],
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any = None,
    device: str = "cpu",
    collate_fn: Any = None,
) -> dict[str, dict]:
    """Compare tree GNN with vs without variable-aware attention."""
    results: dict[str, dict] = {}

    # With variable-aware attention (default).
    logger.info("Ablation: with variable-aware attention")
    model_with = TreeIntegrator()
    results["with_var_attn"] = _train_and_eval(
        model_with, train_data, test_data, oracle, tokenizer,
        config, device, collate_fn,
    )

    # Without: replace VariableAwareAttention with identity.
    logger.info("Ablation: without variable-aware attention")
    model_without = TreeIntegrator()
    model_without.var_attn = torch.nn.Identity()
    results["without_var_attn"] = _train_and_eval(
        model_without, train_data, test_data, oracle, tokenizer,
        config, device, collate_fn,
    )

    return results


def ablate_curriculum(
    config: TrainConfig,
    train_data: list[dict],
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any = None,
    device: str = "cpu",
    collate_fn: Any = None,
) -> dict[str, dict]:
    """Compare training with vs without curriculum scheduling."""
    results: dict[str, dict] = {}

    # With curriculum (default).
    logger.info("Ablation: with curriculum")
    model_with = TreeIntegrator()
    results["with_curriculum"] = _train_and_eval(
        model_with, train_data, test_data, oracle, tokenizer,
        config, device, collate_fn,
    )

    # Without: set all curriculum phases to epoch 0 (everything from start).
    logger.info("Ablation: without curriculum")
    model_without = TreeIntegrator()
    # Use a config that disables curriculum by setting warmup to 0.
    no_curriculum_cfg = replace(config, warmup_epochs=0)
    results["without_curriculum"] = _train_and_eval(
        model_without, train_data, test_data, oracle, tokenizer,
        no_curriculum_cfg, device, collate_fn,
    )

    return results


def ablate_n_samples(
    model: Any,
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any = None,
) -> dict[int, dict]:
    """Evaluate with N = 1, 5, 10, 25, 50 samples."""

    @dataclass
    class _Cfg:
        n_samples: int = 1
        temperature: float = 0.7
        top_p: float = 0.95

    results: dict[int, dict] = {}
    for n in (1, 5, 10, 25, 50):
        results[n] = evaluate(model, test_data, oracle, tokenizer, _Cfg(n_samples=n))
    return results
