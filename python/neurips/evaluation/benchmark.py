"""Evaluation and benchmarking for trained models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from tqdm import tqdm

from neurips.data.vocab import ID_TO_TOKEN
from neurips.evaluation.verify import (
    RustVerificationOracle,
    VerificationOracle,
)

try:
    from neurips._core import simplicity_score as _rust_simplicity_score
except ImportError:
    _rust_simplicity_score = None


def evaluate(
    model: Any,
    test_data: list[dict],
    oracle: VerificationOracle | RustVerificationOracle,
    tokenizer: Any,
    config: Any,
) -> dict[str, Any]:
    """Run model on test set, track solve rates by task and tier."""
    model.eval()
    results: dict[str, Any] = {
        "total": 0,
        "solved": 0,
        "by_task": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_tier": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_task_tier": defaultdict(lambda: {"total": 0, "solved": 0}),
    }

    with torch.no_grad():
        for example in tqdm(test_data, desc="Evaluating"):
            results["total"] += 1
            task = example.get("task", "univariate")
            tier = example.get("difficulty_tier", "medium")
            key = f"{task}_{tier}"

            solved = _sample_and_verify(
                model, example, oracle, tokenizer, config
            )

            results["solved"] += int(solved)
            results["by_task"][task]["total"] += 1
            results["by_task"][task]["solved"] += int(solved)
            results["by_tier"][tier]["total"] += 1
            results["by_tier"][tier]["solved"] += int(solved)
            results["by_task_tier"][key]["total"] += 1
            results["by_task_tier"][key]["solved"] += int(solved)

    return dict(results)


def _sample_and_verify(
    model: Any,
    example: dict,
    oracle: VerificationOracle | RustVerificationOracle,
    tokenizer: Any,
    config: Any,
) -> bool:
    """Generate N candidates and return True if any verifies."""
    n_samples = getattr(config, "n_samples", 25)

    integrand = example.get("integrand_prefix", "")
    task_info = {
        "task": example.get("task", "INDEF"),
        "var": example.get("var", "x"),
        "bounds": example.get("bounds", (0, 1)),
        "params": example.get("params", []),
    }

    candidates: list[str] = []
    for _ in range(n_samples):
        candidate = _generate_tree(model, example, tokenizer)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return False

    # Batch verify when the oracle supports it.
    if isinstance(oracle, RustVerificationOracle):
        results = oracle.verify_many(candidates, integrand, task_info)
        accepted = [
            c for c, ok in zip(candidates, results) if ok
        ]
    else:
        # Sequential fallback for SymPy oracle.
        accepted = [
            c for c in candidates
            if oracle.verify(c, integrand, task_info)
        ]

    if not accepted:
        return False

    # Rank accepted candidates by simplicity score,
    # return the best (simplest) rather than the first.
    if len(accepted) > 1 and _rust_simplicity_score is not None:
        accepted.sort(key=lambda c: _rust_simplicity_score(c))

    return True


def _generate_tree(
    model: Any,
    example: dict,
    tokenizer: Any,
) -> str | None:
    """Tree decoding for TreeIntegrator models."""
    device = next(model.parameters()).device

    sym_ids = example["symbol_ids"].to(device)
    role_feat = example["role_features"].to(device)
    struct_feat = example["struct_features"].to(device)
    edge_idx = example["edge_index"].to(device)
    dep_mask = example.get("dep_mask")
    if dep_mask is not None:
        dep_mask = dep_mask.to(device)

    h = model.node_emb(sym_ids, role_feat, struct_feat)
    h = model.message_pass(h, edge_idx)
    h = model.var_attn(h, dep_mask)
    encoder_out = h.unsqueeze(0) if h.dim() == 2 else h

    nodes = model.decoder.decode_tree(encoder_out)
    if not nodes:
        return None

    # BFS order = prefix order; map symbol_id -> token string.
    tokens = [ID_TO_TOKEN.get(n.symbol_id, "<UNK>") for n in nodes]
    result = " ".join(tokens)
    return result if result.strip() else None


# ---------------------------------------------------------------------------
# Sprint 3: Action-space evaluation
# ---------------------------------------------------------------------------

def evaluate_action_space(
    policy: Any,
    encoder: Any,
    test_data: list[dict],
    env_factory: Any,
    search_fn: Any,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate action-space model on test set.

    Args:
        policy: ActionPolicy model.
        encoder: Tree GNN encoder.
        test_data: Test examples with 'integrand_prefix' key.
        env_factory: Callable for IntegrationEnv.
        search_fn: greedy_search or beam_search from action_search.
        device: Torch device.
    """
    results: dict[str, Any] = {
        "total": 0,
        "solved": 0,
        "by_tier": defaultdict(lambda: {"total": 0, "solved": 0}),
        "avg_steps": 0.0,
        "traces": [],
    }
    total_steps = 0

    for example in tqdm(test_data, desc="Action-space eval"):
        results["total"] += 1
        tier = example.get("difficulty_tier", "medium")
        integrand = example.get("integrand_prefix", "")

        result = search_fn(
            policy=policy,
            encoder=encoder,
            env_factory=env_factory,
            integrand_prefix=integrand,
            var=example.get("var", "x"),
            device=device,
        )

        results["solved"] += int(result.solved)
        results["by_tier"][tier]["total"] += 1
        results["by_tier"][tier]["solved"] += int(result.solved)
        total_steps += result.total_steps

        if result.trace:
            results["traces"].append({
                "integrand": integrand,
                "solved": result.solved,
                "steps": result.total_steps,
                "actions": [t.action_name for t in result.trace],
            })

    if results["total"] > 0:
        results["avg_steps"] = total_steps / results["total"]

    return dict(results)
