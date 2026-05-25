"""Evaluation and benchmarking for trained models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from neurips.data.vocab import EOS, ID_TO_TOKEN, SOS
from neurips.evaluation.verify import (
    RustVerificationOracle,
    VerificationOracle,
)
from neurips.inference.beam_search import diverse_beam_search
from neurips.models.grammar import PrefixGrammarMask

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
    temperature = getattr(config, "temperature", 0.7)
    top_p = getattr(config, "top_p", 0.95)

    integrand = example.get("integrand_prefix", "")
    task_info = {
        "task": example.get("task", "INDEF"),
        "var": example.get("var", "x"),
        "bounds": example.get("bounds", (0, 1)),
        "params": example.get("params", []),
    }

    # Pre-compute encoder output once for all candidates.
    memory = _encode_once(model, example)

    # Use diverse beam search for seq models when temperature=0 (greedy mode).
    use_beam = getattr(config, "use_beam_search", False)

    candidates: list[str] = []
    if use_beam and memory is not None:
        candidates = diverse_beam_search(
            model.decoder,
            memory,
            tokenizer,
            beam_width=n_samples,
        )
    else:
        for _ in range(n_samples):
            candidate = _generate_candidate(
                model, example, tokenizer, temperature, top_p, memory=memory
            )
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

    # Sprint 1: rank accepted candidates by simplicity score,
    # return the best (simplest) rather than the first.
    if len(accepted) > 1 and _rust_simplicity_score is not None:
        accepted.sort(key=lambda c: _rust_simplicity_score(c))

    return True


def _encode_once(model: Any, example: dict) -> torch.Tensor | None:
    """Pre-compute encoder memory for seq models (None for tree models)."""
    is_seq = (
        hasattr(model, "encoder")
        and hasattr(model, "decoder")
        and hasattr(model.decoder, "output_proj")
    )
    if not is_seq:
        return None
    device = next(model.parameters()).device
    src_ids = example["src_ids"].long().unsqueeze(0).to(device)
    features = example["features"].unsqueeze(0).to(device)
    return model.encoder(src_ids, features)


def _generate_candidate(
    model: Any,
    example: dict,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    memory: torch.Tensor | None = None,
) -> str | None:
    """Generate one candidate string from the model.

    Dispatches to seq or tree decoding based on model architecture.
    """
    if memory is not None:
        return _generate_seq(model, example, tokenizer, temperature, top_p, memory=memory)
    return _generate_tree(model, example, tokenizer)


def _nucleus_sample(logits: torch.Tensor, top_p: float) -> int:
    """Sample a token from logits using nucleus (top-p) filtering."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Zero out tokens beyond the cumulative top_p threshold.
    cutoff_mask = cumulative - sorted_probs > top_p
    filtered = sorted_probs.masked_fill(cutoff_mask, 0.0)
    total = filtered.sum()
    if total <= 0:
        return sorted_idx[0].item()

    chosen = torch.multinomial(filtered / total, num_samples=1)
    return sorted_idx[chosen].item()


def _generate_seq(
    model: Any,
    example: dict,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_len: int = 512,
    memory: torch.Tensor | None = None,
) -> str | None:
    """Autoregressive decoding for SeqTransformer models."""
    device = next(model.parameters()).device
    if memory is None:
        src_ids = example["src_ids"].long().unsqueeze(0).to(device)
        features = example["features"].unsqueeze(0).to(device)
        memory = model.encoder(src_ids, features)
    generated: list[int] = [SOS]
    grammar = PrefixGrammarMask()

    for _ in range(max_len):
        tgt = torch.tensor([generated], device=device)
        logits = model.decoder(tgt, memory)
        step_logits = logits[0, -1, :]  # last position

        # Grammar constraint: disallow invalid tokens before softmax.
        mask = grammar.step(generated[-1]) if len(generated) > 1 else grammar.step(SOS)
        step_logits[~mask] = float("-inf")

        if temperature > 0:
            step_logits = step_logits / temperature
            token_id = _nucleus_sample(step_logits, top_p)
        else:
            token_id = step_logits.argmax().item()

        if token_id == EOS:
            break
        generated.append(token_id)

    prefix_str, _ = tokenizer.decode(generated)
    return prefix_str if prefix_str.strip() else None


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


def compare_decoding(
    model: Any,
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any,
) -> dict[str, dict]:
    """Compare greedy, beam-10, and sample-25 decoding."""
    from dataclasses import dataclass

    @dataclass
    class _Cfg:
        n_samples: int = 1
        temperature: float = 0.0
        top_p: float = 1.0
        use_beam_search: bool = False

    strategies: dict[str, _Cfg] = {
        "greedy": _Cfg(n_samples=1, temperature=0.0, top_p=1.0),
        "beam_10": _Cfg(n_samples=10, temperature=0.0, top_p=1.0, use_beam_search=True),
        "sample_25": _Cfg(n_samples=25, temperature=0.7, top_p=0.95),
    }
    return {
        name: evaluate(model, test_data, oracle, tokenizer, cfg)
        for name, cfg in strategies.items()
    }


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
