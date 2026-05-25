"""Error analysis — classify why the model fails on each integral."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from neurips.evaluation.oracle import prefix_to_sympy
from neurips.evaluation.verify import VerificationOracle


def analyze_failures(
    model: Any,
    test_data: list[dict],
    oracle: VerificationOracle,
    tokenizer: Any,
) -> dict[str, Any]:
    """Classify every failure into a category.

    Returns counts and example lists per failure type.
    """
    import torch

    model.eval()
    categories: dict[str, list[dict]] = {
        "wrong_structure": [],
        "wrong_constants": [],
        "wrong_function": [],
        "timeout": [],
        "unparseable": [],
        "all_25_wrong": [],
    }

    with torch.no_grad():
        for example in tqdm(test_data, desc="Analyzing failures"):
            candidates = _sample_candidates(model, example, tokenizer)
            if not candidates:
                categories["unparseable"].append(example)
                continue

            # Check if any candidate verifies
            any_correct = _any_verifies(
                candidates, example, oracle
            )
            if any_correct:
                continue  # not a failure

            # Classify the best candidate
            best = candidates[0]
            target = example.get("antiderivative_prefix", "")
            category = _classify_failure(best, target)
            categories[category].append(example)

    counts = {k: len(v) for k, v in categories.items()}
    return {"counts": counts, "examples": categories}


def _sample_candidates(
    model: Any, example: dict, tokenizer: Any, n: int = 25
) -> list[str]:
    """Generate N candidate strings via _generate_candidate."""
    from neurips.evaluation.benchmark import _generate_candidate

    candidates: list[str] = []
    for _ in range(n):
        result = _generate_candidate(
            model, example, tokenizer, temperature=0.7, top_p=0.95
        )
        if result is not None:
            candidates.append(result)
    return candidates


def _any_verifies(
    candidates: list[str],
    example: dict,
    oracle: VerificationOracle,
) -> bool:
    """Return True if any candidate passes verification."""
    integrand = example.get("integrand_prefix", "")
    task_info = {
        "task": example.get("task", "INDEF"),
        "var": example.get("var", "x"),
        "bounds": example.get("bounds", (0, 1)),
        "params": example.get("params", []),
    }
    for cand in candidates:
        if oracle.verify(cand, integrand, task_info):
            return True
    return False


def _classify_failure(candidate: str, target: str) -> str:
    """Compare candidate to target and classify the failure mode."""
    try:
        prefix_to_sympy(candidate)
    except (ValueError, Exception):
        return "unparseable"

    try:
        prefix_to_sympy(target)
    except (ValueError, Exception):
        return "all_25_wrong"

    cand_tokens = candidate.strip().split()
    tgt_tokens = target.strip().split()

    if _same_structure(cand_tokens, tgt_tokens):
        if _same_functions(cand_tokens, tgt_tokens):
            return "wrong_constants"
        return "wrong_function"
    return "wrong_structure"


def _same_structure(a: list[str], b: list[str]) -> bool:
    """Check if two token lists have the same tree shape."""
    from neurips.data.vocab import ARITY

    def shape(tokens: list[str]) -> list[int]:
        return [ARITY.get(t, 0) for t in tokens]

    return shape(a) == shape(b)


def _same_functions(a: list[str], b: list[str]) -> bool:
    """Check if operator/function tokens match (ignoring leaves)."""
    from neurips.data.vocab import ARITY

    ops_a = [t for t in a if ARITY.get(t, 0) > 0]
    ops_b = [t for t in b if ARITY.get(t, 0) > 0]
    return ops_a == ops_b


