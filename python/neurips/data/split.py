"""Skeleton-based train/test splitting with zero leakage."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from neurips.data.prefix import parse_prefix, tokenize_prefix, tree_depth
from neurips.data.vocab import ARITY


def _compute_skeleton(prefix_str: str) -> str:
    """Compute structural skeleton by replacing leaves with placeholders.

    Keeps operator/function structure but replaces variables, constants,
    and numbers with '_' to capture the expression shape.
    """
    tokens = tokenize_prefix(prefix_str)
    skeleton_tokens: list[str] = []
    for tok in tokens:
        if tok in ARITY:
            skeleton_tokens.append(tok)
        else:
            skeleton_tokens.append("_")
    return " ".join(skeleton_tokens)


def deduplicate(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove exact duplicate integrands."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for pair in pairs:
        key = pair.get("integrand", "")
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique


def group_by_skeleton(
    pairs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group pairs by their structural skeleton."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        skel = _compute_skeleton(pair.get("integrand", ""))
        groups[skel].append(pair)
    return dict(groups)


def _classify_difficulty(depth: int, node_count: int) -> str:
    """Assign difficulty tier based on depth and node count."""
    score = depth * 2 + node_count
    if score <= 10:
        return "easy"
    if score <= 25:
        return "medium"
    if score <= 50:
        return "hard"
    return "very_hard"


def assign_tier(pair: dict[str, Any]) -> str:
    """Assign difficulty tier to an integrand pair."""
    prefix_str = pair.get("integrand", "")
    tokens = tokenize_prefix(prefix_str)
    nodes = parse_prefix(tokens)
    depth = tree_depth(nodes)
    return _classify_difficulty(depth, len(nodes))


def stratified_skeleton_split(
    pairs: list[dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split pairs with skeleton-level stratification.

    1. Group by (task_type, difficulty_tier)
    2. Within each cell, group by skeleton
    3. Shuffle skeletons (seeded)
    4. Assign 80% of skeletons to train, 20% to test
    """
    rng = random.Random(seed)

    # Annotate tiers (copy to avoid mutating input)
    pairs = [
        {**pair, "difficulty": pair.get("difficulty") or assign_tier(pair)}
        for pair in pairs
    ]

    # Group by (task, tier)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        key = (pair.get("task", "indef"), pair["difficulty"])
        cells[key].append(pair)

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for _cell_key, cell_pairs in sorted(cells.items()):
        # Group by skeleton within cell
        skel_groups = group_by_skeleton(cell_pairs)
        skeletons = list(skel_groups.keys())
        rng.shuffle(skeletons)

        split_idx = max(1, int(len(skeletons) * train_ratio))
        train_skels = set(skeletons[:split_idx])

        for skel, skel_pairs in skel_groups.items():
            if skel in train_skels:
                train.extend(skel_pairs)
            else:
                test.extend(skel_pairs)

    return train, test


def save_splits(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Save train and test splits to JSONL files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train), ("test.jsonl", test)]:
        path = out / name
        with open(path, "w", encoding="utf-8") as f:
            for pair in data:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
