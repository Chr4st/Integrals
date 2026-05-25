"""Unified difficulty scorer for adaptive curriculum.

Combines tree depth, operation count, sequence length, and difficulty tier
into a single normalized [0, 1] score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifficultyScore:
    """Difficulty breakdown for a single training example."""

    tree_depth: int
    n_ops: int
    seq_length: int
    tier: int
    score: float


def compute_difficulty(
    tree_depth: int,
    n_ops: int,
    seq_length: int,
    tier: int,
    max_depth: int = 20,
    max_ops: int = 30,
    max_seq_len: int = 512,
    n_tiers: int = 4,
) -> DifficultyScore:
    """Compute normalized difficulty score in [0, 1].

    Uses weighted geometric mean of normalized components:
    - Tree depth (weight 0.3)
    - Operation count (weight 0.25)
    - Sequence length (weight 0.2)
    - Difficulty tier (weight 0.25)
    """
    d_norm = min(tree_depth / max_depth, 1.0)
    o_norm = min(n_ops / max_ops, 1.0)
    s_norm = min(seq_length / max_seq_len, 1.0)
    t_norm = tier / max(n_tiers - 1, 1)

    score = (
        d_norm * 0.30
        + o_norm * 0.25
        + s_norm * 0.20
        + t_norm * 0.25
    )

    return DifficultyScore(
        tree_depth=tree_depth,
        n_ops=n_ops,
        seq_length=seq_length,
        tier=tier,
        score=score,
    )


def batch_difficulty(
    depths: list[int],
    ops: list[int],
    lengths: list[int],
    tiers: list[int],
) -> np.ndarray:
    """Compute difficulty scores for a batch. Returns [N] float array."""
    scores = np.empty(len(depths), dtype=np.float32)
    for i in range(len(depths)):
        scores[i] = compute_difficulty(depths[i], ops[i], lengths[i], tiers[i]).score
    return scores
