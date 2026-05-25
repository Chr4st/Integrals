"""Curriculum learning scheduler for staged training.

Supports static (original) and competence-based adaptive scheduling.
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Literal

import numpy as np


class CurriculumScheduler:
    """Controls which tasks and difficulty tiers are active per epoch."""

    def get_active_tasks(self, epoch: int) -> dict[str, float]:
        """Return task names to sampling weights for this epoch.

        Curriculum phases (matching paper Section 5, 90-epoch schedule):
          1-10:  univariate only, easy/medium
          11-30: add multivariate, up to hard
          31-60: all 5 tasks, all difficulties
          61-90: uniform sampling over all tasks and tiers
        """
        if epoch <= 10:
            return {"univariate": 1.0}
        if epoch <= 30:
            return {
                "univariate": 0.55,
                "multivariate": 0.45,
            }
        if epoch <= 60:
            return {
                "univariate": 0.25,
                "multivariate": 0.20,
                "parametric": 0.15,
                "definite": 0.20,
                "special_fn": 0.20,
            }
        return {
            "univariate": 0.20,
            "multivariate": 0.20,
            "parametric": 0.20,
            "definite": 0.20,
            "special_fn": 0.20,
        }

    def get_max_difficulty(self, epoch: int) -> str:
        """Return the hardest allowed difficulty tier for this epoch.

        Matches paper: epochs 1-10 easy/medium, 11-30 up to hard, 31+ all.
        """
        if epoch <= 10:
            return "medium"
        if epoch <= 30:
            return "hard"
        return "very_hard"


_DIFFICULTY_ORDER: tuple[str, ...] = ("easy", "medium", "hard", "very_hard")


def filter_data(
    train_data: list[dict],
    active_tasks: dict[str, float],
    max_difficulty: str,
) -> list[dict]:
    """Keep examples matching current curriculum stage."""
    max_idx = _DIFFICULTY_ORDER.index(max_difficulty)
    allowed = set(_DIFFICULTY_ORDER[: max_idx + 1])
    return [
        ex
        for ex in train_data
        if ex.get("task") in active_tasks
        and ex.get("difficulty_tier") in allowed
    ]


def weighted_sample(
    filtered_data: list[dict],
    task_weights: dict[str, float],
    n_samples: int,
) -> list[dict]:
    """Sample *n_samples* examples according to task weights."""
    if not filtered_data:
        return []

    tasks = list(task_weights.keys())
    weights = list(task_weights.values())
    by_task: dict[str, list[dict]] = {t: [] for t in tasks}
    for ex in filtered_data:
        t = ex.get("task")
        if t in by_task:
            by_task[t].append(ex)

    chosen_tasks = random.choices(tasks, weights=weights, k=n_samples)
    sampled: list[dict] = []
    for task in chosen_tasks:
        pool = by_task[task]
        if pool:
            sampled.append(random.choice(pool))
    return sampled


# ---------------------------------------------------------------------------
# Index-map based filtering (O(1) per epoch when pre-built)
# ---------------------------------------------------------------------------


def build_index_maps(
    data: list[dict],
) -> dict[str, dict[str, list[int]]]:
    """Pre-group dataset indices by (task, difficulty_tier).

    Returns ``{task: {difficulty: [indices]}}``.
    Call once at dataset load time; use with :func:`filter_indices`.
    """
    maps: dict[str, dict[str, list[int]]] = {}
    for i, ex in enumerate(data):
        task = ex.get("task", "")
        diff = ex.get("difficulty_tier", "")
        maps.setdefault(task, {}).setdefault(diff, []).append(i)
    return maps


def filter_indices(
    index_maps: dict[str, dict[str, list[int]]],
    active_tasks: dict[str, float],
    max_difficulty: str,
) -> list[int]:
    """Return indices matching the current curriculum stage.

    Uses pre-built *index_maps* from :func:`build_index_maps` for O(1)
    lookup instead of scanning the full dataset.
    """
    max_idx = _DIFFICULTY_ORDER.index(max_difficulty)
    allowed = set(_DIFFICULTY_ORDER[: max_idx + 1])
    indices: list[int] = []
    for task in active_tasks:
        task_map = index_maps.get(task, {})
        for diff in allowed:
            indices.extend(task_map.get(diff, []))
    return indices


def weighted_sample_indices(
    index_maps: dict[str, dict[str, list[int]]],
    active_tasks: dict[str, float],
    max_difficulty: str,
    n_samples: int,
) -> list[int]:
    """Sample *n_samples* indices using pre-built maps and task weights."""
    max_idx = _DIFFICULTY_ORDER.index(max_difficulty)
    allowed = set(_DIFFICULTY_ORDER[: max_idx + 1])

    tasks = list(active_tasks.keys())
    weights = list(active_tasks.values())

    # Collect per-task index pools.
    pools: dict[str, list[int]] = {}
    for task in tasks:
        pool: list[int] = []
        task_map = index_maps.get(task, {})
        for diff in allowed:
            pool.extend(task_map.get(diff, []))
        pools[task] = pool

    chosen_tasks = random.choices(tasks, weights=weights, k=n_samples)
    sampled: list[int] = []
    for task in chosen_tasks:
        pool = pools[task]
        if pool:
            sampled.append(random.choice(pool))
    return sampled


# ---------------------------------------------------------------------------
# Competence-based adaptive curriculum (Platanios et al. 2019)
# ---------------------------------------------------------------------------


class CompetenceCurriculum:
    """Adaptive curriculum using a competence function.

    competence(t) = min(1, sqrt(t * (1 - c_0^2) / T + c_0^2))

    At each step, only examples with difficulty <= competence are eligible.
    """

    def __init__(
        self,
        total_steps: int,
        c_0: float = 0.1,
        self_paced_start: float = 0.3,
        loss_window: int = 3,
    ) -> None:
        self._total_steps = total_steps
        self._c_0 = c_0
        self._self_paced_start = self_paced_start
        self._loss_window = loss_window
        self._step = 0
        self._per_example_losses: dict[int, deque[float]] = {}

    def competence(self, step: int | None = None) -> float:
        """Current competence level in [c_0, 1.0]."""
        t = step if step is not None else self._step
        return min(1.0, math.sqrt(
            t * (1.0 - self._c_0 ** 2) / self._total_steps + self._c_0 ** 2
        ))

    def step(self) -> None:
        """Advance internal step counter."""
        self._step += 1

    @property
    def is_self_paced(self) -> bool:
        """Whether self-paced mode is active (after warm-up fraction)."""
        return self._step >= self._total_steps * self._self_paced_start

    def record_loss(self, example_idx: int, loss: float) -> None:
        """Record per-example loss for self-paced scheduling."""
        if example_idx not in self._per_example_losses:
            self._per_example_losses[example_idx] = deque(maxlen=self._loss_window)
        self._per_example_losses[example_idx].append(loss)

    def get_sampling_weights(
        self, indices: list[int], difficulties: np.ndarray
    ) -> np.ndarray:
        """Compute sampling weights based on competence and self-pacing.

        Args:
            indices: dataset indices to consider.
            difficulties: [len(indices)] difficulty scores in [0, 1].

        Returns:
            [len(indices)] sampling weights (unnormalized).
        """
        comp = self.competence()
        weights = np.zeros(len(indices), dtype=np.float32)

        for i, (idx, diff) in enumerate(zip(indices, difficulties)):
            if diff > comp:
                continue

            if not self.is_self_paced:
                weights[i] = 1.0
                continue

            history = self._per_example_losses.get(idx)
            if history is None or len(history) < 2:
                weights[i] = 1.0
                continue

            recent = list(history)
            avg_loss = sum(recent) / len(recent)
            trend = recent[-1] - recent[0]

            if avg_loss < 0.1:
                weights[i] = 0.2  # mastered
            elif trend < -0.01:
                weights[i] = 2.0  # learning zone
            else:
                weights[i] = 0.5  # too hard / stalled

        total = weights.sum()
        if total < 1e-8:
            return np.ones(len(indices), dtype=np.float32)
        return weights

    def state_dict(self) -> dict:
        """Serialize curriculum state for checkpointing."""
        return {
            "step": self._step,
            "total_steps": self._total_steps,
            "c_0": self._c_0,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore curriculum state from checkpoint."""
        self._step = state["step"]
        self._total_steps = state["total_steps"]
        self._c_0 = state["c_0"]
