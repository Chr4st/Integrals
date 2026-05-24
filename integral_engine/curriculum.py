"""Curriculum learning: train on easy expressions first, progressively harder.

Provides a CurriculumSampler that filters dataset indices by difficulty
threshold, updated each epoch. Difficulty is measured by input token length
(proxy for expression complexity, consistent with dataset.py sorting).
"""
from __future__ import annotations

from typing import Iterator, List

import torch
from torch.utils.data import Sampler


def difficulty_score(item: dict) -> int:
    """Score difficulty by input token count (higher = harder)."""
    return len(item["input_tokens"])


class CurriculumSampler(Sampler[int]):
    """Sampler that progressively includes harder examples.

    At epoch 0, only examples with difficulty <= d_min are included.
    By warmup_epochs, all examples are included. Linear ramp between.
    """

    def __init__(
        self,
        difficulties: List[int],
        warmup_epochs: int = 10,
        seed: int = 42,
    ):
        self._difficulties = difficulties
        self._warmup_epochs = max(warmup_epochs, 1)
        self._seed = seed
        self._epoch = 0

        self._d_min = min(difficulties) if difficulties else 0
        self._d_max = max(difficulties) if difficulties else 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _threshold(self) -> float:
        progress = min(self._epoch / self._warmup_epochs, 1.0)
        return self._d_min + (self._d_max - self._d_min) * progress

    def __iter__(self) -> Iterator[int]:
        threshold = self._threshold()
        indices = [
            i for i, d in enumerate(self._difficulties) if d <= threshold
        ]
        gen = torch.Generator()
        gen.manual_seed(self._seed + self._epoch)
        perm = torch.randperm(len(indices), generator=gen).tolist()
        return iter([indices[p] for p in perm])

    def __len__(self) -> int:
        threshold = self._threshold()
        return sum(1 for d in self._difficulties if d <= threshold)


def build_curriculum_sampler(
    dataset_items: list,
    warmup_epochs: int = 10,
    seed: int = 42,
) -> CurriculumSampler:
    """Build a CurriculumSampler from dataset items."""
    difficulties = [difficulty_score(item) for item in dataset_items]
    return CurriculumSampler(difficulties, warmup_epochs=warmup_epochs, seed=seed)
