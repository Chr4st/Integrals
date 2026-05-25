"""Dynamic Weight Averaging (Liu et al., CVPR 2019).

Adjusts per-task weights based on the rate of loss change,
replacing static curriculum weights after a warm-up phase.
"""

from __future__ import annotations

import torch


class DynamicWeightAverager:
    """Track per-task loss history and compute adaptive weights."""

    def __init__(self, temperature: float = 2.0) -> None:
        self.temperature = temperature
        self._prev_losses: dict[str, float] = {}
        self._curr_losses: dict[str, float] = {}

    def update(self, task: str, loss: float) -> None:
        """Record the average loss for a task in the current epoch."""
        self._curr_losses[task] = loss

    def step(self) -> None:
        """Advance epoch: current becomes previous."""
        self._prev_losses = dict(self._curr_losses)
        self._curr_losses = {}

    def get_weights(self, tasks: list[str]) -> dict[str, float]:
        """Compute DWA weights from loss rate of change.

        Returns uniform weights if insufficient history.
        """
        if not self._prev_losses:
            n = len(tasks)
            return {t: 1.0 / n for t in tasks}

        rates: list[float] = []
        valid_tasks: list[str] = []
        for t in tasks:
            prev = self._prev_losses.get(t)
            curr = self._curr_losses.get(t)
            if prev is not None and curr is not None and prev > 0:
                rates.append(curr / prev)
                valid_tasks.append(t)

        if not rates:
            n = len(tasks)
            return {t: 1.0 / n for t in tasks}

        # Softmax over loss rates scaled by temperature.
        rate_tensor = torch.tensor(rates) / self.temperature
        weights = torch.softmax(rate_tensor, dim=0).tolist()

        result = {}
        weight_idx = 0
        for t in tasks:
            if t in valid_tasks:
                result[t] = weights[weight_idx]
                weight_idx += 1
            else:
                result[t] = 1.0 / len(tasks)

        # Renormalize.
        total = sum(result.values())
        return {t: w / total for t, w in result.items()}
