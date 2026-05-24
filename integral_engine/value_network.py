"""Lightweight MLP value estimator for MCTS node evaluation.

Takes pooled encoder memory and predicts a scalar [0, 1]
representing the likelihood that the current partial sequence
leads to a verified antiderivative.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """MLP that maps pooled encoder memory to a solvability score.

    Architecture: Linear(d_model, hidden_dim) -> ReLU
                  -> Linear(hidden_dim, 1) -> Sigmoid
    """

    def __init__(
        self, d_model: int = 512, hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pooled_memory: torch.Tensor) -> torch.Tensor:
        """Predict solvability score from pooled encoder output.

        Args:
            pooled_memory: Tensor of shape (..., d_model).

        Returns:
            Scalar prediction in [0, 1], shape (..., 1).
        """
        return self.net(pooled_memory)
