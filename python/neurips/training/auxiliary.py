"""Auxiliary loss heads — difficulty and depth prediction.

Forces the encoder to learn richer representations by predicting
metadata alongside the main integration task.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryHeads(nn.Module):
    """Lightweight heads for difficulty tier and expression depth."""

    def __init__(self, d_model: int, n_difficulty_tiers: int = 4) -> None:
        super().__init__()
        self.difficulty_head = nn.Linear(d_model, n_difficulty_tiers)
        self.depth_head = nn.Linear(d_model, 1)

    def forward(
        self, encoder_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict from mean-pooled encoder output.

        Args:
            encoder_output: [batch, seq_len, d_model] or [N, d_model].

        Returns:
            (difficulty_logits, depth_pred) — [batch, n_tiers], [batch, 1].
        """
        if encoder_output.dim() == 3:
            pooled = encoder_output.mean(dim=1)
        else:
            pooled = encoder_output
        return self.difficulty_head(pooled), self.depth_head(pooled)


def compute_auxiliary_loss(
    aux_heads: AuxiliaryHeads,
    encoder_output: torch.Tensor,
    batch: dict[str, Any],
    weight: float = 0.1,
) -> torch.Tensor:
    """Compute weighted auxiliary loss from batch metadata.

    Args:
        aux_heads: The auxiliary prediction heads.
        encoder_output: Encoder hidden states.
        batch: Must contain 'difficulty_id' (long) and optionally 'depth' (float).
        weight: Scaling factor for auxiliary losses.

    Returns:
        Scalar auxiliary loss.
    """
    diff_logits, depth_pred = aux_heads(encoder_output)
    loss = torch.tensor(0.0, device=encoder_output.device)

    if "difficulty_id" in batch:
        loss = loss + F.cross_entropy(diff_logits, batch["difficulty_id"])

    if "depth" in batch:
        loss = loss + F.mse_loss(depth_pred.squeeze(-1), batch["depth"].float())

    return weight * loss
