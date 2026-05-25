"""Equivalence-class cross-entropy loss (Sprint 1).

Given K algebraically equivalent target sequences for each training
example, computes loss = min_k CE(prediction, target_k), encouraging
the model to produce *any* correct equivalent form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def equivalence_class_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute min-over-K cross-entropy loss.

    Args:
        logits: Model output logits, shape (B, T, V).
        targets: Equivalent target sequences, shape (B, K, T).
            K is the number of equivalent forms per example.
        mask: Optional boolean mask, shape (B, K, T). True = valid token.

    Returns:
        Scalar loss (mean over batch).
    """
    b, k, t = targets.shape
    v = logits.size(-1)

    # Expand logits to match K targets: (B, 1, T, V) -> (B, K, T, V)
    logits_expanded = logits.unsqueeze(1).expand(b, k, t, v)

    # Reshape for cross_entropy: (B*K*T, V) vs (B*K*T,)
    logits_flat = logits_expanded.reshape(-1, v)
    targets_flat = targets.reshape(-1)

    # Per-token CE (no reduction)
    per_token_ce = F.cross_entropy(
        logits_flat, targets_flat, reduction="none"
    )
    per_token_ce = per_token_ce.view(b, k, t)

    # Apply mask if provided
    if mask is not None:
        per_token_ce = per_token_ce * mask.float()
        # Per-sequence CE = sum of valid tokens / count of valid tokens
        seq_lengths = mask.float().sum(dim=-1).clamp(min=1.0)
        per_seq_ce = per_token_ce.sum(dim=-1) / seq_lengths
        # Mark fully-padded equivalents as +inf so min ignores them
        equiv_valid = mask.any(dim=-1)  # (B, K)
        per_seq_ce = per_seq_ce.masked_fill(~equiv_valid, float("inf"))
    else:
        per_seq_ce = per_token_ce.mean(dim=-1)  # (B, K)

    # Take the minimum CE across K equivalents
    min_ce = per_seq_ce.min(dim=1).values  # (B,)
    # Guard: if all K are padded for a sample, clamp to 0
    min_ce = min_ce.clamp(max=100.0)

    return min_ce.mean()


def standard_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard single-target cross-entropy (for comparison).

    Args:
        logits: (B, T, V)
        targets: (B, T)
        mask: Optional (B, T), True = valid.
    """
    b, t, v = logits.shape
    logits_flat = logits.reshape(-1, v)
    targets_flat = targets.reshape(-1)

    per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")
    per_token = per_token.view(b, t)

    if mask is not None:
        per_token = per_token * mask.float()
        seq_lengths = mask.float().sum(dim=-1).clamp(min=1.0)
        per_seq = per_token.sum(dim=-1) / seq_lengths
    else:
        per_seq = per_token.mean(dim=-1)

    return per_seq.mean()
