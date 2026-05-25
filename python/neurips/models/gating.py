"""Soft gating network between Sequence Transformer and Tree GNN.

Produces per-example weights [w_seq, w_tree] via a learned MLP.
Both experts always contribute — no routing collapse possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGating(nn.Module):
    """Learned soft gate between two expert encoders."""

    def __init__(
        self,
        seq_dim: int = 640,
        tree_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = seq_dim + tree_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        seq_features: torch.Tensor,
        tree_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_features: [batch, seq_dim] — CLS or mean-pooled seq encoder output.
            tree_features: [batch, tree_dim] — pooled tree GNN output.
        Returns:
            [batch, 2] softmax weights for [seq, tree].
        """
        combined = torch.cat([seq_features, tree_features], dim=-1)
        return F.softmax(self.gate(combined), dim=-1)


class GatedIntegrator(nn.Module):
    """Combines seq transformer and tree GNN predictions via soft gating."""

    def __init__(
        self,
        seq_dim: int = 640,
        tree_dim: int = 256,
        vocab_size: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.gating = SoftGating(seq_dim, tree_dim, hidden_dim)
        self.seq_proj = nn.Linear(seq_dim, vocab_size)
        self.tree_proj = nn.Linear(tree_dim, vocab_size)

    def forward(
        self,
        seq_features: torch.Tensor,
        tree_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_features: [batch, seq_dim] seq encoder CLS token.
            tree_features: [batch, tree_dim] tree GNN pooled output.
        Returns:
            [batch, vocab_size] combined logits.
        """
        weights = self.gating(seq_features, tree_features)
        w_seq = weights[:, 0:1]
        w_tree = weights[:, 1:2]

        seq_logits = self.seq_proj(seq_features)
        tree_logits = self.tree_proj(tree_features)

        return w_seq * seq_logits + w_tree * tree_logits
