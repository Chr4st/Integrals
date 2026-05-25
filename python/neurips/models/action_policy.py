"""Action-space policy head for step-by-step integration (Sprint 3).

Takes tree GNN encoder output (pooled tree embedding) and produces
action logits over the 4-action vocabulary:
  0: substitute(u = g(x))
  1: integrate_by_parts(u, dv)
  2: partial_fractions
  3: close(F)
"""

from __future__ import annotations

import torch
import torch.nn as nn


NUM_ACTIONS = 4


class ActionPolicy(nn.Module):
    """Greedy action policy head.

    Architecture: pooled tree embedding → MLP → action logits.
    Reuses the encoder backbone from the existing Tree GNN.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        """Compute action logits.

        Args:
            pooled_embedding: (B, embed_dim) from tree GNN encoder.

        Returns:
            Action logits (B, num_actions).
        """
        return self.mlp(pooled_embedding)

    def select_action(
        self,
        pooled_embedding: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select an action using the policy.

        Args:
            pooled_embedding: (B, embed_dim).
            temperature: Sampling temperature (0 = greedy).

        Returns:
            (action_ids, log_probs) — both shape (B,).
        """
        logits = self.forward(pooled_embedding)

        if temperature <= 0:
            action_ids = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(action_ids)
        else:
            scaled = logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_ids = dist.sample()
            log_probs = dist.log_prob(action_ids)

        return action_ids, log_probs


class SubstitutionParamHead(nn.Module):
    """Predicts the inner function g(x) for substitution actions.

    Outputs a pointer into the expression tree nodes, selecting which
    subtree to substitute. Uses attention over node embeddings.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        pooled: torch.Tensor,
        node_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pointer scores over tree nodes.

        Args:
            pooled: (B, embed_dim) — global pooled embedding.
            node_embeddings: (B, N, embed_dim) — per-node embeddings.
            mask: (B, N) boolean, True = valid node.

        Returns:
            Pointer logits (B, N).
        """
        q = self.query(pooled).unsqueeze(1)  # (B, 1, D)
        k = self.key(node_embeddings)  # (B, N, D)
        scores = (q * k).sum(dim=-1)  # (B, N)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        return scores


class IBPParamHead(nn.Module):
    """Predicts (u, dv) split for integration-by-parts actions.

    Outputs two pointer distributions over tree nodes:
    one for the u-factor and one for the dv-factor.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()
        self.u_head = SubstitutionParamHead(embed_dim)
        self.dv_head = SubstitutionParamHead(embed_dim)

    def forward(
        self,
        pooled: torch.Tensor,
        node_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pointer scores for u and dv.

        Returns:
            (u_logits, dv_logits) — both shape (B, N).
        """
        u_logits = self.u_head(pooled, node_embeddings, mask)
        dv_logits = self.dv_head(pooled, node_embeddings, mask)
        return u_logits, dv_logits
