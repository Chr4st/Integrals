"""Tree-aware positional encoding for the Tree GNN.

Variants:
  - depth_index: sinusoidal depth encoding + learnable child-index embedding
  - rwse: random-walk structural encoding (k-step landing probabilities)
  - laplacian: first-k eigenvectors of the normalized graph Laplacian
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

TreePEType = Literal["none", "depth_index", "rwse", "laplacian"]


class DepthIndexPE(nn.Module):
    """Encode tree depth (sinusoidal) and child index (learnable)."""

    def __init__(self, d_out: int = 256, max_depth: int = 32) -> None:
        super().__init__()
        self.child_emb = nn.Embedding(3, d_out)  # 0=root, 1=left, 2=right
        pe = torch.zeros(max_depth, d_out)
        position = torch.arange(max_depth).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_out, 2).float() * (-math.log(10_000.0) / d_out)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("depth_pe", pe)

    def forward(
        self, depths: torch.Tensor, child_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            depths: [N] integer depth of each node (0=root).
            child_indices: [N] integer (0=root, 1=left, 2=right).
        Returns:
            [N, d_out] positional features.
        """
        depth_enc = self.depth_pe[depths.clamp(max=self.depth_pe.size(0) - 1)]
        child_enc = self.child_emb(child_indices)
        return depth_enc + child_enc


class RWSE(nn.Module):
    """Random-Walk Structural Encoding.

    Computes k-step random walk landing probabilities as positional features.
    """

    def __init__(self, k: int = 8, d_out: int = 256) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k, d_out)

    def forward(self, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        Args:
            edge_index: [2, E] parent→child edges.
            n_nodes: total number of nodes.
        Returns:
            [N, d_out] RWSE features.
        """
        device = edge_index.device
        adj = torch.zeros(n_nodes, n_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

        deg = adj.sum(dim=1).clamp(min=1.0)
        transition = adj / deg.unsqueeze(1)

        rw_diag = torch.zeros(n_nodes, self.k, device=device)
        power = transition.clone()
        for step in range(self.k):
            rw_diag[:, step] = power.diagonal()
            if step < self.k - 1:
                power = power @ transition

        return self.proj(rw_diag)


class LaplacianPE(nn.Module):
    """Laplacian Positional Encoding using first-k eigenvectors."""

    def __init__(self, k: int = 8, d_out: int = 256) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(k, d_out)

    def forward(self, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        Args:
            edge_index: [2, E] parent→child edges.
            n_nodes: total number of nodes.
        Returns:
            [N, d_out] Laplacian PE features.
        """
        device = edge_index.device
        adj = torch.zeros(n_nodes, n_nodes, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0

        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        lap = torch.eye(n_nodes, device=device) - (
            deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        )

        k_actual = min(self.k, n_nodes - 1)
        if k_actual <= 0:
            return torch.zeros(n_nodes, self.proj.in_features, device=device)

        eigenvalues, eigenvectors = torch.linalg.eigh(lap)
        # Skip first eigenvector (constant), take next k
        pe_raw = eigenvectors[:, 1 : k_actual + 1]

        # Sign invariance: use absolute values
        pe_raw = pe_raw.abs()

        # Pad if fewer eigenvectors than k
        if pe_raw.size(1) < self.k:
            padding = torch.zeros(
                n_nodes, self.k - pe_raw.size(1), device=device
            )
            pe_raw = torch.cat([pe_raw, padding], dim=1)

        return self.proj(pe_raw)


def build_tree_pe(pe_type: TreePEType, d_out: int = 256, k: int = 8) -> nn.Module | None:
    """Factory for tree positional encoding modules."""
    if pe_type == "none":
        return None
    if pe_type == "depth_index":
        return DepthIndexPE(d_out)
    if pe_type == "rwse":
        return RWSE(k, d_out)
    if pe_type == "laplacian":
        return LaplacianPE(k, d_out)
    raise ValueError(f"Unknown tree PE type: {pe_type}")
