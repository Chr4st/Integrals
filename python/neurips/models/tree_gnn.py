"""Tree GNN encoder with variable-aware attention (~12.1M params with decoder)."""

from __future__ import annotations

from typing import Final

import torch
import torch.nn as nn

from .tree_decoder import TreeDecoder, TreeNode
from .tree_positional import TreePEType, build_tree_pe

# ── Constants ────────────────────────────────────────────────────────

_D_NODE: Final[int] = 256
_N_ROUNDS: Final[int] = 8
_N_HEADS: Final[int] = 8


# ── Node embedding ───────────────────────────────────────────────────


class NodeEmbedding(nn.Module):
    """Project heterogeneous node features into a shared space."""

    def __init__(
        self,
        n_symbols: int = 256,
        symbol_dim: int = 64,
        role_in: int = 12,
        role_dim: int = 64,
        struct_in: int = 40,
        struct_dim: int = 128,
    ) -> None:
        super().__init__()
        self.symbol_emb = nn.Embedding(n_symbols, symbol_dim)
        self.role_mlp = nn.Sequential(
            nn.Linear(role_in, 32), nn.SiLU(), nn.Linear(32, role_dim)
        )
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_in, 64), nn.SiLU(), nn.Linear(64, struct_dim)
        )

    def forward(
        self,
        symbol_ids: torch.Tensor,
        role_features: torch.Tensor,
        struct_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            symbol_ids: [N] long tensor.
            role_features: [N, 12].
            struct_features: [N, 40].
        Returns:
            [N, 256] node embeddings (64 + 64 + 128).
        """
        s = self.symbol_emb(symbol_ids)        # [N, 64]
        r = self.role_mlp(role_features)        # [N, 64]
        t = self.struct_mlp(struct_features)    # [N, 128]
        return torch.cat([s, r, t], dim=-1)     # [N, 256]


# ── Message passing ──────────────────────────────────────────────────


class MessageRound(nn.Module):
    """One round of parent↔child message passing."""

    def __init__(self, d: int = _D_NODE) -> None:
        super().__init__()
        self.parent_msg = nn.Linear(d, d)
        self.child_msg = nn.Linear(d, d)
        # input: node_emb ∥ parent_agg ∥ child_agg → 3d
        self.update = nn.Sequential(
            nn.Linear(d * 3, d * 2), nn.SiLU(), nn.Linear(d * 2, d)
        )
        self.norm = nn.LayerNorm(d)

    def forward(
        self, h: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h: [N, d] node embeddings.
            edge_index: [2, E] — row 0 = parent, row 1 = child.
        Returns:
            Updated [N, d] embeddings with residual + layernorm.
        """
        n = h.size(0)
        device = h.device

        parent_idx = edge_index[0]  # [E]
        child_idx = edge_index[1]   # [E]

        # Parent → child messages.
        p_msg = self.parent_msg(h[parent_idx])  # [E, d]
        # Aggregate at child positions via scatter_mean.
        child_agg = _scatter_mean(p_msg, child_idx, n, device)

        # Child → parent messages.
        c_msg = self.child_msg(h[child_idx])    # [E, d]
        parent_agg = _scatter_mean(c_msg, parent_idx, n, device)

        combined = torch.cat([h, parent_agg, child_agg], dim=-1)  # [N, 3d]
        return self.norm(h + self.update(combined))


def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Scatter mean aggregation — uses ``scatter_reduce`` when available."""
    d = src.size(-1)
    idx_expanded = index.unsqueeze(-1).expand(-1, d)

    # PyTorch >= 2.1: efficient fused scatter_reduce.
    if hasattr(torch.Tensor, "scatter_reduce"):
        out = torch.zeros(num_nodes, d, device=device)
        out.scatter_reduce_(0, idx_expanded, src, reduce="mean", include_self=False)
        return out

    # Fallback: manual scatter_add + count.
    out = torch.zeros(num_nodes, d, device=device)
    out.scatter_add_(0, idx_expanded, src)
    count = torch.zeros(num_nodes, 1, device=device)
    idx_1d = index.unsqueeze(-1)
    count.scatter_add_(0, idx_1d, src.new_ones(idx_1d.shape))
    count.clamp_(min=1.0)
    return out / count


class TreeMessagePassing(nn.Module):
    """8 rounds of bidirectional tree message passing."""

    def __init__(self, d: int = _D_NODE, n_rounds: int = _N_ROUNDS) -> None:
        super().__init__()
        self.rounds = nn.ModuleList([MessageRound(d) for _ in range(n_rounds)])

    def forward(self, node_embs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = node_embs
        for rnd in self.rounds:
            h = rnd(h, edge_index)
        return h


# ── Variable-aware attention ─────────────────────────────────────────


class VariableAwareAttention(nn.Module):
    """Multihead attention with a learned same-group dependency bias."""

    def __init__(self, d: int = _D_NODE, n_heads: int = _N_HEADS) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.dep_bias = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        node_embs: torch.Tensor,
        dep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            node_embs: [N, d] (unbatched) or [1, N, d].
            dep_mask: [N, N] float tensor — 1.0 where nodes share
                      the same variable dependency group, 0.0 otherwise.
        Returns:
            [N, d] refined embeddings (squeezed back if input was 2-d).
        """
        squeezed = node_embs.dim() == 2
        if squeezed:
            node_embs = node_embs.unsqueeze(0)  # [1, N, d]

        attn_bias: torch.Tensor | None = None
        if dep_mask is not None:
            # MHA expects [n_heads * batch, N, N] for 3-D mask.
            bias_2d = self.dep_bias * dep_mask  # [N, N]
            attn_bias = bias_2d.unsqueeze(0).expand(self.n_heads, -1, -1)  # [H, N, N]

        out, _ = self.attn(
            node_embs, node_embs, node_embs, attn_mask=attn_bias
        )
        if squeezed:
            out = out.squeeze(0)
        return out


# ── Full integrator model ────────────────────────────────────────────


class TreeIntegrator(nn.Module):
    """End-to-end tree-based integrator: embed → PE → message-pass → attend → decode."""

    def __init__(self, d: int = _D_NODE, pe_type: TreePEType = "none") -> None:
        super().__init__()
        self.node_emb = NodeEmbedding()
        self.tree_pe = build_tree_pe(pe_type, d)
        self.message_pass = TreeMessagePassing(d)
        self.var_attn = VariableAwareAttention(d)
        self.decoder = TreeDecoder(d)

    def forward(
        self,
        symbol_ids: torch.Tensor,
        role_features: torch.Tensor,
        struct_features: torch.Tensor,
        edge_index: torch.Tensor,
        dep_mask: torch.Tensor | None = None,
        depths: torch.Tensor | None = None,
        child_indices: torch.Tensor | None = None,
    ) -> list[TreeNode]:
        """
        Args:
            symbol_ids: [N] node symbol ids.
            role_features: [N, 12].
            struct_features: [N, 40].
            edge_index: [2, E] parent→child edges.
            dep_mask: [N, N] variable-dependency group mask.
            depths: [N] integer node depths (for depth_index PE).
            child_indices: [N] integer child positions (for depth_index PE).
        Returns:
            List of TreeNode from greedy decoding.
        """
        h = self.node_emb(symbol_ids, role_features, struct_features)

        if self.tree_pe is not None:
            from .tree_positional import DepthIndexPE, RWSE, LaplacianPE

            if isinstance(self.tree_pe, DepthIndexPE):
                if depths is not None and child_indices is not None:
                    h = h + self.tree_pe(depths, child_indices)
            elif isinstance(self.tree_pe, (RWSE, LaplacianPE)):
                h = h + self.tree_pe(edge_index, symbol_ids.size(0))

        h = self.message_pass(h, edge_index)
        h = self.var_attn(h, dep_mask)
        encoder_out = h.unsqueeze(0) if h.dim() == 2 else h
        return self.decoder.decode_tree(encoder_out)

    def count_parameters(self) -> int:
        """Total trainable parameters (~12.1M)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
