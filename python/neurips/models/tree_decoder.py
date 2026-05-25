"""Top-down tree decoder — predicts tree structure from encoder output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn

import torch.nn.functional as F

from .activations import SwiGLU
from .grammar import arity_of

# ── Data structures ──────────────────────────────────────────────────

_D_MODEL: Final[int] = 256
_N_HEADS: Final[int] = 8
_N_LAYERS: Final[int] = 8
_MAX_LEVELS: Final[int] = 8


@dataclass(frozen=True)
class TreeNode:
    """Immutable decoded tree node."""

    symbol_id: int
    arity: int
    level: int
    parent_idx: int | None


# ── Decoder ──────────────────────────────────────────────────────────


class TreeDecoder(nn.Module):
    """Top-down autoregressive tree decoder.

    At each level, predicts a symbol for every active frontier node,
    then expands children based on predicted arity.
    """

    def __init__(
        self,
        d_model: int = _D_MODEL,
        n_heads: int = _N_HEADS,
        n_layers: int = _N_LAYERS,
        vocab_size: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(n_layers)]
        )
        self.cross_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        self.cross_ffs = nn.ModuleList(
            [SwiGLU(d_model) for _ in range(n_layers)]
        )
        self.ff_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )

        self.symbol_head = nn.Linear(d_model, vocab_size)
        self.child_init = nn.Linear(d_model, d_model * 2)  # left + right
        self.root_emb = nn.Parameter(torch.randn(d_model))

    # ── helpers ───────────────────────────────────────────────────

    def _attend(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Run *query* through all cross-attention layers.

        If *cached_kv* is provided, bypasses MHA's internal K/V
        projection using pre-computed multi-head K/V tensors and
        ``F.scaled_dot_product_attention`` directly.
        """
        d = self.d_model
        x = query
        for i, (attn, norm, ff, ff_norm) in enumerate(zip(
            self.cross_attn, self.cross_norms, self.cross_ffs, self.ff_norms
        )):
            residual = x
            x = norm(x)
            if cached_kv is not None:
                k, v = cached_kv[i]
                # Compute Q projection from the normed query.
                W = attn.in_proj_weight
                b = attn.in_proj_bias
                q = F.linear(x, W[:d], b[:d] if b is not None else None)
                B, T = x.shape[:2]
                n_h = attn.num_heads
                h_d = d // n_h
                q = q.view(B, T, n_h, h_d).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q, k, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, d)
                x = F.linear(attn_out, attn.out_proj.weight, attn.out_proj.bias)
            else:
                x, _ = attn(x, memory, memory)
            x = x + residual

            residual = x
            x = ff_norm(x)
            x = ff(x) + residual
        return x

    def _build_kv_cache(
        self, memory: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-compute key/value projections for all layers.

        Returns multi-head K/V tensors shaped [B, n_heads, S, head_dim]
        so ``_attend`` can skip MHA's internal K/V projection.
        """
        d = self.d_model
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        B, S = memory.shape[:2]
        for attn in self.cross_attn:
            n_h = attn.num_heads
            h_d = d // n_h
            W = attn.in_proj_weight  # [3d, d]
            b = attn.in_proj_bias    # [3d] or None
            k = F.linear(memory, W[d : 2 * d], b[d : 2 * d] if b is not None else None)
            v = F.linear(memory, W[2 * d :], b[2 * d :] if b is not None else None)
            k = k.view(B, S, n_h, h_d).transpose(1, 2)
            v = v.view(B, S, n_h, h_d).transpose(1, 2)
            cache.append((k, v))
        return cache

    # ── main decode ──────────────────────────────────────────────

    @torch.no_grad()
    def decode_tree(
        self,
        encoder_output: torch.Tensor,
        max_levels: int = _MAX_LEVELS,
    ) -> list[TreeNode]:
        """Greedily decode a single tree (batch dim squeezed).

        Args:
            encoder_output: [1, src_len, d_model] or [src_len, d_model].
            max_levels: depth budget.
        Returns:
            Flat list of *TreeNode* in breadth-first order.
        """
        if encoder_output.dim() == 2:
            encoder_output = encoder_output.unsqueeze(0)

        device = encoder_output.device
        nodes: list[TreeNode] = []
        cached_kv = self._build_kv_cache(encoder_output)

        # Frontier: list of (embedding, parent_idx).
        frontier: list[tuple[torch.Tensor, int | None]] = [
            (self.root_emb.unsqueeze(0).unsqueeze(0), None)  # [1,1,d]
        ]

        for level in range(max_levels):
            if not frontier:
                break

            # Stack frontier embeddings → [1, F, d_model].
            query = torch.cat([emb for emb, _ in frontier], dim=1)
            refined = self._attend(query, encoder_output, cached_kv=cached_kv)  # [1, F, d]

            logits = self.symbol_head(refined)  # [1, F, vocab]
            pred_ids = logits.argmax(dim=-1).squeeze(0)  # [F]

            next_frontier: list[tuple[torch.Tensor, int | None]] = []
            for i, (_, parent_idx) in enumerate(frontier):
                sid = pred_ids[i].item()
                a = arity_of(sid)
                node_idx = len(nodes)
                nodes.append(TreeNode(symbol_id=sid, arity=a, level=level, parent_idx=parent_idx))

                if a > 0:
                    children = self.child_init(refined[:, i : i + 1, :])  # [1,1,2d]
                    left = children[:, :, : self.d_model]
                    right = children[:, :, self.d_model :]
                    # For arity > 2, reuse right embedding for extra children.
                    for c in range(min(a, 2)):
                        emb = left if c == 0 else right
                        next_frontier.append((emb, node_idx))
                    for _ in range(max(a - 2, 0)):
                        next_frontier.append((right, node_idx))

            frontier = next_frontier

        return nodes

    def forward(
        self,
        node_query: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward: given node queries, predict symbol logits.

        Args:
            node_query: [batch, n_nodes, d_model] — teacher-forced embeddings.
            encoder_output: [batch, src_len, d_model].
        Returns:
            Logits [batch, n_nodes, vocab_size].
        """
        refined = self._attend(node_query, encoder_output)
        return self.symbol_head(refined)
