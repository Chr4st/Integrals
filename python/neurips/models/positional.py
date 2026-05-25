"""Positional encoding variants for the sequence transformer.

Supports: sinusoidal (Vaswani), RoPE (Su et al.), ALiBi (Press et al.), NoPE.
Select via config: [model.seq_transformer.pe] type = "rope" | "alibi" | "sinusoidal" | "nope"
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


PEType = Literal["sinusoidal", "rope", "alibi", "nope"]


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class NoPE(nn.Module):
    """No positional encoding — control variant."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (Su et al., RoFormer).

    Applied inside attention, not added to embeddings. Call ``apply_rotary``
    on Q and K tensors before computing attention scores.
    """

    def __init__(self, dim: int, max_len: int = 512, base: float = 10_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, max_len: int) -> None:
        t = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: [batch, heads, seq_len, head_dim]
        cos, sin: [1, seq_len, head_dim]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class ALiBiSlopes(nn.Module):
    """Attention with Linear Biases (Press et al.).

    Produces a per-head bias matrix added to attention scores.
    Zero learned parameters.
    """

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def _nearest_power_of_2(n: int) -> int:
            return 1 << (n - 1).bit_length() >> 1

        if (n_heads & (n_heads - 1)) == 0:
            return torch.tensor(
                [2 ** (-(2 ** -(math.log2(n_heads) - 3)) * i) for i in range(1, n_heads + 1)]
            )
        closest_pow2 = _nearest_power_of_2(n_heads)
        base_slopes = ALiBiSlopes._get_slopes(closest_pow2)
        extra_slopes = ALiBiSlopes._get_slopes(closest_pow2 * 2)
        extra_slopes = extra_slopes[1::2][: n_heads - closest_pow2]
        return torch.cat([base_slopes, extra_slopes])

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias [n_heads, seq_len, seq_len] to add to attn scores."""
        positions = torch.arange(seq_len, device=self.slopes.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        return self.slopes.unsqueeze(-1).unsqueeze(-1) * rel_pos.unsqueeze(0).abs().neg()


class DropInPE(nn.Module):
    """Wrapper that applies dropout only (for RoPE/ALiBi which don't add to embeddings)."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)


def build_pe(
    pe_type: PEType,
    d_model: int,
    max_len: int = 512,
    dropout: float = 0.1,
    n_heads: int = 10,
    base: float = 10_000.0,
) -> nn.Module:
    """Factory for positional encoding modules."""
    if pe_type == "sinusoidal":
        return SinusoidalPE(d_model, max_len, dropout)
    if pe_type == "nope":
        return NoPE(dropout)
    if pe_type == "rope":
        return DropInPE(dropout)
    if pe_type == "alibi":
        return DropInPE(dropout)
    raise ValueError(f"Unknown PE type: {pe_type}")
