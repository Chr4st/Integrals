"""Rotary Position Embeddings (RoPE) for the IntegralTransformer.

Provides drop-in replacements for nn.TransformerEncoderLayer and
nn.TransformerDecoderLayer that inject positional information via
rotation of Q/K vectors rather than additive sinusoidal embeddings.

Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
Position Embedding" (2021). arXiv:2104.09864.

All modules use batch_first=True and norm_first=True (pre-norm) to
match the existing IntegralTransformer architecture in model.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Precompute and cache RoPE rotation frequencies.

    For dimension index i in [0, d_model/2), the frequency is:
        theta_i = 10000^(-2i / d_model)

    Forward returns (cos, sin) tensors of shape (1, seq_len, d_model)
    that can be applied to Q and K via apply_rotary_pos_emb.
    """

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.max_len = max_len

        # theta_i = 10000^(-2i / d_model) for i in [0, d_model/2)
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, d_model, 2).float() / d_model)
        )
        self.register_buffer("inv_freq", inv_freq)

        # Pre-build the cache for max_len positions.
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache up to seq_len positions."""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # (seq_len, d_model/2)
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate to cover full d_model: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        # (1, seq_len, d_model) for broadcasting with (batch, seq, d_model)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tensors for the given sequence length.

        Each has shape (1, seq_len, d_model).
        """
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :],
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rearrange pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...].

    This implements the "rotate by 90 degrees" part of the 2D rotation.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to Q and K tensors.

    The standard RoPE formula for each pair (x_2i, x_{2i+1}):
        [cos(m*theta_i)  -sin(m*theta_i)] [x_2i    ]
        [sin(m*theta_i)   cos(m*theta_i)] [x_{2i+1}]

    Implemented efficiently as: x * cos + rotate_half(x) * sin

    Args:
        q: Query tensor, shape (batch, nhead, seq_len, head_dim)
        k: Key tensor, shape (batch, nhead, seq_len, head_dim)
        cos: Cosine tensor, shape (1, 1, seq_len, head_dim)
        sin: Sine tensor, shape (1, 1, seq_len, head_dim)

    Returns:
        Rotated (q, k) with same shapes.
    """
    q_rotated = q * cos + _rotate_half(q) * sin
    k_rotated = k * cos + _rotate_half(k) * sin
    return q_rotated, k_rotated


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embeddings on Q and K.

    Unlike nn.MultiheadAttention, this applies RoPE rotations inside
    the attention computation (after Q/K projection, before dot product).

    Args:
        d_model: Total model dimension.
        nhead: Number of attention heads.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention with RoPE on Q and K.

        Args:
            query: (batch, tgt_len, d_model)
            key: (batch, src_len, d_model)
            value: (batch, src_len, d_model)
            attn_mask: (tgt_len, src_len) bool mask, True = masked.
            key_padding_mask: (batch, src_len) bool, True = padded.

        Returns:
            Output tensor of shape (batch, tgt_len, d_model).
        """
        batch, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # Project and reshape to (batch, nhead, seq_len, head_dim)
        q = self.q_proj(query).view(batch, tgt_len, self.nhead, self.head_dim)
        q = q.transpose(1, 2)
        k = self.k_proj(key).view(batch, src_len, self.nhead, self.head_dim)
        k = k.transpose(1, 2)
        v = self.v_proj(value).view(batch, src_len, self.nhead, self.head_dim)
        v = v.transpose(1, 2)

        # Get RoPE cos/sin for Q and K sequence lengths
        cos_q, sin_q = self.rope(tgt_len)
        cos_k, sin_k = self.rope(src_len)

        # Reshape for broadcasting: (1, 1, seq_len, head_dim)
        cos_q = cos_q.unsqueeze(1)
        sin_q = sin_q.unsqueeze(1)
        cos_k = cos_k.unsqueeze(1)
        sin_k = sin_k.unsqueeze(1)

        # Apply RoPE to Q and K independently
        q = q * cos_q + _rotate_half(q) * sin_q
        k = k * cos_k + _rotate_half(k) * sin_k

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask (True = ignore)
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(
                attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Apply key padding mask (True = ignore)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)
        # (batch, nhead, tgt_len, head_dim) -> (batch, tgt_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, tgt_len, self.d_model)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE attention.

    Drop-in replacement for nn.TransformerEncoderLayer. Uses pre-norm
    (norm_first=True) to match the existing IntegralTransformer.

    Architecture:
        x -> LayerNorm -> RoPEMultiheadAttention -> residual
          -> LayerNorm -> FFN -> residual

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: (batch, seq_len, d_model)
            src_mask: (seq_len, seq_len) attention mask, True = masked.
            src_key_padding_mask: (batch, seq_len) padding mask.

        Returns:
            Output of shape (batch, seq_len, d_model).
        """
        # Pre-norm self-attention
        x = self.norm1(src)
        x = self.self_attn(x, x, x, attn_mask=src_mask,
                           key_padding_mask=src_key_padding_mask)
        x = src + self.dropout1(x)

        # Pre-norm FFN
        residual = x
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout2(F.gelu(self.linear1(x2))))
        return residual + self.dropout3(x2)


class RoPEDecoderLayer(nn.Module):
    """Transformer decoder layer with RoPE on self-attention.

    Drop-in replacement for nn.TransformerDecoderLayer. Self-attention
    uses RoPE; cross-attention uses standard (no positional encoding
    needed for cross-attention since encoder positions are already
    encoded in the memory representations).

    Architecture:
        x -> LayerNorm -> RoPE-Self-Attention(causal) -> residual
          -> LayerNorm -> Standard-Cross-Attention -> residual
          -> LayerNorm -> FFN -> residual

    Args:
        d_model: Model dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Self-attention with RoPE
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention: standard (no RoPE needed for attending to memory)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            tgt: (batch, tgt_len, d_model)
            memory: (batch, src_len, d_model) encoder output.
            tgt_mask: (tgt_len, tgt_len) causal mask, True = masked.
            memory_mask: (tgt_len, src_len) cross-attention mask.
            tgt_key_padding_mask: (batch, tgt_len) padding mask.
            memory_key_padding_mask: (batch, src_len) padding mask.

        Returns:
            Output of shape (batch, tgt_len, d_model).
        """
        # Pre-norm self-attention with RoPE
        x = self.norm1(tgt)
        x = self.self_attn(x, x, x, attn_mask=tgt_mask,
                           key_padding_mask=tgt_key_padding_mask)
        x = tgt + self.dropout1(x)

        # Pre-norm cross-attention (standard, no RoPE)
        residual = x
        x2 = self.norm2(x)
        x2, _ = self.cross_attn(
            x2, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        x = residual + self.dropout2(x2)

        # Pre-norm FFN
        residual = x
        x3 = self.norm3(x)
        x3 = self.linear2(self.dropout3(F.gelu(self.linear1(x3))))
        return residual + self.dropout4(x3)
