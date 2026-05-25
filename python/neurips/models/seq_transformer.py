"""Sequence Transformer — encoder-decoder baseline (~95M params).

Supports pluggable positional encoding (sinusoidal, RoPE, ALiBi, NoPE)
and FlashAttention-2 via PyTorch 2.x SDPA backend.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurips.models.grammar import FEAT_ID
from neurips.models.positional import (
    ALiBiSlopes,
    PEType,
    RotaryEmbedding,
    apply_rotary,
    build_pe,
)

_FEAT_DIM = 344


def _enable_flash_sdpa() -> None:
    """Enable FlashAttention-2 backend if available."""
    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass


_enable_flash_sdpa()


class SeqEncoder(nn.Module):
    """Token embedding + pluggable PE + feature injection + transformer."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 640,
        n_heads: int = 10,
        n_layers: int = 10,
        d_ff: int = 2560,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pe_type: PEType = "sinusoidal",
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = build_pe(pe_type, d_model, max_seq_len, dropout, n_heads, rope_base)
        self.feat_proj = nn.Linear(_FEAT_DIM, d_model)

        if pe_type == "rope":
            self.rotary = RotaryEmbedding(d_model // n_heads, max_seq_len, rope_base)
        else:
            self.rotary = None

        if pe_type == "alibi":
            self.alibi = ALiBiSlopes(n_heads)
        else:
            self.alibi = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        src_ids: torch.Tensor,
        features: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_emb(src_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        if features is not None:
            feat_emb = self.feat_proj(features)
            feat_mask = (src_ids == FEAT_ID).unsqueeze(-1)
            x = x + feat_emb.unsqueeze(1) * feat_mask.to(x.dtype)

        return self.layers(x, src_key_padding_mask=src_key_padding_mask)


class SeqDecoder(nn.Module):
    """Token embedding + PE + transformer decoder + output projection."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 640,
        n_heads: int = 10,
        n_layers: int = 10,
        d_ff: int = 2560,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pe_type: PEType = "sinusoidal",
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.pe_type = pe_type
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = build_pe(pe_type, d_model, max_seq_len, dropout, n_heads, rope_base)

        if pe_type == "rope":
            self.rotary = RotaryEmbedding(d_model // n_heads, max_seq_len, rope_base)
        else:
            self.rotary = None

        if pe_type == "alibi":
            self.alibi = ALiBiSlopes(n_heads)
        else:
            self.alibi = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_emb(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        if tgt_mask is None:
            tgt_len = tgt_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=tgt_ids.device
            )

        x = self.layers(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(x)


class SeqTransformer(nn.Module):
    """Encoder-decoder transformer for prefix-notation sequences."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 640,
        n_heads: int = 10,
        n_layers: int = 10,
        d_ff: int = 2560,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pe_type: PEType = "sinusoidal",
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        self.pe_type = pe_type
        self.encoder = SeqEncoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout,
            pe_type, rope_base,
        )
        self.decoder = SeqDecoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout,
            pe_type, rope_base,
        )
        self.decoder.token_emb.weight = self.encoder.token_emb.weight
        self.decoder.output_proj.weight = self.decoder.token_emb.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        features: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(src_ids, features, src_key_padding_mask)
        return self.decoder(
            tgt_ids, memory, memory_key_padding_mask=memory_key_padding_mask
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
