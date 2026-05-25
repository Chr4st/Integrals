"""Sequence Transformer — encoder-decoder baseline (~95M params)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neurips.models.grammar import FEAT_ID

# ── Sinusoidal positional encoding ───────────────────────────────────


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
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """*x*: [batch, seq_len, d_model]."""
        return self.dropout(x + self.pe[:, : x.size(1)])


# ── Encoder ──────────────────────────────────────────────────────────

# Feature dimension: 344 (264 depth-encoded heads + 16 var + 40 sig + 24 task).
_FEAT_DIM = 344


class SeqEncoder(nn.Module):
    """Token embedding + sinusoidal PE + feature injection + transformer."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 640,
        n_heads: int = 10,
        n_layers: int = 10,
        d_ff: int = 2560,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)
        self.feat_proj = nn.Linear(_FEAT_DIM, d_model)

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
        """
        Args:
            src_ids: [batch, src_len] token ids.
            features: [batch, feat_dim] numeric features (injected at first
                      position whose token id == FEAT_ID, i.e. id 3).
            src_key_padding_mask: [batch, src_len] True → ignore.
        Returns:
            Encoder hidden states [batch, src_len, d_model].
        """
        x = self.token_emb(src_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        if features is not None:
            feat_emb = self.feat_proj(features)  # [batch, d_model]
            # Inject at every position whose id == FEAT_ID.
            feat_mask = (src_ids == FEAT_ID).unsqueeze(-1)  # [B, L, 1] bool
            x = x + feat_emb.unsqueeze(1) * feat_mask.to(x.dtype)

        return self.layers(x, src_key_padding_mask=src_key_padding_mask)


# ── Decoder ──────────────────────────────────────────────────────────


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
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPE(d_model, max_seq_len, dropout)

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
        """
        Args:
            tgt_ids: [batch, tgt_len] target token ids.
            memory: [batch, src_len, d_model] encoder output.
            tgt_mask: [tgt_len, tgt_len] causal mask.
            memory_key_padding_mask: [batch, src_len].
        Returns:
            Logits [batch, tgt_len, vocab_size].
        """
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


# ── Full model ───────────────────────────────────────────────────────


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
    ) -> None:
        super().__init__()
        self.encoder = SeqEncoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout
        )
        self.decoder = SeqDecoder(
            vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout
        )
        # Weight tying: share embeddings between encoder and decoder,
        # and tie output projection to decoder embedding (GPT-2/T5 style).
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
        """
        Args:
            src_ids: [batch, src_len].
            tgt_ids: [batch, tgt_len].
            features: [batch, 344] optional numeric features.
        Returns:
            Logits [batch, tgt_len, vocab_size].
        """
        memory = self.encoder(src_ids, features, src_key_padding_mask)
        return self.decoder(
            tgt_ids, memory, memory_key_padding_mask=memory_key_padding_mask
        )

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
