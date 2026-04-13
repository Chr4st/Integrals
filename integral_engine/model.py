"""Encoder-decoder Transformer for integral prediction.

Architecture:
  - 8 attention heads, 6 encoder/decoder layers, 512-dim embeddings
  - Feed-forward dim 2048, dropout 0.1
  - Depth feature injection: FEATURE_DIM → 512-dim linear, prepended as [FEAT] token
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    FEATURE_DIM,
    PAD_INDEX,
    SEQ_VOCAB_SIZE,
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class IntegralTransformer(nn.Module):
    """Encoder-decoder Transformer for integral prediction."""

    def __init__(
        self,
        vocab_size: int = SEQ_VOCAB_SIZE,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        feature_dim: int = FEATURE_DIM,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_INDEX)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_INDEX)
        self.feature_proj = nn.Linear(feature_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=512, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _make_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def encode(
        self,
        src: torch.Tensor,
        depth_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source with depth feature prepended as [FEAT] token."""
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        feat_emb = self.feature_proj(depth_features).unsqueeze(1)
        src_emb = torch.cat([feat_emb, src_emb], dim=1)
        src_emb = self.pos_encoder(src_emb)
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target sequence."""
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_mask = self._make_causal_mask(tgt.size(1), tgt.device)
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(output)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        depth_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward pass."""
        if src_key_padding_mask is not None:
            feat_mask = torch.zeros(
                src_key_padding_mask.size(0), 1,
                dtype=torch.bool, device=src_key_padding_mask.device,
            )
            memory_key_padding_mask = torch.cat(
                [feat_mask, src_key_padding_mask], dim=1,
            )
        else:
            memory_key_padding_mask = None

        memory = self.encode(src, depth_features, memory_key_padding_mask)
        return self.decode(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask)

    @staticmethod
    def _length_penalty(length: int, alpha: float) -> float:
        """Wu et al. (2016) length penalty."""
        return ((5.0 + length) / 6.0) ** alpha

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        depth_features: torch.Tensor,
        max_len: int = 256,
        beam_width: int = 10,
        length_penalty_alpha: float = 0.7,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[Tuple[List[int], float]]:
        """Beam search generation with length-normalized scoring.

        Returns list of (token_indices, raw_log_prob) sorted by
        length-normalized score.
        """
        self.eval()
        device = src.device

        if src_key_padding_mask is not None:
            feat_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            mem_mask = torch.cat([feat_mask, src_key_padding_mask], dim=1)
        else:
            mem_mask = None

        memory = self.encode(src, depth_features, mem_mask)

        beams: List[Tuple[List[int], float]] = [([BOS_INDEX], 0.0)]
        completed: List[Tuple[List[int], float]] = []

        def _normalized_score(seq: List[int], raw_score: float) -> float:
            return raw_score / self._length_penalty(len(seq), length_penalty_alpha)

        for _ in range(max_len):
            if not beams:
                break

            all_candidates: List[Tuple[List[int], float]] = []

            for seq, score in beams:
                if seq[-1] == EOS_INDEX:
                    completed.append((seq, score))
                    continue

                tgt = torch.tensor([seq], device=device)
                logits = self.decode(tgt, memory, memory_key_padding_mask=mem_mask)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                topk_probs, topk_indices = log_probs.topk(beam_width)

                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                    all_candidates.append((seq + [idx], score + prob))

            all_candidates.sort(
                key=lambda x: _normalized_score(x[0], x[1]), reverse=True,
            )
            beams = all_candidates[:beam_width]

            if len(completed) >= beam_width:
                break

        completed.extend(beams)
        completed.sort(
            key=lambda x: _normalized_score(x[0], x[1]), reverse=True,
        )
        return completed[:beam_width]

    @torch.no_grad()
    def sample(
        self,
        src: torch.Tensor,
        depth_features: torch.Tensor,
        max_len: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], float]:
        """Temperature sampling with nucleus (top-p) filtering.

        Returns (token_indices, raw_log_prob) for a single sample.
        """
        self.eval()
        device = src.device

        if src_key_padding_mask is not None:
            feat_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            mem_mask = torch.cat([feat_mask, src_key_padding_mask], dim=1)
        else:
            mem_mask = None

        memory = self.encode(src, depth_features, mem_mask)

        seq = [BOS_INDEX]
        total_log_prob = 0.0

        for _ in range(max_len):
            tgt = torch.tensor([seq], device=device)
            logits = self.decode(tgt, memory, memory_key_padding_mask=mem_mask)
            logits = logits[0, -1] / temperature

            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")

            # Restore original order and sample
            logits = torch.zeros_like(logits).scatter_(0, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            log_prob = F.log_softmax(logits, dim=-1)[idx].item()
            total_log_prob += log_prob
            seq.append(idx)

            if idx == EOS_INDEX:
                break

        return seq, total_log_prob
