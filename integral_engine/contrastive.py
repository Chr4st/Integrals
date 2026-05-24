"""Contrastive symbolic-numeric pre-training (Spec 6).

Aligns symbolic encoder representations with numeric evaluation embeddings
via InfoNCE contrastive loss. The trained symbolic encoder can then be
transferred to the IntegralTransformer for improved downstream performance.

Architecture:
  - NumericEncoder: MLP mapping fixed-point evaluations to d_model space
  - ContrastiveModel: Parallel symbolic + numeric encoders producing
    aligned embeddings for contrastive learning
  - InfoNCE loss with temperature-scaled cosine similarity
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from integral_engine.vocabulary import PAD_INDEX, SEQ_VOCAB_SIZE

# Fixed evaluation grid for numeric encoding
_DEFAULT_EVAL_POINTS = np.linspace(-5, 5, 64)


class NumericEncoder(nn.Module):
    """MLP encoder mapping numeric evaluations to d_model embeddings.

    Takes n_points function evaluations at fixed sample points and
    projects them into the shared embedding space.
    """

    def __init__(self, n_points: int = 64, hidden: int = 256, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_points, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map numeric evaluations to embedding space.

        Args:
            x: (batch, n_points) float tensor of function evaluations.

        Returns:
            (batch, d_model) embedding tensor.
        """
        return self.net(x)


class ContrastiveModel(nn.Module):
    """Dual-encoder model for contrastive symbolic-numeric alignment.

    The symbolic encoder mirrors the IntegralTransformer encoder architecture
    (nn.TransformerEncoder with batch_first=True, norm_first=True) so weights
    can be transferred after pre-training.
    """

    def __init__(
        self,
        vocab_size: int = SEQ_VOCAB_SIZE,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        n_eval_points: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # Symbolic encoder (matches IntegralTransformer.encoder architecture)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_INDEX)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.symbolic_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        # Numeric encoder
        self.numeric_encoder = NumericEncoder(
            n_points=n_eval_points, d_model=d_model,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def symbolic_pool(
        self, encoder_output: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool encoder output, excluding padding positions.

        Args:
            encoder_output: (batch, seq_len, d_model) encoder hidden states.
            mask: (batch, seq_len) bool tensor where True = padding.

        Returns:
            (batch, d_model) pooled representation.
        """
        if mask is None:
            return encoder_output.mean(dim=1)

        # Invert mask: True for real tokens, False for padding
        token_mask = ~mask
        # (batch, seq_len, 1) for broadcasting
        weights = token_mask.unsqueeze(-1).float()
        # Sum of real token embeddings / count of real tokens
        summed = (encoder_output * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        token_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        numeric_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute aligned symbolic and numeric embeddings.

        Args:
            token_ids: (batch, seq_len) int tensor of token indices.
            padding_mask: (batch, seq_len) bool tensor (True = padding).
            numeric_values: (batch, n_eval_points) float tensor.

        Returns:
            Tuple of (sym_embeddings, num_embeddings), each (batch, d_model).
        """
        # Symbolic path
        import math

        src_emb = self.embedding(token_ids) * math.sqrt(self.d_model)
        enc_out = self.symbolic_encoder(
            src_emb, src_key_padding_mask=padding_mask,
        )
        sym_emb = self.symbolic_pool(enc_out, padding_mask)

        # Numeric path
        num_emb = self.numeric_encoder(numeric_values)

        return sym_emb, num_emb


def info_nce_loss(
    sym_emb: torch.Tensor,
    num_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE contrastive loss between symbolic and numeric embeddings.

    Matching pairs are on the diagonal (sym[i] matches num[i]).
    Uses cosine similarity scaled by temperature.

    Args:
        sym_emb: (batch, d_model) symbolic embeddings.
        num_emb: (batch, d_model) numeric embeddings.
        temperature: scaling factor for similarity logits.

    Returns:
        Scalar loss tensor.
    """
    # L2-normalize for cosine similarity
    sym_norm = F.normalize(sym_emb, p=2, dim=1)
    num_norm = F.normalize(num_emb, p=2, dim=1)

    # Cosine similarity matrix: (batch, batch)
    similarity = torch.mm(sym_norm, num_norm.t()) / temperature

    # Labels: diagonal entries are the matching pairs
    batch_size = sym_emb.size(0)
    labels = torch.arange(batch_size, device=sym_emb.device)

    # Symmetric loss: sym->num and num->sym
    loss_sym_to_num = F.cross_entropy(similarity, labels)
    loss_num_to_sym = F.cross_entropy(similarity.t(), labels)

    return (loss_sym_to_num + loss_num_to_sym) / 2.0


def evaluate_at_points(
    expr: sp.Basic,
    var: sp.Symbol,
    points: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a SymPy expression at fixed sample points.

    NaN and Inf values are replaced with 0 for robustness in
    numeric encoding.

    Args:
        expr: SymPy expression to evaluate.
        var: integration variable.
        points: evaluation grid (default: linspace(-5, 5, 64)).

    Returns:
        numpy array of float evaluations, same length as points.
    """
    if points is None:
        points = _DEFAULT_EVAL_POINTS

    f = sp.lambdify(var, expr, modules=["numpy"])
    try:
        values = np.array(f(points), dtype=np.float64)
    except (TypeError, ValueError, ZeroDivisionError):
        values = np.zeros(len(points), dtype=np.float64)

    # Ensure correct shape for scalar expressions
    if values.ndim == 0:
        values = np.full(len(points), float(values), dtype=np.float64)

    # Replace non-finite values with 0
    values = np.where(np.isfinite(values), values, 0.0)

    return values.astype(np.float64)


def transfer_encoder_weights(
    contrastive_model: ContrastiveModel,
    integral_transformer: nn.Module,
) -> None:
    """Transfer symbolic encoder weights from contrastive model to IntegralTransformer.

    Copies the symbolic_encoder state_dict into integral_transformer.encoder,
    handling any key name differences between the two architectures.

    Args:
        contrastive_model: trained ContrastiveModel with symbolic_encoder.
        integral_transformer: target IntegralTransformer whose encoder
            will receive the pre-trained weights.
    """
    src_state = contrastive_model.symbolic_encoder.state_dict()
    tgt_state = integral_transformer.encoder.state_dict()

    # Map keys: both use nn.TransformerEncoder so keys should match directly
    mapped = {}
    for key in tgt_state:
        if key in src_state and src_state[key].shape == tgt_state[key].shape:
            mapped[key] = src_state[key]

    integral_transformer.encoder.load_state_dict(mapped, strict=False)
