"""Born-Again self-distillation (Furlanello et al., ICML 2018).

Train a student model using KL divergence from a frozen teacher
combined with standard cross-entropy loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurips.data.vocab import PAD


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 3.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Combined KL + CE loss for self-distillation.

    Args:
        student_logits: [N, vocab] student predictions.
        teacher_logits: [N, vocab] frozen teacher predictions.
        targets: [N] ground truth token ids.
        alpha: Interpolation weight (0=pure CE, 1=pure KL).
        temperature: Softmax temperature for KL term.
        label_smoothing: Label smoothing for CE term.

    Returns:
        Scalar loss.
    """
    # Hard label loss (standard CE with label smoothing).
    ce_loss = F.cross_entropy(
        student_logits, targets, ignore_index=PAD,
        label_smoothing=label_smoothing,
    )

    # Soft label loss (KL divergence at elevated temperature).
    # Mask PAD positions.
    pad_mask = targets != PAD
    if pad_mask.any():
        s_log = F.log_softmax(student_logits[pad_mask] / temperature, dim=-1)
        t_prob = F.softmax(teacher_logits[pad_mask] / temperature, dim=-1)
        kl_loss = F.kl_div(s_log, t_prob, reduction="batchmean") * (temperature ** 2)
    else:
        kl_loss = torch.tensor(0.0, device=student_logits.device)

    return alpha * kl_loss + (1 - alpha) * ce_loss


def distill_step(
    student: nn.Module,
    teacher: nn.Module,
    batch: dict[str, Any],
    alpha: float = 0.5,
    temperature: float = 3.0,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """One distillation step for seq models: forward both, compute combined loss.

    Only supports SeqTransformer. Tree models have a non-differentiable
    decode_tree() interface that does not produce logits for KL divergence.
    """
    with torch.no_grad():
        t_logits = teacher(
            batch["src_ids"],
            batch["tgt_ids"][:, :-1],
            batch.get("features"),
        )

    s_logits = student(
        batch["src_ids"],
        batch["tgt_ids"][:, :-1],
        batch.get("features"),
    )
    targets = batch["tgt_ids"][:, 1:].contiguous().view(-1)

    return distillation_loss(
        s_logits.reshape(-1, s_logits.size(-1)),
        t_logits.reshape(-1, t_logits.size(-1)),
        targets,
        alpha=alpha,
        temperature=temperature,
        label_smoothing=label_smoothing,
    )
