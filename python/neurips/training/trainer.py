"""Training loop for seq transformer and tree GNN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from contextlib import contextmanager

from neurips.data.vocab import PAD
from neurips.training.pcgrad import pcgrad_backward


@contextmanager
def _nullcontext():
    yield


def _amp_dtype() -> torch.dtype:
    """Select BF16 on Ampere+ GPUs, fall back to FP16 on older hardware."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 90
    batch_size: int = 256
    grad_clip: float = 1.0
    n_samples: int = 25
    temperature: float = 0.7
    top_p: float = 0.95
    use_amp: bool = True
    patience: int = 15
    warmup_epochs: int = 5
    grad_accum_steps: int = 1
    label_smoothing: float = 0.1
    swa_start_pct: float = 0.75
    swa_lr: float = 1e-5
    use_pcgrad: bool = False
    aux_weight: float = 0.1
    curriculum_type: str = "static"


def train_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    model_type: str,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    do_step: bool = True,
    grad_accum_steps: int = 1,
    label_smoothing: float = 0.1,
    use_pcgrad: bool = False,
) -> float:
    """Run one training step with optional BF16 AMP. Returns scalar loss.

    When *do_step* is False, gradients accumulate without an optimizer step.
    """
    ctx = torch.amp.autocast("cuda", dtype=_amp_dtype()) if use_amp else _nullcontext()
    with ctx:
        if use_pcgrad and "task_id" in batch:
            task_losses = _per_task_losses(model, batch, model_type, label_smoothing)
            total_loss = sum(task_losses) / len(task_losses)
        else:
            total_loss = _compute_loss(model, batch, model_type, label_smoothing)
            task_losses = None

    # Scale all losses uniformly for gradient accumulation.
    if grad_accum_steps > 1:
        total_loss = total_loss / grad_accum_steps
        if task_losses is not None:
            task_losses = [l / grad_accum_steps for l in task_losses]

    if use_pcgrad and task_losses is not None and len(task_losses) > 1:
        pcgrad_backward(task_losses, model)
    else:
        total_loss.backward()

    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    loss_val = total_loss.item()
    return loss_val * (grad_accum_steps if grad_accum_steps > 1 else 1)


def _compute_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    model_type: str,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """Forward pass and loss computation for either model type."""
    if model_type == "seq":
        logits = model(
            batch["src_ids"],
            batch["tgt_ids"][:, :-1],
            batch.get("features"),
        )
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["tgt_ids"][:, 1:].contiguous().view(-1),
            ignore_index=PAD,
            label_smoothing=label_smoothing,
        )
    if model_type == "tree":
        predicted = model(batch["input_trees"], batch["int_var"])
        return F.cross_entropy(
            predicted.reshape(-1, predicted.size(-1)),
            batch["target_symbols"].reshape(-1),
            ignore_index=PAD,
            label_smoothing=label_smoothing,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _per_task_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    model_type: str,
    label_smoothing: float = 0.1,
) -> list[torch.Tensor]:
    """Compute per-task losses for PCGrad by splitting batch on task_id."""
    task_ids = batch["task_id"]
    unique_tasks = task_ids.unique()

    if len(unique_tasks) <= 1:
        return [_compute_loss(model, batch, model_type, label_smoothing)]

    losses: list[torch.Tensor] = []
    for tid in unique_tasks:
        mask = task_ids == tid
        sub_batch = {k: v[mask] if isinstance(v, torch.Tensor) and v.size(0) == task_ids.size(0) else v for k, v in batch.items()}
        losses.append(_compute_loss(model, sub_batch, model_type, label_smoothing))
    return losses


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    label_smoothing: float = 0.1,
    use_pcgrad: bool = False,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    n_total = len(dataloader)
    remainder = n_total % grad_accum_steps
    tail_start = n_total - remainder if remainder > 0 else n_total

    for i, batch in enumerate(tqdm(dataloader, desc="Training", leave=False, total=n_total)):
        batch = _to_device(batch, device)
        do_step = (i + 1) % grad_accum_steps == 0
        # Use actual window size for the tail partial group.
        effective_accum = remainder if i >= tail_start else grad_accum_steps
        # Force step on the very last batch.
        if i == n_total - 1:
            do_step = True
        total_loss += train_step(
            model, batch, optimizer, model_type,
            grad_clip=grad_clip, use_amp=use_amp,
            do_step=do_step, grad_accum_steps=effective_accum,
            label_smoothing=label_smoothing,
            use_pcgrad=use_pcgrad,
        )
    return total_loss / max(n_total, 1)


def validate(
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    model_type: str,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    """Validate without gradients. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            batch = _to_device(batch, device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=_amp_dtype()):
                    total_loss += _compute_loss(
                        model, batch, model_type
                    ).item()
            else:
                total_loss += _compute_loss(
                    model, batch, model_type
                ).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def _to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move tensor values in a batch dict to the target device."""
    non_blocking = device.type == "cuda"
    return {
        k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
