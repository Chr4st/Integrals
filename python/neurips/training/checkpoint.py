"""Checkpoint save/load utilities for training state."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    path: Path,
) -> None:
    """Persist training state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path,
    )


def snapshot_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
) -> dict:
    """Capture an immutable copy of training state for background saving.

    Deep-copies optimizer state to avoid racing with the training loop
    (AdamW momentum tensors and step counters are mutable).
    """
    return {
        "epoch": epoch,
        "model_state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state_dict": copy.deepcopy(scheduler.state_dict()),
    }


def save_snapshot(snapshot: dict, path: Path) -> None:
    """Write a pre-built snapshot dict to disk (safe from background thread)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: CosineAnnealingLR | None = None,
) -> int:
    """Restore training state. Returns the saved epoch number."""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt["epoch"])
