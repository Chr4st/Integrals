"""High-level train function with curriculum scheduling."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader, Subset

logger = logging.getLogger(__name__)

from neurips.models.grammar import move_grammar_masks_to
from neurips.training.curriculum import (
    CompetenceCurriculum,
    CurriculumScheduler,
    build_index_maps,
    weighted_sample_indices,
)
from neurips.training.difficulty import batch_difficulty
from neurips.training.checkpoint import save_checkpoint, save_snapshot, snapshot_state
from neurips.training.trainer import TrainConfig, train_epoch, validate


def train(
    model: torch.nn.Module,
    train_data: list[dict],
    val_data: list[dict],
    config: TrainConfig,
    model_type: str,
    device: torch.device,
    output_dir: Path = Path("checkpoints"),
    collate_fn: Callable[[list[Any]], dict[str, Any]] | None = None,
) -> dict[str, list[float]]:
    """Full training loop with curriculum, AMP, early stopping, and checkpointing.

    Returns history dict with train_loss and val_loss per epoch.
    """
    model = model.to(device)
    move_grammar_masks_to(device)

    # Enable efficient SDPA kernels (FlashAttention-2, memory-efficient).
    if device.type == "cuda" and hasattr(torch.backends, "cuda"):
        for attr in ("enable_flash_sdp", "enable_mem_efficient_sdp"):
            if hasattr(torch.backends.cuda, attr):
                getattr(torch.backends.cuda, attr)(True)

    # Enable TF32 for matmuls on Ampere+ GPUs (~7% free throughput).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    # torch.compile for both model types on CUDA.
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as exc:
            logger.warning("torch.compile failed for %s: %s", model_type, exc)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # LR schedule: linear warmup then cosine annealing.
    warmup = config.warmup_epochs
    cosine_epochs = max(config.epochs - warmup, 1)
    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup]
    )

    use_amp = config.use_amp and device.type == "cuda"

    curriculum = CurriculumScheduler()

    # Competence-based curriculum (optional — activated via config).
    use_competence = config.curriculum_type == "competence"
    competence_curriculum: CompetenceCurriculum | None = None
    if use_competence:
        total_steps = config.epochs * (len(train_data) // config.batch_size)
        competence_curriculum = CompetenceCurriculum(total_steps=total_steps)
        logger.info("Using competence-based adaptive curriculum (T=%d)", total_steps)

    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    ckpt_threads: list[threading.Thread] = []

    use_pin = device.type == "cuda"
    n_workers = 4 if len(train_data) > 1000 else 0

    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=use_pin,
        persistent_workers=n_workers > 0,
    )

    # Pre-build index maps for O(1) curriculum filtering.
    index_maps = build_index_maps(train_data)

    # SWA: averaged model for final weight ensemble.
    swa_start = int(config.epochs * config.swa_start_pct)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
    swa_active = False

    for epoch in range(1, config.epochs + 1):
        # Curriculum filtering via pre-built maps (O(n_samples) not O(N)).
        active_tasks = curriculum.get_active_tasks(epoch)
        max_diff = curriculum.get_max_difficulty(epoch)
        indices = weighted_sample_indices(
            index_maps, active_tasks, max_diff, n_samples=len(train_data),
        )
        epoch_subset = Subset(train_data, indices)

        train_loader = DataLoader(
            epoch_subset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=use_pin,
            persistent_workers=n_workers > 0,
        )

        t_loss = train_epoch(
            model, train_loader, optimizer, model_type, device,
            grad_clip=config.grad_clip, use_amp=use_amp,
            grad_accum_steps=config.grad_accum_steps,
            label_smoothing=config.label_smoothing,
            use_pcgrad=config.use_pcgrad,
        )

        # Skip validation on early epochs (loss is noisy anyway).
        run_val = epoch >= 10 or epoch % 3 == 0
        if run_val:
            v_loss = validate(
                model, val_loader, model_type, device, use_amp=use_amp,
            )
        else:
            v_loss = float("inf")

        # Switch to SWA schedule at swa_start epoch.
        if epoch >= swa_start:
            if not swa_active:
                swa_active = True
                logger.info("SWA activated at epoch %d", epoch)
            swa_model.update_parameters(model)
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
            epoch, config.epochs, t_loss, v_loss, current_lr,
        )

        # Async checkpoint every 5 epochs (snapshot on main thread first).
        if epoch % 5 == 0:
            snap = snapshot_state(model, optimizer, scheduler, epoch)
            t = threading.Thread(
                target=save_snapshot,
                args=(snap, output_dir / f"{model_type}_epoch{epoch}.pt"),
            )
            t.start()
            ckpt_threads.append(t)

        # Save best model + early stopping (only when validated).
        if run_val and v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                output_dir / f"{model_type}_best.pt",
            )
        elif run_val:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, config.patience,
                )
                break

    # Ensure all background checkpoint writes complete before returning.
    for t in ckpt_threads:
        t.join()

    # Finalize SWA: update batch-norm stats and save averaged model.
    if swa_active:
        logger.info("Updating SWA batch-norm statistics...")
        # Build a loader from full training data for BN update.
        bn_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=n_workers,
            pin_memory=use_pin,
        )
        update_bn(bn_loader, swa_model, device=device)
        # Save the inner module (not AveragedModel wrapper) for checkpoint compat.
        save_checkpoint(
            swa_model.module, optimizer, scheduler, config.epochs,
            output_dir / f"{model_type}_swa.pt",
        )
        logger.info("SWA model saved to %s", output_dir / f"{model_type}_swa.pt")

    return history
