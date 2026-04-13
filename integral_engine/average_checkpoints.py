"""Checkpoint weight averaging (Izmailov et al. 2018).

Averages the state_dict tensors of the last K checkpoints to produce
a single averaged checkpoint. Consistently yields +1-2% accuracy with
no additional training.

Usage:
    python -m integral_engine.average_checkpoints \
        --checkpoint-dir checkpoints/ \
        --output checkpoints/averaged.pt \
        --k 5
"""
from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import List

import torch


def find_recent_checkpoints(checkpoint_dir: str, k: int = 5) -> List[str]:
    """Find the K most recent checkpoint files by modification time.

    Looks for files matching epoch_*.pt or checkpoint_*.pt patterns.
    Falls back to any .pt files excluding best.pt and averaged.pt.
    """
    patterns = [
        os.path.join(checkpoint_dir, "epoch_*.pt"),
        os.path.join(checkpoint_dir, "checkpoint_*.pt"),
    ]

    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))

    if not paths:
        # Fallback: any .pt file except best.pt and averaged.pt
        all_pts = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        paths = [
            p for p in all_pts
            if os.path.basename(p) not in ("best.pt", "averaged.pt")
        ]

    # Sort by modification time (newest last)
    paths.sort(key=os.path.getmtime)

    return paths[-k:]


def average_checkpoints(
    checkpoint_paths: List[str],
    output_path: str,
) -> None:
    """Average state_dict tensors across checkpoints and save.

    Args:
        checkpoint_paths: List of checkpoint file paths to average.
        output_path: Where to save the averaged checkpoint.
    """
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided")

    print(f"Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    # Load all state dicts
    state_dicts = []
    reference_checkpoint = None
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state_dicts.append(ckpt["model_state_dict"])
            if reference_checkpoint is None:
                reference_checkpoint = ckpt
        else:
            state_dicts.append(ckpt)

    if not state_dicts:
        raise ValueError("No valid state_dicts found in checkpoints")

    # Average all tensors
    averaged: OrderedDict = OrderedDict()
    for key in state_dicts[0]:
        tensors = [sd[key].float() for sd in state_dicts]
        averaged[key] = sum(tensors) / len(tensors)

    # Build output checkpoint with metadata
    output_ckpt = {
        "model_state_dict": averaged,
        "averaged_from": [os.path.basename(p) for p in checkpoint_paths],
        "num_averaged": len(checkpoint_paths),
    }

    # Carry forward non-model metadata from reference checkpoint
    if reference_checkpoint is not None:
        for key in ("feature_schema_hash", "vocab_size", "config"):
            if key in reference_checkpoint:
                output_ckpt[key] = reference_checkpoint[key]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(output_ckpt, output_path)
    print(f"Saved averaged checkpoint to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Average checkpoint weights")
    parser.add_argument(
        "--checkpoint-dir", required=True,
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output", default="checkpoints/averaged.pt",
        help="Output path for averaged checkpoint",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of most recent checkpoints to average",
    )
    args = parser.parse_args()

    paths = find_recent_checkpoints(args.checkpoint_dir, args.k)
    if not paths:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        exit(1)

    average_checkpoints(paths, args.output)
