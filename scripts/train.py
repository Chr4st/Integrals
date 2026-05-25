"""CLI: train seq transformer and/or tree GNN models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from _common import detect_device, load_config, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train neural symbolic integration models"
    )
    parser.add_argument(
        "--model", choices=["seq", "tree", "both"], default="both",
        help="which model(s) to train",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.toml",
        help="path to TOML config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/final/",
        help="directory with train/val JSONL files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/",
        help="directory for model checkpoints",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "mps"], default=None,
        help="device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="override epoch count from config",
    )
    return parser.parse_args()


def _train_seq(
    cfg: dict, train_data: list[dict], val_data: list[dict],
    device: torch.device, output_dir: Path,
) -> None:
    from neurips.models.seq_transformer import SeqTransformer
    from neurips.training.train import train
    from neurips.training.trainer import TrainConfig

    seq_cfg = cfg.get("model", {}).get("seq_transformer", {})
    training_cfg = cfg.get("training", {})

    model = SeqTransformer(
        vocab_size=seq_cfg.get("vocab_size", 256),
        d_model=seq_cfg.get("d_model", 640),
        n_heads=seq_cfg.get("n_heads", 10),
        n_layers=seq_cfg.get("n_layers", 10),
        d_ff=seq_cfg.get("d_ff", 2560),
        max_seq_len=seq_cfg.get("max_seq_len", 512),
        dropout=seq_cfg.get("dropout", 0.1),
    )
    print(f"SeqTransformer: {model.count_parameters():,} params")

    config = TrainConfig(
        lr=training_cfg.get("lr", 3e-4),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        epochs=training_cfg.get("epochs", 60),
        batch_size=training_cfg.get("batch_size", 256),
        grad_clip=training_cfg.get("grad_clip", 1.0),
    )
    train(model, train_data, val_data, config, "seq", device, output_dir)


def _train_tree(
    cfg: dict, train_data: list[dict], val_data: list[dict],
    device: torch.device, output_dir: Path,
) -> None:
    # TODO: import TreeIntegrator once phase 10-11 models are built
    print("[skip] Tree GNN not yet implemented", file=sys.stderr)


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    print(f"Device: {device}")

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs

    data_dir = Path(args.data_dir)
    train_data = load_jsonl(data_dir / "train.jsonl")
    val_data = load_jsonl(data_dir / "val.jsonl")
    print(f"Train: {len(train_data):,}  Val: {len(val_data):,}")

    output_dir = Path(args.output_dir)
    models = ["seq", "tree"] if args.model == "both" else [args.model]

    for m in models:
        print(f"\n{'='*60}\nTraining {m} model\n{'='*60}")
        if m == "seq":
            _train_seq(cfg, train_data, val_data, device, output_dir)
        elif m == "tree":
            _train_tree(cfg, train_data, val_data, device, output_dir)


if __name__ == "__main__":
    main()
