"""CLI: evaluate trained models, run ablations, error analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import detect_device, load_config, load_jsonl
from _eval_dispatch import run_ablation, run_error_analysis, run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate neural symbolic integration models"
    )
    parser.add_argument(
        "--model", choices=["seq", "tree", "both"], default="both",
        help="which model(s) to evaluate",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="checkpoint path (or 'best')",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.toml",
        help="TOML config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/final/",
        help="directory with test/val JSONL",
    )
    parser.add_argument(
        "--split", choices=["test", "val"], default="test",
        help="which data split to evaluate on",
    )
    parser.add_argument(
        "--ablation",
        choices=["rounds", "var_attn", "curriculum", "n_samples"],
        default=None, help="run a specific ablation study",
    )
    parser.add_argument(
        "--error-analysis", action="store_true",
        help="run error analysis on failures",
    )
    return parser.parse_args()


def _resolve_checkpoint(
    checkpoint: str | None, model_type: str
) -> Path:
    if checkpoint == "best" or checkpoint is None:
        return Path("checkpoints") / f"{model_type}_best.pt"
    return Path(checkpoint)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = detect_device()

    data_dir = Path(args.data_dir)
    test_data = load_jsonl(data_dir / f"{args.split}.jsonl")
    print(f"Loaded {len(test_data):,} examples from {args.split}")

    if args.ablation:
        train_data = load_jsonl(data_dir / "train.jsonl")
        run_ablation(args.ablation, cfg, train_data, test_data)
        return

    if args.error_analysis:
        run_error_analysis(cfg, test_data)
        return

    models = (
        ["seq", "tree"] if args.model == "both" else [args.model]
    )
    for m in models:
        ckpt = _resolve_checkpoint(args.checkpoint, m)
        print(f"\nEvaluating {m} from {ckpt}")
        results = run_evaluation(m, cfg, test_data, ckpt, device)
        if results:
            total = results.get("total", 0)
            solved = results.get("solved", 0)
            rate = 100.0 * solved / max(total, 1)
            print(f"  Overall: {solved}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
