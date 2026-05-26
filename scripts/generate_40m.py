"""Generate 40M pairs matching L&C volume with 80/20 random split.

Usage:
    python scripts/generate_40m.py --output data/40m/
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

CHUNK_SIZE = 500_000

TASK_COUNTS = {
    "univariate": 40_000_000,
    "multivariate": 40_000_000,
    "definite": 40_000_000,
    "parametric": 40_000_000,
    "special": 40_000_000,
}

TOTAL = sum(TASK_COUNTS.values())


def generate_all(output_dir: Path, seed: int) -> None:
    from neurips._core import GenConfig, generate_batch

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig()
    random.seed(seed)

    all_path = output_dir / "all_pairs.jsonl"
    total_written = 0
    t0 = time.time()

    with open(all_path, "w") as f:
        for mode, count in TASK_COUNTS.items():
            mode_start = time.time()
            mode_written = 0

            for offset in range(0, count, CHUNK_SIZE):
                n = min(CHUNK_SIZE, count - offset)
                pairs = generate_batch(n, mode, cfg)

                for integrand, antideriv in pairs:
                    f.write(json.dumps({
                        "integrand": str(integrand),
                        "antiderivative": str(antideriv),
                        "mode": mode,
                    }) + "\n")
                    mode_written += 1

                elapsed = time.time() - mode_start
                rate = mode_written / elapsed if elapsed > 0 else 0
                print(
                    f"  [{mode}] {mode_written:>10,} / {count:,} "
                    f"({rate:,.0f} pairs/sec)"
                )

            total_written += mode_written
            print(
                f"  [{mode}] done: {mode_written:,} pairs in "
                f"{time.time() - mode_start:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"\nGenerated {total_written:,} pairs in {elapsed:.1f}s")
    print(f"Rate: {total_written / elapsed:,.0f} pairs/sec")
    print(f"File: {all_path} ({all_path.stat().st_size / 1e9:.2f} GB)")

    split(all_path, output_dir, seed)


def split(all_path: Path, output_dir: Path, seed: int) -> None:
    print("\nShuffling and splitting 80/20...")
    t0 = time.time()

    with open(all_path) as f:
        lines = f.readlines()

    print(f"  Loaded {len(lines):,} lines in {time.time() - t0:.1f}s")

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * 0.8)
    train_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"

    with open(train_path, "w") as f:
        f.writelines(train_lines)

    with open(test_path, "w") as f:
        f.writelines(test_lines)

    print(f"  Train: {len(train_lines):,} pairs -> {train_path}")
    print(f"  Test:  {len(test_lines):,} pairs -> {test_path}")
    print(f"  Split done in {time.time() - t0:.1f}s")

    all_path.unlink()
    print(f"  Removed {all_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/40m"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_all(args.output, args.seed)


if __name__ == "__main__":
    main()
