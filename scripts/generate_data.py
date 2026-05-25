"""CLI: generate integration training data via Rust PyO3 backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# Default counts per mode
_DEFAULT_COUNTS: dict[str, int] = {
    "univariate": 4_500_000,
    "multivariate": 3_000_000,
    "parametric": 2_250_000,
    "definite": 2_250_000,
    "special_fn": 3_000_000,
}

_CHUNK_SIZE: int = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate symbolic integration data"
    )
    parser.add_argument(
        "--mode",
        choices=["all", *_DEFAULT_COUNTS.keys()],
        default="all",
        help="which task type(s) to generate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="number of pairs (default: mode-specific)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/",
        help="output directory for JSONL files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_CHUNK_SIZE,
        help="pairs per generation chunk (limits peak memory)",
    )
    return parser.parse_args()


def _generate_mode(
    mode: str, count: int, output_dir: Path, seed: int,
    chunk_size: int = _CHUNK_SIZE,
) -> int:
    """Generate data for a single mode in chunks. Returns count of valid pairs."""
    try:
        from neurips._core import generate_batch, verify_batch
    except ImportError:
        print(
            f"  [skip] neurips._core not available — "
            f"cannot generate {mode} data",
            file=sys.stderr,
        )
        return 0

    out_path = output_dir / f"{mode}.jsonl"
    total_valid = 0

    with open(out_path, "w") as f:
        for offset in range(0, count, chunk_size):
            n = min(chunk_size, count - offset)
            chunk_seed = int(hashlib.sha256(
                f"{mode}:{offset}:{seed}".encode()
            ).hexdigest(), 16) % (2**31)
            print(
                f"  [{mode}] chunk {offset:,}–{offset + n:,} of {count:,} "
                f"(seed={chunk_seed})..."
            )
            raw = generate_batch(mode, n, chunk_seed)

            valid = verify_batch(raw)
            for item in valid:
                f.write(json.dumps(item) + "\n")
            total_valid += len(valid)

    print(f"  Wrote {total_valid:,} verified {mode} pairs -> {out_path}")
    return total_valid


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = (
        list(_DEFAULT_COUNTS.keys())
        if args.mode == "all"
        else [args.mode]
    )

    total = 0
    for mode in modes:
        count = args.count or _DEFAULT_COUNTS[mode]
        total += _generate_mode(
            mode, count, output_dir, args.seed,
            chunk_size=args.chunk_size,
        )

    print(f"\nTotal valid pairs: {total:,}")


if __name__ == "__main__":
    main()
