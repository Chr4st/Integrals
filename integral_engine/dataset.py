"""SIRD dataset processor: CSV → verified (integrand, antiderivative) pairs → JSONL.

Reads the SIRD CSV files containing (subexpression, rule_name) pairs,
extracts unique integrands, integrates each with SymPy (with timeout),
verifies by differentiation, and writes tokenized training data to
sharded JSONL files.

Usage:
    python -m integral_engine.dataset --input data/expr_2109869.csv --output data/processed/
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sympy as sp
from tqdm import tqdm

from integral_engine.feature_extractor import extract_features
from integral_engine.tokenizer import to_prefix
from integral_engine.vocabulary import MAX_INPUT_LEN, MAX_OUTPUT_LEN


def _normalize_antiderivative(result: sp.Basic, x: sp.Symbol) -> sp.Basic:
    """Normalize an antiderivative so that F(eval_point) = 0.

    Tries x=0 first, then x=1, then x=-1. If all evaluation points
    produce non-finite or complex values, returns the result as-is.
    """
    for eval_point in (sp.Integer(0), sp.Integer(1), sp.Integer(-1)):
        try:
            val = result.subs(x, eval_point)
            if val.is_finite and val.is_real:
                simplified_val = sp.simplify(val)
                if simplified_val.is_number:
                    return sp.simplify(result - simplified_val)
        except Exception:
            continue
    return result


def _integrate_and_verify(expr_str: str) -> Optional[str]:
    """Integrate expression, verify by differentiation, and normalize.

    Returns string representation of canonical antiderivative (F(0)=0),
    or None if integration/verification failed.
    """
    try:
        x = sp.Symbol("x")
        expr = sp.sympify(expr_str)

        if x not in expr.free_symbols:
            result = expr * x
        else:
            result = sp.integrate(expr, x)

        if result.has(sp.Integral):
            return None
        if result.has(sp.Piecewise):
            return None

        diff_result = sp.diff(result, x)
        if sp.simplify(diff_result - expr) != 0:
            return None

        result = _normalize_antiderivative(result, x)

        # Re-verify after normalization
        if sp.simplify(sp.diff(result, x) - expr) != 0:
            return None

        return str(result)
    except Exception:
        return None


def extract_unique_integrands(csv_path: str) -> List[str]:
    """Extract unique integrand expressions from SIRD CSV."""
    integrands = set()
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if row:
                expr_str = row[0].strip()
                if "_u" in expr_str or "_v" in expr_str:
                    continue
                integrands.add(expr_str)
    return sorted(integrands)


def process_dataset(
    csv_path: str,
    output_dir: str,
    max_workers: int = 4,
    timeout_per_expr: float = 10.0,
    shard_size: int = 10000,
) -> Dict[str, int]:
    """Process SIRD CSV into tokenized training data.

    Args:
        csv_path: Path to input CSV.
        output_dir: Directory for output JSONL shards.
        max_workers: Number of parallel workers for integration.
        timeout_per_expr: Timeout per expression in seconds.
        shard_size: Number of pairs per shard file.

    Returns:
        Dict with counts: total_unique, verified, train, val, test.
    """
    os.makedirs(output_dir, exist_ok=True)
    x = sp.Symbol("x")

    print(f"Reading integrands from {csv_path}...")
    integrands = extract_unique_integrands(csv_path)
    print(f"Found {len(integrands)} unique integrands")

    verified_pairs: List[Tuple[str, str]] = []
    print("Integrating and verifying...")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for expr_str in integrands:
            future = pool.submit(_integrate_and_verify, expr_str)
            futures[future] = expr_str

        for future in tqdm(futures, total=len(futures)):
            expr_str = futures[future]
            try:
                result = future.result(timeout=timeout_per_expr + 5)
                if result is not None:
                    verified_pairs.append((expr_str, result))
            except (TimeoutError, Exception):
                continue

    print(f"Verified {len(verified_pairs)} pairs out of {len(integrands)}")

    print("Tokenizing and extracting features...")
    processed: List[Dict] = []
    for integrand_str, antideriv_str in tqdm(verified_pairs):
        try:
            integrand = sp.sympify(integrand_str)
            antideriv = sp.sympify(antideriv_str)

            input_tokens = to_prefix(integrand)
            output_tokens = to_prefix(antideriv)

            if len(input_tokens) > MAX_INPUT_LEN - 2:
                continue
            if len(output_tokens) > MAX_OUTPUT_LEN - 2:
                continue

            depth_feat = extract_features(integrand, x).tolist()

            processed.append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "depth_feature": depth_feat,
            })
        except Exception:
            continue

    print(f"Processed {len(processed)} tokenized pairs")

    # Split: stratified by input length (proxy for depth)
    processed.sort(key=lambda p: len(p["input_tokens"]))
    train, val, test = [], [], []
    for i, item in enumerate(processed):
        if i % 10 == 0:
            val.append(item)
        elif i % 10 == 1:
            test.append(item)
        else:
            train.append(item)

    splits = {"train": train, "val": val, "test": test}
    counts = {"total_unique": len(integrands), "verified": len(verified_pairs)}

    for split_name, data in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for shard_idx in range(0, len(data), shard_size):
            shard = data[shard_idx : shard_idx + shard_size]
            shard_path = os.path.join(
                split_dir, f"shard_{shard_idx // shard_size:04d}.jsonl.gz",
            )
            with gzip.open(shard_path, "wt") as f:
                for item in shard:
                    f.write(json.dumps(item) + "\n")

        counts[split_name] = len(data)
        print(f"  {split_name}: {len(data)} pairs")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(counts, f, indent=2)

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process SIRD dataset")
    parser.add_argument("--input", required=True, help="Path to SIRD CSV")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    process_dataset(args.input, args.output, args.workers, args.timeout)
