"""Backward data generation: random F(x) → diff → f(x) → verified pair.

Implements the Lample & Charton (ICLR 2020) backward generation technique:
build random expression trees as antiderivatives F(x), differentiate to get
f(x), tokenize both, and output JSONL shards in the same format as dataset.py.

Usage:
    python -m integral_engine.data_generator --output data/backward/ --n 100000
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import signal
from typing import Dict, List, Optional, Tuple

import sympy as sp
from tqdm import tqdm

from integral_engine.feature_extractor import extract_features
from integral_engine.tokenizer import to_prefix
from integral_engine.vocabulary import MAX_INPUT_LEN, MAX_OUTPUT_LEN

_x = sp.Symbol("x")

# Operator pools by arity
_UNARY_OPS = [sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt, sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh]
_BINARY_OPS = [sp.Add, sp.Mul, sp.Pow]

# Coefficient set: small integers and simple rationals
_COEFF_INTS = list(range(-10, 11))
_COEFF_INTS.remove(0)
_COEFF_RATIONALS = [
    sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4), sp.Rational(1, 5),
    sp.Rational(1, 6), sp.Rational(2, 3), sp.Rational(3, 2), sp.Rational(3, 4),
    sp.Rational(2, 5), sp.Rational(5, 2),
    sp.Rational(-1, 2), sp.Rational(-1, 3), sp.Rational(-2, 3),
]

# Difficulty tiers: (min_depth, max_depth)
DIFFICULTY_TIERS = {
    "easy": (1, 2),
    "medium": (3, 4),
    "hard": (5, 6),
}


def _random_coefficient(rng: random.Random) -> sp.Basic:
    """Pick a random coefficient (integer or rational)."""
    if rng.random() < 0.7:
        return sp.Integer(rng.choice(_COEFF_INTS))
    return rng.choice(_COEFF_RATIONALS)


def _random_leaf(rng: random.Random) -> sp.Basic:
    """Generate a random leaf: x, integer constant, or coefficient * x."""
    choice = rng.random()
    if choice < 0.5:
        return _x
    if choice < 0.7:
        return _random_coefficient(rng)
    # coefficient * x or x**small_int
    if rng.random() < 0.5:
        return _random_coefficient(rng) * _x
    exp = rng.choice([2, 3, 4, sp.Rational(1, 2)])
    return _x ** exp


def _random_expression_tree(rng: random.Random, depth: int) -> sp.Basic:
    """Build a random expression tree of the given depth.

    At depth 0, return a leaf. Otherwise, pick a random operator and
    recursively build subtrees of depth-1.
    """
    if depth <= 0:
        return _random_leaf(rng)

    op_type = rng.random()

    if op_type < 0.4:
        # Unary function applied to subtree
        func = rng.choice(_UNARY_OPS)
        child = _random_expression_tree(rng, depth - 1)

        # Guard domains: log/sqrt need positive inner, asin/acos need bounded
        if func == sp.log:
            child = sp.Abs(child) + 1
        elif func == sp.sqrt:
            child = child ** 2 + 1
        elif func in (sp.asin, sp.acos):
            # Compose with sin to keep in [-1, 1]
            child = sp.sin(child)
        elif func == sp.tan:
            # Avoid blowups, wrap in something bounded
            child = _random_expression_tree(rng, max(0, depth - 2))

        return func(child)

    if op_type < 0.7:
        # Binary: Add or Mul
        op = rng.choice([sp.Add, sp.Mul])
        left = _random_expression_tree(rng, depth - 1)
        right = _random_expression_tree(rng, depth - 1)
        return op(left, right)

    if op_type < 0.85:
        # Power with small integer exponent
        base = _random_expression_tree(rng, depth - 1)
        exp = rng.choice([2, 3, sp.Rational(1, 2), sp.Rational(-1, 1)])
        return sp.Pow(base, exp)

    # Coefficient * subtree
    coeff = _random_coefficient(rng)
    subtree = _random_expression_tree(rng, depth - 1)
    return coeff * subtree


def _expression_hash(expr: sp.Basic) -> str:
    """Canonical hash for deduplication."""
    return hashlib.sha256(str(sp.srepr(expr)).encode()).hexdigest()[:16]


class _Timeout(Exception):
    pass


def _timeout_handler(signum: int, frame: object) -> None:
    raise _Timeout()


def _try_generate_pair(
    rng: random.Random,
    depth: int,
    timeout_seconds: float = 5.0,
) -> Optional[Tuple[sp.Basic, sp.Basic, List[str], List[str]]]:
    """Attempt to generate one (f, F) pair at the given depth.

    Returns (integrand, antiderivative, input_tokens, output_tokens) or None.
    Uses SIGALRM to enforce per-pair timeout on simplification.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(int(timeout_seconds) or 1)
    try:
        # Build random antiderivative F(x)
        antideriv = _random_expression_tree(rng, depth)

        # Simplify to canonical form (this is the slow step)
        antideriv = sp.nsimplify(sp.cancel(sp.expand(antideriv)), rational=True)

        # Differentiate to get integrand f(x) — always succeeds
        integrand = sp.diff(antideriv, _x)

        if integrand == 0:
            return None

        # Skip if result is too complex (Piecewise, unevaluated Integral)
        if integrand.has(sp.Piecewise, sp.Integral):
            return None
        if antideriv.has(sp.Piecewise, sp.Integral):
            return None

        # Check x appears in integrand
        if _x not in integrand.free_symbols:
            return None

        # Tokenize
        input_tokens = to_prefix(integrand)
        output_tokens = to_prefix(antideriv)

        # Check lengths
        if len(input_tokens) > MAX_INPUT_LEN - 2:
            return None
        if len(output_tokens) > MAX_OUTPUT_LEN - 2:
            return None
        if len(input_tokens) < 1 or len(output_tokens) < 1:
            return None

        return (integrand, antideriv, input_tokens, output_tokens)
    except (_Timeout, Exception):
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def generate_backward_pairs(
    n: int,
    max_attempts_per_pair: int = 10,
    seed: int = 42,
    tier_weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Generate n verified (integrand, antiderivative) pairs.

    Args:
        n: Target number of pairs.
        max_attempts_per_pair: Max random tries per desired pair.
        seed: Random seed for reproducibility.
        tier_weights: Weight for each difficulty tier. Default: equal.

    Returns:
        List of dicts with keys: input_tokens, output_tokens, depth_feature, source.
    """
    rng = random.Random(seed)
    weights = tier_weights or {t: 1.0 for t in DIFFICULTY_TIERS}
    tier_names = list(weights.keys())
    tier_probs = [weights[t] for t in tier_names]
    total = sum(tier_probs)
    tier_probs = [p / total for p in tier_probs]

    seen_hashes: set = set()
    pairs: List[Dict] = []
    attempts = 0
    max_total = n * max_attempts_per_pair

    pbar = tqdm(total=n, desc="Generating backward pairs")
    while len(pairs) < n and attempts < max_total:
        attempts += 1

        # Pick difficulty tier
        tier = rng.choices(tier_names, weights=tier_probs, k=1)[0]
        min_d, max_d = DIFFICULTY_TIERS[tier]
        depth = rng.randint(min_d, max_d)

        result = _try_generate_pair(rng, depth)
        if result is None:
            continue

        integrand, antideriv, input_tokens, output_tokens = result

        # Deduplicate by integrand hash
        h = _expression_hash(integrand)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # Extract depth features
        try:
            depth_feat = extract_features(integrand, _x).tolist()
        except Exception:
            continue

        pairs.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "depth_feature": depth_feat,
            "source": "backward",
            "tier": tier,
        })
        pbar.update(1)

    pbar.close()
    return pairs


def write_shards(
    pairs: List[Dict],
    output_dir: str,
    shard_size: int = 10000,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, int]:
    """Write pairs to train/val/test JSONL.gz shards.

    Returns counts per split.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle for random split
    shuffled = list(pairs)
    random.Random(0).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])

    splits = {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }

    counts: Dict[str, int] = {}
    for split_name, data in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for shard_idx in range(0, len(data), shard_size):
            shard = data[shard_idx:shard_idx + shard_size]
            shard_path = os.path.join(
                split_dir, f"shard_{shard_idx // shard_size:04d}.jsonl.gz",
            )
            with gzip.open(shard_path, "wt") as f:
                for item in shard:
                    # Write only the fields consumed by IntegralDataset
                    f.write(json.dumps({
                        "input_tokens": item["input_tokens"],
                        "output_tokens": item["output_tokens"],
                        "depth_feature": item["depth_feature"],
                    }) + "\n")

        counts[split_name] = len(data)

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"total": n, **counts}, f, indent=2)

    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate backward integration pairs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n", type=int, default=100000, help="Number of pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shard-size", type=int, default=10000, help="Pairs per shard")
    args = parser.parse_args()

    pairs = generate_backward_pairs(n=args.n, seed=args.seed)
    print(f"Generated {len(pairs)} pairs")

    counts = write_shards(pairs, args.output, shard_size=args.shard_size)
    for split, count in counts.items():
        print(f"  {split}: {count}")
