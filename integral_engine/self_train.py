"""Self-training with verification loop.

Implements expert iteration: run inference on unlabeled integrands, verify
predictions by SymPy differentiation (perfect oracle), add verified pairs
to training data, retrain. Each round expands the training set with examples
the model can already solve but that weren't in the original data.

Usage:
    python -m integral_engine.self_train \
        --checkpoint checkpoints/best.pt \
        --pool-size 10000 \
        --output data/self_train/ \
        --rounds 3
"""
from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog
import sympy as sp
from tqdm import tqdm

from integral_engine.data_generator import (
    DIFFICULTY_TIERS,
    _random_expression_tree,
)
from integral_engine.feature_extractor import extract_features
from integral_engine.tokenizer import encode, from_prefix, to_prefix
from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    PAD_INDEX,
    SEQ_TOKENS,
)

logger = structlog.get_logger(__name__)
_x = sp.Symbol("x")


def generate_unlabeled_pool(
    n: int,
    seed: int = 1000,
    tier_weights: Optional[Dict[str, float]] = None,
) -> List[sp.Basic]:
    """Generate a pool of random integrands (no antiderivatives needed).

    Uses backward generation to create F(x), differentiates to get f(x),
    but discards F(x). This gives integrands that are guaranteed to have
    antiderivatives (since we built them from F), but the model must
    rediscover F independently.
    """
    rng = random.Random(seed)
    weights = tier_weights or {"easy": 0.2, "medium": 0.4, "hard": 0.4}
    tier_names = list(weights.keys())
    tier_probs = [weights[t] for t in tier_names]
    total = sum(tier_probs)
    tier_probs = [p / total for p in tier_probs]

    pool: List[sp.Basic] = []
    seen: set = set()
    attempts = 0

    pbar = tqdm(total=n, desc="Generating unlabeled pool")
    while len(pool) < n and attempts < n * 20:
        attempts += 1
        tier = rng.choices(tier_names, weights=tier_probs, k=1)[0]
        min_d, max_d = DIFFICULTY_TIERS[tier]
        depth = rng.randint(min_d, max_d)

        try:
            antideriv = _random_expression_tree(rng, depth)
            antideriv = sp.nsimplify(sp.cancel(sp.expand(antideriv)), rational=True)
            integrand = sp.diff(antideriv, _x)

            if integrand == 0 or _x not in integrand.free_symbols:
                continue
            if integrand.has(sp.Piecewise, sp.Integral):
                continue

            tokens = to_prefix(integrand)
            if len(tokens) > MAX_INPUT_LEN - 2 or len(tokens) < 1:
                continue

            key = str(sp.srepr(integrand))
            if key in seen:
                continue
            seen.add(key)

            pool.append(integrand)
            pbar.update(1)
        except Exception:
            continue

    pbar.close()
    return pool


def verify_candidate(
    candidate_tokens: List[int],
    integrand: sp.Basic,
) -> Optional[sp.Basic]:
    """Verify a single candidate sequence by SymPy differentiation.

    Returns the verified antiderivative expression, or None.
    """
    tokens = []
    for idx in candidate_tokens:
        if idx in (BOS_INDEX, EOS_INDEX, PAD_INDEX):
            continue
        if 0 <= idx < len(SEQ_TOKENS):
            tokens.append(SEQ_TOKENS[idx])

    if not tokens:
        return None

    try:
        candidate_expr = from_prefix(tokens)
    except (ValueError, IndexError):
        return None

    try:
        diff = sp.diff(candidate_expr, _x)
        if sp.simplify(diff - integrand) == 0:
            return candidate_expr
    except Exception:
        pass

    return None


def run_inference_round(
    pool: List[sp.Basic],
    checkpoint_path: str,
    n_samples: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.95,
    min_log_prob: float = -50.0,
) -> List[Dict]:
    """Run model inference on unlabeled pool, verify, return verified pairs."""
    import torch
    from integral_engine.model import IntegralTransformer

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = raw.get("model_config", {})
    model = IntegralTransformer(**model_config) if model_config else IntegralTransformer()

    if "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    verified: List[Dict] = []

    for integrand in tqdm(pool, desc="Self-training inference"):
        try:
            src_indices = encode(integrand)
            src = torch.tensor([src_indices], device=device)
            depth_feat = torch.tensor(
                [extract_features(integrand, _x).tolist()],
                dtype=torch.float32, device=device,
            )

            for _ in range(n_samples):
                seq, log_prob = model.sample(
                    src, depth_feat, max_len=256,
                    temperature=temperature, top_p=top_p,
                )

                if log_prob < min_log_prob:
                    continue

                result = verify_candidate(seq, integrand)
                if result is not None:
                    input_tokens = to_prefix(integrand)
                    output_tokens = to_prefix(result)

                    if (len(output_tokens) <= MAX_OUTPUT_LEN - 2
                            and len(input_tokens) <= MAX_INPUT_LEN - 2):
                        feat = extract_features(integrand, _x).tolist()
                        verified.append({
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "depth_feature": feat,
                            "source": "self_train",
                            "log_prob": log_prob,
                        })
                        break

        except Exception:
            logger.debug("self_train_inference_error", exc_info=True)
            continue

    logger.info("self_train_round_complete",
                pool_size=len(pool), verified=len(verified))
    return verified


def write_verified_shards(
    pairs: List[Dict],
    output_dir: str,
    shard_size: int = 10000,
) -> None:
    """Write verified self-training pairs to JSONL.gz shards."""
    os.makedirs(output_dir, exist_ok=True)

    for shard_idx in range(0, len(pairs), shard_size):
        shard = pairs[shard_idx : shard_idx + shard_size]
        shard_path = os.path.join(
            output_dir, f"shard_{shard_idx // shard_size:04d}.jsonl.gz",
        )
        with gzip.open(shard_path, "wt") as f:
            for item in shard:
                serializable = {
                    "input_tokens": item["input_tokens"],
                    "output_tokens": item["output_tokens"],
                    "depth_feature": item["depth_feature"],
                }
                f.write(json.dumps(serializable) + "\n")

    logger.info("shards_written", output_dir=output_dir, n_pairs=len(pairs))


def merge_data_dirs(
    original_dir: str,
    new_dir: str,
    output_dir: str,
    new_ratio: float = 0.1,
) -> int:
    """Merge original training data with self-training data.

    Samples new_ratio fraction from new data, keeps all original data.
    Returns total pair count.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_items: List[Dict] = []
    for shard_path in sorted(Path(original_dir).glob("shard_*.jsonl.gz")):
        with gzip.open(shard_path, "rt") as f:
            for line in f:
                original_items.append(json.loads(line))

    new_items: List[Dict] = []
    for shard_path in sorted(Path(new_dir).glob("shard_*.jsonl.gz")):
        with gzip.open(shard_path, "rt") as f:
            for line in f:
                new_items.append(json.loads(line))

    n_new = max(1, int(len(original_items) * new_ratio / (1 - new_ratio)))
    if n_new > len(new_items):
        n_new = len(new_items)

    rng = random.Random(42)
    sampled_new = rng.sample(new_items, n_new) if n_new < len(new_items) else new_items

    merged = original_items + sampled_new
    rng.shuffle(merged)

    for shard_idx in range(0, len(merged), 10000):
        shard = merged[shard_idx : shard_idx + 10000]
        shard_path = os.path.join(
            output_dir, f"shard_{shard_idx // 10000:04d}.jsonl.gz",
        )
        with gzip.open(shard_path, "wt") as f:
            for item in shard:
                f.write(json.dumps(item) + "\n")

    logger.info("data_merged",
                original=len(original_items), new=len(sampled_new),
                total=len(merged))
    return len(merged)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-training with verification")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Samples per integrand")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--min-log-prob", type=float, default=-50.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--original-train-dir", default=None,
                        help="Original training data dir (for merging)")
    parser.add_argument("--new-ratio", type=float, default=0.1,
                        help="Fraction of new data in merged set")
    args = parser.parse_args()

    print("Generating unlabeled pool...")
    pool = generate_unlabeled_pool(args.pool_size, seed=args.seed)
    print(f"Pool size: {len(pool)}")

    print("Running inference + verification...")
    verified = run_inference_round(
        pool, args.checkpoint,
        n_samples=args.n_samples,
        temperature=args.temperature,
        min_log_prob=args.min_log_prob,
    )
    print(f"Verified: {len(verified)} / {len(pool)} ({100*len(verified)/max(len(pool),1):.1f}%)")

    verified_dir = os.path.join(args.output, "verified")
    write_verified_shards(verified, verified_dir)

    if args.original_train_dir:
        merged_dir = os.path.join(args.output, "merged_train")
        total = merge_data_dirs(
            args.original_train_dir, verified_dir, merged_dir,
            new_ratio=args.new_ratio,
        )
        print(f"Merged dataset: {total} pairs")
