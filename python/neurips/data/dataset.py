"""Pre-computed tensor dataset for efficient training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from neurips.data.features import FEATURE_DIM, extract_features as _py_extract_features
from neurips.data.tokenizer import IntegralTokenizer

# Prefer Rust FFI for feature extraction (10-50x faster).
_HAS_RUST_FEATURES = False
try:
    from neurips._core import extract_features as _rust_extract_features
    from neurips.data.prefix import python_prefix_to_rust as _to_rust
    _HAS_RUST_FEATURES = True
except ImportError:
    pass

_rust_fallback_warned = 0


def _extract_features_fast(
    prefix_str: str, var: str, task: str, param_count: int = 0,
) -> "torch.Tensor":
    global _rust_fallback_warned
    if _HAS_RUST_FEATURES:
        try:
            rust_prefix = _to_rust(prefix_str)
            vec = _rust_extract_features(rust_prefix, var, task, "none", param_count)
            return torch.tensor(vec, dtype=torch.float32)
        except Exception:
            if _rust_fallback_warned < 3:
                import logging
                logging.getLogger(__name__).warning(
                    "Rust feature extraction failed for '%s', falling back to Python",
                    prefix_str[:60],
                )
                _rust_fallback_warned += 1
    feat = _py_extract_features(prefix_str, var, task, param_count=param_count)
    return torch.from_numpy(feat)

# Difficulty tier → integer for index maps.
_DIFFICULTY_IDS: dict[str, int] = {
    "easy": 0, "medium": 1, "hard": 2, "very_hard": 3,
}

# Task name → integer for index maps.
_TASK_IDS: dict[str, int] = {
    "univariate": 0, "multivariate": 1, "parametric": 2,
    "definite": 3, "special_fn": 4,
}

# Map pipeline task names to tokenizer task types.
_TASK_TO_TOKENIZER: dict[str, str] = {
    "univariate": "indef",
    "multivariate": "indef",
    "parametric": "indef",
    "definite": "def",
    "special_fn": "indef",
}


class IntegralDataset(Dataset):
    """Pre-computed tensor dataset with compact uint8 token storage.

    Token IDs are stored as uint8 (vocab_size=256) and cast to long
    on access to keep the full dataset footprint ~8x smaller in RAM.
    """

    def __init__(self, cache_path: Path, mmap: bool = False) -> None:
        load_kw: dict[str, Any] = {"weights_only": True}
        if mmap:
            load_kw["mmap"] = True
        data = torch.load(cache_path, **load_kw)
        self.src_ids: torch.Tensor = data["src_ids"]      # [N, max_src_len] uint8
        self.tgt_ids: torch.Tensor = data["tgt_ids"]      # [N, max_tgt_len] uint8
        self.features: torch.Tensor = data["features"]     # [N, FEATURE_DIM]
        self.task_ids: torch.Tensor = data["task_ids"]     # [N]
        self.diff_ids: torch.Tensor = data["diff_ids"]     # [N]

    def __len__(self) -> int:
        return self.src_ids.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "src_ids": self.src_ids[idx].long(),
            "tgt_ids": self.tgt_ids[idx].long(),
            "features": self.features[idx],
            "task_id": self.task_ids[idx],
            "diff_id": self.diff_ids[idx],
        }


def load_sharded_dataset(
    shard_dir: Path, mmap: bool = False,
) -> Dataset:
    """Load multiple .pt shards and combine via ConcatDataset."""
    from torch.utils.data import ConcatDataset

    shards = sorted(shard_dir.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No .pt shards in {shard_dir}")
    datasets = [IntegralDataset(p, mmap=mmap) for p in shards]
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def precompute_dataset(
    raw_data: list[dict[str, Any]],
    tokenizer: IntegralTokenizer,
    output_path: Path,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
) -> IntegralDataset:
    """Tokenize and cache all examples as contiguous tensors.

    Args:
        raw_data: list of dicts with keys ``integrand``, ``antiderivative``,
            ``task``, ``var``, ``difficulty_tier``, etc.
        tokenizer: encoder for prefix strings.
        output_path: path to write the ``.pt`` cache file.
        max_src_len: maximum source sequence length (padded/truncated).
        max_tgt_len: maximum target sequence length (padded/truncated).

    Returns:
        The loaded :class:`IntegralDataset`.
    """
    n = len(raw_data)
    src_ids = torch.zeros(n, max_src_len, dtype=torch.uint8)
    tgt_ids = torch.zeros(n, max_tgt_len, dtype=torch.uint8)
    features = torch.zeros(n, FEATURE_DIM, dtype=torch.float32)
    task_ids = torch.zeros(n, dtype=torch.long)
    diff_ids = torch.zeros(n, dtype=torch.long)

    for i, ex in enumerate(raw_data):
        task = ex.get("task", "univariate")
        var = ex.get("var", "x")
        integrand = ex.get("integrand", "")
        antideriv = ex.get("antiderivative", "")
        params = ex.get("params", [])
        difficulty = ex.get("difficulty_tier", "medium")

        # Encode source (integrand with metadata).
        tok_task = _TASK_TO_TOKENIZER.get(task, "indef")
        src = tokenizer.encode(integrand, tok_task, var, params=params)
        src_len = min(len(src), max_src_len)
        src_ids[i, :src_len] = torch.tensor(src[:src_len], dtype=torch.uint8)

        # Encode target (antiderivative).
        tgt = tokenizer.encode(antideriv, tok_task, var, params=params)
        tgt_len = min(len(tgt), max_tgt_len)
        tgt_ids[i, :tgt_len] = torch.tensor(tgt[:tgt_len], dtype=torch.uint8)

        # Extract features (use tokenizer-mapped task name for one-hot match).
        features[i] = _extract_features_fast(integrand, var, tok_task, param_count=len(params))

        # Metadata for curriculum filtering.
        task_ids[i] = _TASK_IDS.get(task, 0)
        diff_ids[i] = _DIFFICULTY_IDS.get(difficulty, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "features": features,
            "task_ids": task_ids,
            "diff_ids": diff_ids,
        },
        output_path,
    )
    return IntegralDataset(output_path)
