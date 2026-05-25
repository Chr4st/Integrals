"""Shared utilities for CLI scripts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    import tomli
except ImportError:
    import tomllib as tomli  # type: ignore[no-redef]


def detect_device(requested: str | None = None) -> torch.device:
    """Auto-detect or use the requested compute device."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> dict:
    """Load a TOML config file."""
    with open(path, "rb") as f:
        return tomli.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
