# Phase 6: Data Pipeline — Split, Dedup, Save

## Goal
Deduplicate verified pairs, split 80/20 by skeleton, stratify by difficulty
and task type. Produce final train.jsonl and test.jsonl ready for training.

## Why Skeleton-Based Splitting
Random split leaks information. `∫ 3x² dx` and `∫ 7x² dx` are the same
integral with different constants. If one is in train and the other in test,
the model gets free answers. Skeleton splitting puts ALL constant-variants
of the same structure in the same split.

## File: `python/neurips/data/split.py`

### Step 1: Exact Deduplication
```python
def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicate integrands (same prefix string)."""
```
Use a set of integrand prefix strings. Keep the first occurrence, drop the rest.
Expected: 1.275M → ~1.15M (remove ~10% exact duplicates from random generation).

### Step 2: Skeleton Grouping
```python
def group_by_skeleton(pairs: list[dict]) -> dict[str, list[dict]]:
    """Group pairs by skeleton. Uses Rust skeleton() via PyO3."""
```
Call the Rust `skeleton()` function from Phase 3 on each integrand.
Group all pairs with the same skeleton string.
Expected: ~1.15M pairs → ~360K unique skeletons (~3.2 variants per skeleton).

### Step 3: Difficulty Tier Assignment
```python
def assign_tier(pair: dict) -> str:
    """Assign difficulty tier based on integrand complexity."""
    depth = pair["integrand_depth"]
    nodes = pair["integrand_nodes"]
    if depth <= 3 and nodes <= 5:
        return "easy"       # ~30% of data
    elif depth <= 6 and nodes <= 15:
        return "medium"     # ~40% of data
    elif depth <= 10 and nodes <= 30:
        return "hard"       # ~25% of data
    else:
        return "very_hard"  # ~5% of data
```

### Step 4: Stratified Skeleton Split
```python
def stratified_skeleton_split(
    pairs: list[dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:
```
Algorithm:
1. Group pairs by (task_type, difficulty_tier) → cells
2. Within each cell, group by skeleton
3. Shuffle skeletons within each cell (using seed for reproducibility)
4. Walk through skeletons, assign to train until cell hits 80%
5. Remaining skeletons go to test
6. Collect all train pairs and all test pairs across cells

This guarantees:
- NO skeleton appears in both train and test (zero leakage)
- Each (task, difficulty) cell has 80/20 ratio
- Reproducible with seed

### Step 5: Save Final Datasets
```python
def save_splits(train: list[dict], test: list[dict], output_dir: str):
```
Write to:
- `data/final/train.jsonl` (one JSON per line, shuffled)
- `data/final/test.jsonl` (one JSON per line, shuffled)
- `data/final/stats.json` (counts by task type, tier, split)

## File: `python/neurips/data/stats.py`

Print a summary table after splitting:
```
                  Train        Test         Total
                  -----        ----         -----
Univ. easy:       108K (80%)   27K (20%)    135K
Univ. medium:     144K (80%)   36K (20%)    180K
Univ. hard:        90K (80%)   23K (20%)    113K
Univ. v.hard:      18K (80%)    5K (20%)     23K
Multi. easy:       61K (80%)   15K (20%)     76K
...
Special fn hard:   38K (80%)   10K (20%)     48K
                  -----        ----         -----
TOTAL:            920K (80%)   230K (20%)   1.15M
```

## Verification
- `len(train) + len(test) == total` (no pairs lost)
- `set(train_skeletons) & set(test_skeletons) == empty` (zero overlap)
- Per-cell ratio is within 79-81% (rounding)
- `stats.json` matches printed table
- Reload test: `load(save(train)) == train` (serialization round-trip)
