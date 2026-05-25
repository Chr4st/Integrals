# P4-F007: Competence-Based Adaptive Curriculum

**Status**: draft
**Priority**: P1 (existing static scheduler is suboptimal)

## Problem

Current curriculum scheduler uses static difficulty progression. Research shows adaptive curricula outperform static ones, but AdaRFT (the initially proposed replacement) is validated only on 7B+ models.

For sub-100M models, competence-based curriculum (Platanios et al. 2019) and self-paced learning (Kumar et al. 2010) are the canonical validated approaches. These adjust difficulty based on model competence rather than fixed schedules.

## Solution

### Competence-Based Scheduling

Replace static difficulty progression with a competence function:

```
competence(t) = min(1, sqrt(t * (1 - c_0²) / T + c_0²))
```

Where:
- `t` = current training step
- `T` = total steps for full competence
- `c_0` = initial competence (start with easiest 10% of data)

At each step, sample training examples whose difficulty ≤ competence(t).

### Difficulty Measurement

Use the existing difficulty tier (4 tiers) from `auxiliary.py` plus:
- Expression tree depth (from features.py)
- Number of distinct operations
- Sequence length in prefix tokens

Combine into a single scalar difficulty score normalized to [0, 1].

### Self-Paced Extension

After competence scheduling warm-up (first 30% of training), switch to self-paced:
- Track per-example loss over last 3 epochs
- Examples where loss is consistently low → "mastered" → sample less
- Examples where loss is high but decreasing → "learning zone" → sample more
- Examples where loss is high and static → "too hard" → defer

### Integration with DWA

Keep DWA (Dynamic Weight Averaging) for multi-task loss balancing. The curriculum controls *which* examples are seen; DWA controls *how much* each loss contributes.

## Acceptance Criteria

- [ ] Competence function implementation with configurable c_0 and T
- [ ] Difficulty scorer combining tree depth + op count + seq length + difficulty tier
- [ ] Self-paced tracker: per-example loss history over sliding window
- [ ] Sampling strategy: prioritize "learning zone" examples
- [ ] A/B comparison vs static scheduler on 500K-pair run (measure: val loss convergence speed)
- [ ] ≥1.3x faster convergence to same val loss as static scheduler
- [ ] Graceful degradation: if adaptive is worse, fallback to static automatically
- [ ] Curriculum state saved/restored from checkpoints

## Affected Files

- `python/neurips/training/curriculum.py` — replace/extend with competence-based
- NEW: `python/neurips/training/difficulty.py` — unified difficulty scorer
- `python/neurips/training/train.py` — integrate new scheduler
- `python/neurips/training/checkpoint.py` — save curriculum state
- `configs/default.toml` — add `[training.curriculum]` section
- `tests/python/test_curriculum.py` — update for new scheduler

## Risks

- Competence function may be too conservative (train on easy data too long)
- Self-paced sampling adds overhead: tracking per-example loss requires O(N) storage
- If difficulty scorer correlates poorly with model difficulty, curriculum hurts
- Need 500K-pair A/B test before committing (blocks on F001 data generation)
