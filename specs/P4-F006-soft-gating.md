# P4-F006: Soft Gating Between Seq Transformer and Tree GNN

**Status**: draft
**Priority**: P2 (depends on both models being individually strong)

## Problem

Seq transformer (95M) and tree GNN (12M) produce independent predictions. Currently no mechanism to combine their strengths per-integrand. Research (Barket 2025) shows tree representations beat sequence-based for some CAS tasks, but sequence models handle long expressions better.

Hard MoE with 2 experts risks routing collapse. Soft gating avoids this — both experts always contribute, with learned weights.

## Solution

### Soft Gating Network

1. **Feature extraction**: concatenate seq transformer encoder CLS token (640-dim) and tree GNN pooled embedding (256-dim) → 896-dim
2. **Gating MLP**: 896 → 128 → 2 (softmax) → produces weights [w_seq, w_tree]
3. **Weighted combination**: final logits = w_seq * seq_logits + w_tree * tree_logits
4. Both models share the decoder vocabulary and output space

### Training Strategy

- Train both models independently first (existing pipeline)
- Freeze both encoders, train gating network for 5 epochs on validation performance
- Optionally: fine-tune all three jointly with small LR (1e-5) for 3 epochs

### Alternative: Learned Ensemble (simpler)

If soft gating adds too much complexity:
- Per-difficulty-tier fixed weights learned on validation set
- No gating network, just a 4-element weight vector (one per difficulty tier)

## Acceptance Criteria

- [ ] Gating network produces per-example weights for seq and tree predictions
- [ ] Soft-gated model accuracy ≥ max(seq_alone, tree_alone) on validation set
- [ ] Gating weights interpretable: correlate with expression properties (depth, length, operator mix)
- [ ] No routing collapse: both experts receive weight > 0.1 on ≥80% of examples
- [ ] Training pipeline: independent → freeze → gate → optional joint fine-tune
- [ ] Ablation: soft gating vs fixed-weight ensemble vs oracle (always pick better model)

## Affected Files

- NEW: `python/neurips/models/gating.py` — soft gating network
- `python/neurips/training/train.py` — add gated training phase
- `scripts/train.py` — add `--gated` mode
- `python/neurips/evaluation/benchmark.py` — evaluate gated model
- `configs/default.toml` — add `[model.gating]` section

## Risks

- If one model dominates across all integrands, gating adds complexity for zero gain
- Gating network has access to encoder features → may learn to ignore one model entirely
- Joint fine-tuning may destabilize pre-trained encoders → use very small LR
