# P4-F003: Positional Encoding Ablation Framework

**Status**: draft
**Priority**: P1 (high confidence gap, needs ablation before committing)

## Problem

Seq transformer uses fixed sinusoidal PE (Vaswani 2017). Modern transformers universally use RoPE. However, all RoPE evidence is from NLP — prefix-notation symbolic math has different token distribution and sequence structure. ALiBi (zero learned params, better extrapolation) is a plausible competitor. Must ablate before committing.

## Solution

Implement a PE abstraction layer and 4 PE variants:

1. **Sinusoidal** (current baseline) — fixed, added to embeddings
2. **RoPE** — rotary encoding applied in attention, relative position via rotation matrices
3. **ALiBi** — linear attention bias, no embedding modification, zero params
4. **NoPE** — no positional encoding (control: does position even matter for prefix notation?)

### Ablation Protocol

- Train each variant for 10 epochs on a fixed 100K-pair subset
- Measure: validation loss, exact-match accuracy (top-1 and top-25 sampling), and length generalization (train on len≤256, eval on len 257-512)
- Report results in a comparison table
- Winner becomes default PE for full training

## Acceptance Criteria

- [ ] PE factory function: `build_pe(config) -> PositionalEncoding`
- [ ] RoPE implementation with configurable base frequency (default 10000)
- [ ] ALiBi implementation with per-head slopes
- [ ] NoPE variant (just dropout, no position signal)
- [ ] Ablation script that trains all 4 variants and produces comparison CSV
- [ ] Length generalization test: train on ≤256, eval on 257-512
- [ ] Winner selected based on val loss + length generalization

## Affected Files

- `python/neurips/models/seq_transformer.py` — refactor PE into pluggable module
- NEW: `python/neurips/models/positional.py` — PE implementations (RoPE, ALiBi, NoPE, Sinusoidal)
- `configs/default.toml` — add `[model.seq_transformer.pe]` section
- NEW: `scripts/ablate_pe.py` — ablation runner script
- `tests/python/test_models.py` — test each PE variant forward pass

## Risks

- NoPE might win (prefix notation encodes structure in token order) — this would be surprising but valid
- RoPE base frequency may need tuning for math vocab (smaller than NLP vocab)
- Ablation takes ~4 GPU-hours per variant (16 hours total)
