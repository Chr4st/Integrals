# P4-F001: Data Scaling + Pipeline Hardening

**Status**: draft
**Priority**: P0 (highest — scaling laws show this dominates all architectural improvements)

## Problem

At 95M params, optimal token-to-param ratio is ~15 (Otte 2025). Current dataset: 1.5M pairs. Required: ~100M pairs. The model is 66x under-provisioned on data. Scaling laws for symbolic regression show steeper exponents (beta~0.21) than language modeling — compute converts to performance faster.

Additionally, the data pipeline lacks canonicalization. Equivalent expressions under commutativity/associativity produce redundant training signal, estimated 30-60% duplication at scale.

## Solution

### Phase A: Canonicalization (Rust)

Integrate the `egg` crate (MIT, Rust-native e-graph library) into `rust/core/src/gen.rs`:

1. Define rewrite rules for: commutativity (`a+b = b+a`), associativity, distributivity, trig identities (`sin²+cos²=1`), log/exp inverses
2. After generating an expression, run equality saturation to find canonical form
3. Hash canonical form for deduplication
4. Cap e-graph expansion to 1000 nodes per expression (prevent combinatorial explosion)

### Phase B: Equivalence-Class Augmentation

For each canonical expression, extract K=8 diverse equivalent forms from the e-graph:
- Select forms maximizing edit distance from each other
- Feed as targets for the existing equivalence-class CE loss (min-over-K)
- This replaces the current SymPy-based equivalence generation

### Phase C: Scale Generation to 15M

1. Parallelize Rust generator across all CPU cores (rayon crate)
2. Batch SymPy verification calls (PyO3 → multiprocessing pool)
3. Store in Apache Parquet via `arrow2` for efficient random-access during training
4. Target: 15M pairs as intermediate milestone, then 50M, then 100M

## Acceptance Criteria

- [ ] `egg` crate integrated with ≥10 rewrite rules covering trig, log, algebraic identities
- [ ] Canonicalization reduces dataset size by ≥20% (measured dedup rate on 1.5M set)
- [ ] Equivalence augmentation produces K=8 diverse forms per expression
- [ ] E-graph expansion capped at 1000 nodes; generation time <100ms/expression median
- [ ] Parquet output with train/val/test splits, random-access compatible
- [ ] Pipeline generates 15M verified pairs within 24 hours on available hardware
- [ ] Property test: canonical(a+b) == canonical(b+a) for all generated expressions

## Affected Files

- `rust/core/Cargo.toml` — add `egg` dependency
- `rust/core/src/gen.rs` — canonicalization + augmentation integration
- `rust/core/src/lib.rs` — export new module
- NEW: `rust/core/src/canonical.rs` — rewrite rules + e-graph runner
- NEW: `rust/core/src/parquet.rs` — arrow2 Parquet writer
- `configs/default.toml` — add `[data.scaling]` section

## Risks

- E-graph combinatorial explosion on complex expressions → mitigated by 1000-node cap
- Trig identity rewrite rules may not cover all integration-relevant equivalences → prototype on 100 expressions first
- 24hr generation target depends on hardware; may need cloud burst
