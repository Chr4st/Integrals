# Implementation Phases

Execute in order. Each phase depends on the phases before it.

| Phase | File | What it builds | Depends on |
|-------|------|----------------|------------|
| 01 | `phase-01-scaffolding.md` | Rust + Python project structure | — |
| 02 | `phase-02-expression-tree.md` | ExprNode data structure (Rust) | 01 |
| 03 | `phase-03-tree-operations.md` | Differentiation, skeleton, comparison | 02 |
| 04a | `phase-04a-data-gen-core.md` | Random tree generator + batch pipeline | 02, 03 |
| 04b | `phase-04b-data-gen-modes.md` | 5 task-specific generation modes | 04a |
| 05 | `phase-05-cas-verification.md` | SymPy validation + filtering | 04b |
| 06 | `phase-06-data-pipeline.md` | Skeleton split, dedup, 80/20 train/test | 03, 05 |
| 07a | `phase-07a-tokenizer.md` | Token vocabulary (256 tokens) | 02 |
| 07b | `phase-07b-features.md` | 688-dim structural feature extractor | 02 |
| 08 | `phase-08-seq-transformer.md` | Sequence transformer baseline (95M) | 07a, 07b |
| 09a | `phase-09a-grammar-mask.md` | Arity-stack grammar constraints | 07a |
| 09b | `phase-09b-sampling.md` | Temperature sampling (N=25) + verify | 08, 09a |
| 10a | `phase-10a-node-embedding.md` | Tree node embeddings (256-dim) | 07a |
| 10b | `phase-10b-message-passing.md` | GNN message passing (8 rounds) | 10a |
| 11a | `phase-11a-tree-decoder.md` | Top-down tree decoder | 10b |
| 11b | `phase-11b-variable-attention.md` | Variable-aware attention + full assembly | 10b, 11a |
| 12a | `phase-12a-training-loop.md` | Training loop, optimizer, checkpointing | 06, 09b, 11b |
| 12b | `phase-12b-curriculum.md` | Curriculum learning scheduler | 12a |
| 13a | `phase-13a-oracle-core.md` | Verification oracle (indefinite + definite) | 02 |
| 13b | `phase-13b-oracle-advanced.md` | Oracle: multivariate + parametric | 13a |
| 14a | `phase-14a-evaluation.md` | Benchmarking + comparison table | 12b, 13b |
| 14b | `phase-14b-ablations.md` | Ablation studies + error analysis | 14a |
| 15 | `phase-15-paper.md` | NeurIPS 2026 paper | 14b |

## Parallel Execution Opportunities

These phase groups can run in parallel (via multiple agents):

```
Group A (data):     01 → 02 → 03 → 04a → 04b → 05 → 06
Group B (seq):      07a + 07b → 08 → 09a → 09b     (after 02)
Group C (tree):     10a → 10b → 11a → 11b           (after 07a)
Group D (oracle):   13a → 13b                        (after 02)

Merge:              12a → 12b  (needs A + B + C)
Final:              14a → 14b → 15  (needs everything)
```

## Architecture Summary

| | Sequence Transformer | Tree GNN |
|---|---|---|
| Input format | Flat prefix sequence | Native tree |
| Encoder | 10-layer transformer | 8-round message passing |
| Decoder | Left-to-right tokens | Top-down tree levels |
| Parameters | ~95M | ~9M |
| Novel? | No (Lample & Charton style) | Yes (first tree-native) |
