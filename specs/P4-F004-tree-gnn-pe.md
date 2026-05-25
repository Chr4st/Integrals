# P4-F004: Tree GNN Positional Encoding

**Status**: draft
**Priority**: P1 (validated gap — GNN lacks position signal)

## Problem

Tree GNN (256-dim, 8 message-passing rounds) has no positional encoding. Node embeddings encode symbol type and structural features but not position within the expression tree. Graph transformer literature shows Laplacian PE and random-walk encoding significantly improve GNN expressiveness on structured graphs.

Expression trees are well-ordered (left/right children matter for non-commutative ops like subtraction, division). Position information helps distinguish `a-b` from `b-a`.

## Solution

Implement 3 tree-aware PE variants and ablate:

### 1. Depth + Child-Index Encoding (lightweight)
- Encode tree depth (0=root) as sinusoidal embedding
- Encode child index (0=left, 1=right) as learnable embedding
- Concatenate with node features before first message-passing round

### 2. Random-Walk Structural Encoding (RWSE)
- Compute k-step random walk probabilities from each node
- Use landing probabilities as k-dimensional feature vector
- k=8 (matching message-passing rounds)

### 3. Laplacian PE
- Compute normalized graph Laplacian of expression tree
- Use first k eigenvectors as positional features (k=8)
- Handle sign ambiguity via SignNet or absolute values

### Ablation Protocol
- Same as F003: 10 epochs on 100K subset
- Measure tree GNN validation loss and exact-match accuracy
- Compare: no PE (current), depth+index, RWSE, Laplacian PE

## Acceptance Criteria

- [ ] PE factory: `build_tree_pe(config) -> TreePositionalEncoding`
- [ ] Depth+child-index encoding (zero extra compute at inference)
- [ ] RWSE implementation with configurable k
- [ ] Laplacian PE with sign invariance handling
- [ ] Ablation script comparing all 4 variants on 100K subset
- [ ] Winner integrated into tree GNN forward pass
- [ ] No regression on existing test suite

## Affected Files

- `python/neurips/models/tree_gnn.py` — inject PE into NodeEmbedding
- NEW: `python/neurips/models/tree_positional.py` — PE implementations
- `configs/default.toml` — add `[model.tree_gnn.pe]` section
- NEW: `scripts/ablate_tree_pe.py` — ablation runner
- `tests/python/test_tree_gnn_correctness.py` — test PE variants

## Risks

- Laplacian PE is expensive to compute for each batch (eigendecomposition) — cache per-expression
- Expression trees may be too simple for PE to help (depth is already encoded via message-passing rounds)
- Depth+child-index may suffice since expression trees are ordered unlike general graphs
