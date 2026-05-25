# Phase 15: NeurIPS Paper

## Goal
Write the NeurIPS 2026 submission. 8 pages + references + appendix.
The paper's story: "expressions are trees, so we built a tree-native model
for integration, and compared it fairly against the standard sequence approach."

## File: `paper/main.tex`

Use NeurIPS 2026 LaTeX template. Structure:

### Title
"Tree-Native Neural Integration: Graph Message Passing vs Sequence
Translation for Symbolic Antidifferentiation"

### Abstract (150 words)
- Problem: neural symbolic integration treats expressions as flat sequences
- Insight: expressions are trees; tree-native processing is more natural
- Contribution: first GNN-based integration model with top-down tree decoder
- Results: [X]% solve rate with 10x fewer parameters than sequence baseline
- Extra: first model handling multivariate symbolic integration

### 1. Introduction (1 page)
- Integration is the inverse of differentiation
- Lample & Charton showed transformers can learn it
- But all existing work flattens trees to sequences (wasteful)
- We propose tree-native processing: GNN encoder + top-down decoder
- Contributions:
  1. First tree-native architecture for symbolic integration
  2. First neural model for multivariate symbolic integration
  3. Fair comparison: tree GNN (9M params) vs sequence transformer (95M params)
  4. Skeleton-based data splitting (no leakage)

### 2. Related Work (0.5 page)
- Lample & Charton (2020): seq2seq integration
- AlphaIntegrator (2024): search-based
- SIRD (2023): rule prediction
- Tree transformers in NLP (Shiv & Quirk 2019): positional encoding for trees
- GNNs for symbolic math: none for integration (our gap)

### 3. Method (3 pages)
- 3.1 Task formulation (input tree → output tree, verified by differentiation)
- 3.2 Tree GNN encoder (message passing, variable-aware attention)
- 3.3 Top-down tree decoder (level-by-level generation)
- 3.4 Sequence transformer baseline (standard enc-dec, grammar mask)
- 3.5 Verification oracle (differentiation, sampling N=25)
- 3.6 Data generation (backward generation, 5 task types, skeleton split)

### 4. Experiments (2 pages)
- 4.1 Setup (dataset size, split, training details, hardware)
- 4.2 Main results (comparison table: tree GNN vs seq transformer)
- 4.3 Multivariate results (the novel contribution)
- 4.4 Ablation studies (message rounds, variable attention, curriculum, N samples)
- 4.5 Error analysis (what does each model get wrong?)

### 5. Discussion (0.5 page)
- Tree-native processing: when is it worth the engineering complexity?
- Parameter efficiency: 9M vs 95M, implications for deployment
- Multivariate gap: why no one did this before, what it enables
- Limitations: expression tree size limits, CAS dependency for verification

### 6. Conclusion (0.5 page)
- Summary of contributions
- Future work: higher dimensions, definite multivariate, PDE integration

## Figures
1. **Architecture comparison** (full page): side-by-side diagram showing
   sequence pipeline vs tree pipeline for the same integral
2. **Message passing visualization**: 4-panel showing how node embeddings
   evolve over rounds 1, 3, 5, 8
3. **Results table**: the main comparison table from Phase 14
4. **Ablation plots**: line charts for message rounds, N samples, etc.
5. **Error analysis**: bar chart of failure types per model

## Appendix
- A: Full vocabulary table
- B: Hyperparameter details
- C: Additional ablation results
- D: Example predictions (10 integrals, both models' outputs)
- E: Data generation details and statistics

## Verification
- Paper compiles with `pdflatex main.tex`
- Within 8-page limit (excluding references and appendix)
- All figures and tables referenced in text
- All claims supported by experimental results
- NeurIPS formatting guidelines met
