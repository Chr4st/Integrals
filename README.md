# Neural Symbolic Integration

Tree-native neural architecture for learning indefinite and definite integration from synthetic data. A 12.1M-parameter Tree GNN encoder-decoder, verified against a Rust symbolic algebra backend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Rust Core (neurips_core)                                           │
│  ├── gen.rs ─────────── Random expression tree generation            │
│  ├── gen_coverage.rs ── Coverage-guaranteed skeleton enumeration      │
│  ├── diff.rs ────────── Symbolic differentiation (chain/product/     │
│  │                       quotient rules, smart constructors)         │
│  ├── verify.rs ──────── CAS verification (symbolic + numerical)      │
│  ├── canonical.rs ───── E-graph equality saturation (egg crate)      │
│  ├── features.rs ────── 344-dim structural feature extraction        │
│  └── env.rs ─────────── 4-action integration environment            │
│                                                                      │
│         ↓ PyO3 FFI (neurips._core)                                  │
│                                                                      │
│  Python Data Pipeline                                                │
│  ├── generate_data.py ── Batch generation + SymPy fallback verify    │
│  ├── split.py ────────── Skeleton-stratified split (zero leakage)    │
│  ├── tokenizer.py ────── Base-100 number encoding, vocab=256         │
│  └── dataset.py ──────── Precomputed uint8 tensor cache (8× RAM↓)   │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        Model Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Tree GNN Encoder-Decoder (12.1M params)                 │       │
│  │                                                          │       │
│  │  Encoder (~5.6M):                                        │       │
│  │   NodeEmbedding:                                         │       │
│  │    symbol(256→64) + role(12→64) + struct(128→128) → 256  │       │
│  │   Tree PE (configurable):                                │       │
│  │    depth_index / rwse / laplacian                         │       │
│  │   8× Bidirectional MessageRound                          │       │
│  │   VariableAwareAttention (8 heads)                       │       │
│  │                                                          │       │
│  │  Decoder (~6.5M):                                        │       │
│  │   Top-down autoregressive (BFS, 8 levels)                │       │
│  │   8× CrossAttention layers (256-dim, 8 heads)            │       │
│  │   SwiGLU FFN (d_hidden=682)                              │       │
│  │   Symbol head → vocab (256)                              │       │
│  └──────────────────────────┬───────────────────────────────┘       │
│                             ▼                                        │
│               Grammar-Constrained Decoding                           │
│               (PrefixGrammarMask, arity-based)                       │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     Inference / Search                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Action Policy (4 actions) ── substitute, IBP, partial_frac, close   │
│  PUCT-MCTS ── neural value network, LRU verification cache           │
│  SubstitutionParamHead ── pointer attention over tree nodes           │
│  IBPParamHead ── dual pointer (u, dv) factorization                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        Training                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Loss: equivalence_class_ce (min-over-K equivalent targets)          │
│  Optimizer: AdamW (lr=3e-4, wd=0.01)                                │
│  Schedule: linear warmup (5 ep) → cosine annealing → SWA (75%+)     │
│  AMP: BF16 on Ampere+, FP16 fallback                                │
│  Curriculum: static 4-phase or competence-based adaptive             │
│  Self-play: REINFORCE + MCTS-guided trajectory generation            │
│  Auxiliary: difficulty classification + depth regression (0.1 wt)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Technical Details

### Data Pipeline (Rust → Python)

**Expression Generation** (`gen.rs` + `gen_coverage.rs`):
- 5 generation modes: univariate, multivariate, definite, parametric, special_fn
- Coverage-guaranteed skeleton enumeration: 200+ skeletons across 15+ structural families
- 70% skeleton-based (uniform across families) + 30% random exploration
- Rayon-parallel differentiation for batch pair generation
- Smart constructors simplify during generation (0-multiplication, 1-power, double-neg)

**Differentiation** (`diff.rs`, 399 lines):
- Fused differentiation + simplification
- Handles all 25 unary operators and 7 binary operators
- Chain rule, product rule, quotient rule, power rule with special cases

**Verification** (`verify.rs`, 412 lines):
- Symbolic: differentiate candidate, compare residual to zero via canonicalization
- Numerical: 20 random points in [0.1, 5.0], tolerance 1e-6
- Batch: parallelized via Rayon (1000× faster than SymPy)

**Canonicalization** (`canonical.rs`, 175 lines):
- E-graph equality saturation using the `egg` crate
- 20 algebraic rewrite rules (commutativity, associativity, identity, inverse, distributivity, log/exp, trig)
- `canonicalize()`: extract smallest-AST equivalent form (node limit: 1000)
- `canonical_hash()`: deduplication via deterministic hashing of canonical form
- `diverse_equivalents()`: extract K diverse forms for data augmentation

**Feature Extraction** (`features.rs`, 318 lines):
- 344-dimensional structural feature vector per expression
- 264-dim: function-type × depth-bin histogram (33 types × 8 bins)
- 16-dim: variable role features (presence, integration target, depth stats)
- 40-dim: signature classification (5 types × 8 bins: gaussian, oscillatory, rational, algebraic singularity, exponential growth)
- 24-dim: task metadata + complexity scalars

### Tree GNN (12.1M params)

| Component | Dimensions |
|-----------|-----------|
| node_dim | 256 |
| message_rounds | 8 |
| decoder_levels | 8 |
| attention_heads | 8 |

**Encoder (~5.6M params)**:

*NodeEmbedding*: heterogeneous features → shared 256-dim space
- Symbol embedding (256 tokens → 64-dim)
- Role MLP (12 features → 64-dim): root/leaf flags, child index, arity, depth parity
- Structural MLP (128 features → 128-dim): depth, subtree size, sibling count
- Concatenate → 256-dim

*Message Passing*: 8 rounds of bidirectional parent↔child propagation, scatter_mean aggregation, residual + LayerNorm per round. MLP update: [h‖m_p‖m_c] (768→512→256).

*Variable-Aware Attention*: multi-head attention (8 heads) with learned dependency bias. Bottom-up boolean mask identifies nodes depending on integration variable; pairwise bias B_ij = β·1[d_i = d_j] added to attention logits.

**Decoder (~6.5M params)**:

*Top-down autoregressive tree decoder*: BFS frontier expansion up to 8 levels. Each level: 8× cross-attention layers (pre-norm, 256-dim, 8 heads) with SwiGLU FFN (d_hidden=682). Symbol head predicts node operator/operand. Child initialization generates seed embeddings for next level.

**Tree Positional Encoding** (pluggable):
- `depth_index`: sinusoidal depth + learnable child-index (root/left/right)
- `rwse`: k-step random walk landing probabilities → linear projection
- `laplacian`: first-k eigenvectors of normalized graph Laplacian

### Action-Space Policy

4-action vocabulary for step-by-step integration:
| Action | ID | Parameters |
|--------|---:|-----------|
| substitute(u=g(x)) | 0 | SubstitutionParamHead: pointer over tree nodes |
| integrate_by_parts(u, dv) | 1 | IBPParamHead: dual pointer (u-factor, dv-factor) |
| partial_fractions | 2 | None |
| close(F) | 3 | None |

**ValueHead**: MLP → Tanh, estimates solve probability in [-1, 1]

**MCTS** (100 simulations default):
- PUCT selection: Q + c·P·√N_parent/(1+N_child), c=1.4
- Neural value estimation (no random rollout)
- LRU verification cache (100K entries) for env step deduplication
- Dirichlet noise at root for exploration (α=0.3, frac=0.25)

### Training Pipeline

**Curriculum** (4-phase static schedule, 90 epochs):
1. Epochs 1–10: univariate only, easy/medium tiers
2. Epochs 11–30: +multivariate, up to hard
3. Epochs 31–60: all 5 task types, all difficulty tiers
4. Epochs 61–90: uniform sampling

**Competence-based adaptive** (optional):
- competence(t) = min(1, √(t·(1−c₀²)/T + c₀²)), c₀=0.1
- Self-paced mode after 30% warmup: up-weight "learning zone" examples (loss trending down), down-weight mastered/stalled

**Equivalence-Class CE Loss**:
- Given K equivalent target antiderivatives per example
- Loss = min_{k=1..K} CE(logits, target_k)
- Encourages learning any correct equivalent form

**Optimization**:
- AdamW: lr=3e-4, weight_decay=0.01
- Warmup: 5 epochs linear (0.01 → 1.0)
- Cosine annealing to η_min=1e-6
- SWA: activated at 75% of training, lr=1e-5
- Gradient clipping: max_norm=1.0
- AMP: BF16 on Ampere+, auto-fallback to FP16
- torch.compile(mode="max-autotune") on CUDA

### Inference

**Grammar Masking**: PrefixGrammarMask enforces valid prefix-notation at every decoding step (O(1) per token via incremental arity stack).

**Verification**: Rust CAS (symbolic diff + numerical check) validates each candidate. First verified solution wins.

## Prior Work

This architecture is compared against Lample & Charton (2019), who trained a 95M-parameter encoder-decoder seq2seq transformer on 40M pairs with 32 V100 GPUs. Our tree-native approach achieves comparable results with ~8× fewer parameters by operating directly on expression tree structure rather than serialized token sequences.

## Quickstart

```bash
# Build Rust core (requires Rust 1.70+)
cd rust/core && cargo build --release

# Install Python package (editable, with Rust extension)
pip install -e ".[dev]"

# Generate training data (1.5M pairs)
python scripts/generate_data.py --config configs/default.toml --output data/

# Generate coverage-guaranteed data (Rust FFI or Python fallback)
python scripts/generate_covered.py --total 1500000 --output data/covered/

# Train tree GNN model
python scripts/train.py --config configs/default.toml --data-dir data/

# Run tree PE ablation
python scripts/ablate_tree_pe.py --output-dir results/tree_pe_ablation

# Evaluate
python scripts/evaluate.py --model tree --checkpoint checkpoints/tree_best.pt
```

## Configuration

All hyperparameters in `configs/default.toml`. Key knobs:

```toml
[model.tree_gnn]
pe_type = "depth_index"   # Enable tree positional encoding

[training.curriculum]
type = "competence"       # Switch to adaptive curriculum

[inference.mcts]
n_simulations = 200       # More search budget
```

## Project Structure

```
├── rust/core/src/         Rust symbolic algebra backend (4K lines)
│   ├── expr/              Expression tree definition + parsing
│   ├── gen.rs             Random tree generation (5 modes)
│   ├── gen_coverage.rs    Coverage-guaranteed skeleton generation
│   ├── diff.rs            Symbolic differentiation
│   ├── verify.rs          CAS verification (symbolic + numerical)
│   ├── canonical.rs       E-graph equality saturation
│   ├── features.rs        344-dim feature extraction
│   ├── env.rs             4-action integration environment
│   └── actions.rs         Action-space operations
├── python/neurips/
│   ├── data/              Tokenizer, features, verification, splitting
│   ├── models/            Tree GNN, tree decoder, policy heads
│   ├── training/          Curriculum, loss, trainer, self-play
│   ├── inference/         MCTS, action search
│   └── evaluation/        Benchmarking, oracle, analysis
├── scripts/               Training, evaluation, ablation runners
├── configs/               TOML configuration
├── specs/                 Feature specifications
└── tests/python/          Unit + integration tests
```

## References

- Lample & Charton (2019). Deep Learning for Symbolic Mathematics. arXiv:1912.01412
- Platanios et al. (2019). Competence-based Curriculum Learning. NAACL 2019
- Barket et al. (2025). Tree-Based Deep Learning for Ranking Integration Algorithms. arXiv:2508.06383
- EGG-SR (2025). Embedding Symbolic Equivalence into Symbolic Regression. arXiv:2511.05849
- AlphaIntegrator (2024). Transformer Action Search for Symbolic Integration Proofs. arXiv:2410.02666
- BFS-Prover (2025). Scalable Best-First Tree Search for LLM Provers. ACL 2025
- CRANE (2025). Reasoning with Constrained LLM Generation. ICML 2025
