# Neural Symbolic Integration

Dual-architecture neural system for learning indefinite and definite integration from synthetic data. Combines a Sequence Transformer (95M params) with a Tree GNN (12M params), verified against a Rust symbolic algebra backend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Rust Core (neurips_core)                                           │
│  ├── gen.rs ─────────── Random expression tree generation            │
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
│  ┌─────────────────────┐         ┌──────────────────────┐          │
│  │  Seq Transformer    │         │  Tree GNN            │          │
│  │  (~95M params)      │         │  (~12M params)       │          │
│  │                     │         │                      │          │
│  │  Encoder:           │         │  NodeEmbedding:      │          │
│  │   token_emb(256→640)│         │   symbol(256→64)     │          │
│  │   PE (configurable) │         │   role(12→64)        │          │
│  │   feat_proj(344→640)│         │   struct(128→128)    │          │
│  │   10× TransformerEnc│         │   ────────────→256   │          │
│  │                     │         │  Tree PE (config):   │          │
│  │  Decoder:           │         │   depth_index/rwse/  │          │
│  │   10× TransformerDec│         │   laplacian          │          │
│  │   output_proj(→256) │         │  8× MessageRound     │          │
│  │   weight-tied (3×)  │         │  VariableAwareAttn   │          │
│  │                     │         │  TreeDecoder (8-lvl)  │          │
│  └─────────┬───────────┘         └──────────┬───────────┘          │
│            │                                 │                       │
│            └──────────┐    ┌─────────────────┘                       │
│                       ▼    ▼                                         │
│               ┌──────────────────┐                                   │
│               │  Soft Gating     │                                   │
│               │  (optional)      │                                   │
│               │  [896→128→2]     │                                   │
│               │  softmax weights │                                   │
│               └────────┬─────────┘                                   │
│                        ▼                                             │
│               Grammar-Constrained Decoding                           │
│               (PrefixGrammarMask, arity-based)                       │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     Inference / Search                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Diverse Beam Search ── n_groups=2, diversity penalty, grammar mask   │
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
│  Distillation: Born-Again (α·KL + (1-α)·CE, T=3)                   │
│  PCGrad: per-task gradient projection (optional)                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Technical Details

### Data Pipeline (Rust → Python)

**Expression Generation** (`gen.rs`, 570 lines):
- 5 generation modes: univariate, multivariate, definite, parametric, special_fn
- Recursive tree construction with configurable depth/node budgets
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

### Sequence Transformer (95M params)

| Component | Dimensions |
|-----------|-----------|
| d_model | 640 |
| n_heads | 10 (head_dim=64) |
| n_layers | 10 encoder + 10 decoder |
| d_ff | 2560 (4× expansion) |
| max_seq_len | 512 |
| vocab_size | 256 |

**Positional encoding** (pluggable via config):
- `sinusoidal`: fixed Vaswani et al. (current default)
- `rope`: Rotary Position Embedding (Su et al.) — applied inside attention
- `alibi`: Attention with Linear Biases (Press et al.) — zero learned params
- `nope`: no encoding (control variant)

**Weight tying** (3-way): encoder.token_emb = decoder.token_emb, decoder.output_proj.weight = decoder.token_emb.weight

**FlashAttention-2**: enabled via PyTorch 2.x SDPA backend (auto-detected at import time). Falls back to memory-efficient attention on older GPUs.

### Tree GNN (12M params)

| Component | Dimensions |
|-----------|-----------|
| node_dim | 256 |
| message_rounds | 8 |
| decoder_levels | 8 |
| attention_heads | 8 |

**NodeEmbedding**: heterogeneous features → shared 256-dim space
- Symbol embedding (256 tokens → 64-dim)
- Role MLP (12 features → 64-dim)
- Structural MLP (128 features → 128-dim)
- Concatenate → 256-dim

**Message Passing**: bidirectional parent↔child, scatter_mean aggregation, 8 rounds

**Variable-Aware Attention**: multihead attention with learned dependency bias for nodes sharing integration variables

**Tree Decoder**: top-down autoregressive, BFS frontier expansion up to 8 levels, cross-attention to encoder output

**Tree Positional Encoding** (pluggable):
- `depth_index`: sinusoidal depth + learnable child-index (root/left/right)
- `rwse`: k-step random walk landing probabilities → linear projection
- `laplacian`: first-k eigenvectors of normalized graph Laplacian

### Action-Space Policy (Sprint 3)

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
- Gradient accumulation: configurable steps
- Label smoothing: 0.1
- AMP: BF16 on Ampere+, auto-fallback to FP16
- torch.compile(mode="max-autotune") on CUDA

### Inference

**Diverse Beam Search**: 2 diversity groups, penalty=0.5, grammar-constrained. Generates N=25 candidates per example.

**Grammar Masking**: PrefixGrammarMask enforces valid prefix-notation at every decoding step (O(1) per token via incremental arity stack).

**Verification**: Rust CAS (symbolic diff + numerical check) validates each candidate. First verified solution wins.

## Efficiency Analysis & Research Assessment

Each architectural choice evaluated against 2024-2026 literature:

### Optimal Choices (keep as-is)

| Component | Evidence |
|-----------|----------|
| **Encoder-decoder architecture** | Ewer et al. (2024) prove enc-only models reach perfect accuracy on structured symbolic tasks where decoder-only fails. Lample & Charton baseline remains unbeaten for end-to-end integration. |
| **E-graph canonicalization** | EGG-SR (ICLR 2026) validates e-graphs for symbolic math: tighter MCTS regret bounds, reduced DRL gradient variance. Rust FFI cost amortized over data generation, not inference. |
| **Grammar-constrained decoding** | CRANE (ICML 2025) confirms up to 10% accuracy improvement from structured generation constraints. Prefix-notation grammar is simple enough for O(1) enforcement. |
| **PUCT-MCTS** | AlphaIntegrator (ETH, Oct 2024) validates MCTS for integration. DeepSearch (2025) confirms MCTS excels when branching factor is moderate and value estimates are noisy — both apply here. |
| **Equivalence-class CE** | EGG-SR (2025) validates aggregating rewards across equivalent forms reduces gradient variance. Sound theoretical motivation for symbolic integration where multiple correct answers exist. |

### Suboptimal — Recommended Changes

| Component | Current | Better | Evidence | Impact |
|-----------|---------|--------|----------|--------|
| **Token-to-param ratio** | 15:1 (1.4B tokens) | 100–300:1 (9.5–28B tokens) | LLaMA 3 trains at 1875:1. Epoch AI (2025): average open-weight ratio now 300:1. For inference-optimized small models, overtraining is standard practice. Synthetic data generation is cheap here. | **Critical** — largest potential accuracy gain |
| **Default PE** | Sinusoidal | RoPE | Dominant in production (LLaMA 3, Gemma). Exponential decay models tree-distance in prefix notation. ALiBi's windowed attention harms long-range symbolic deps. | **Medium** — expect 1-3% accuracy gain |
| **Dual architecture** | Separate Tree GNN (12M) | Tree PE in main transformer | Barket et al. (Aug 2025): Tree Transformer with tree-PE achieves ~90% on integration algorithm selection, outperforming both flat-seq and standalone GNNs. Transformers subsume GNN message-passing (arXiv:2506.22084). | **High** — eliminates fusion complexity, reclaims 12M params |
| **Soft gating** | Softmax weighting | Sigmoid gating | Sigmoid gating is more sample-efficient than softmax (arXiv:2405.13997). If dual architecture retained, switch activation. | **Low** — marginal improvement if dual-arch kept |
| **Curriculum** | Static Platanios (2019) | Self-paced (model-confidence-based) | Self-Adaptive CL (ACL 2025) uses model's own confidence to pace, eliminating hand-defined difficulty metrics. | **Low** — competence-based already implemented as option |

### Acceptable — Monitor for Future Improvement

| Component | Status | Alternative to Watch |
|-----------|--------|---------------------|
| MCTS | Good default | BFS-Prover (ACL 2025): simpler best-first search beats MCTS when policy confidence is high. Consider hybrid: BFS for easy, MCTS for hard. |
| Weight tying | Standard practice | May limit decoder expressiveness at this scale. Ablate. |
| SWA | Good practice | EMA (exponential moving average) with β=0.999 may be simpler and equally effective. |
| PCGrad | Optional, off by default | GradDrop (2024) or Nash-MTL may be more stable multi-task solutions. |

## Quickstart

```bash
# Build Rust core (requires Rust 1.70+)
cd rust/core && cargo build --release

# Install Python package (editable, with Rust extension)
pip install -e ".[dev]"

# Generate training data (1.5M pairs)
python scripts/generate_data.py --config configs/default.toml --output data/

# Train sequence transformer
python scripts/train.py --model seq --config configs/default.toml --data data/

# Run PE ablation
python scripts/ablate_pe.py --output-dir results/pe_ablation

# Evaluate
python scripts/evaluate.py --model seq --checkpoint checkpoints/seq_best.pt
```

## Configuration

All hyperparameters in `configs/default.toml`. Key knobs:

```toml
[model.seq_transformer]
pe_type = "rope"          # Switch default PE

[model.tree_gnn]
pe_type = "depth_index"   # Enable tree positional encoding

[model.gating]
enabled = true            # Activate soft gating

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
│   ├── diff.rs            Symbolic differentiation
│   ├── verify.rs          CAS verification (symbolic + numerical)
│   ├── canonical.rs       E-graph equality saturation
│   ├── features.rs        344-dim feature extraction
│   ├── env.rs             4-action integration environment
│   └── actions.rs         Action-space operations
├── python/neurips/
│   ├── data/              Tokenizer, features, verification, splitting
│   ├── models/            Seq transformer, tree GNN, gating, policy heads
│   ├── training/          Curriculum, loss, trainer, self-play, distillation
│   ├── inference/         Beam search, MCTS, action search
│   └── evaluation/        Benchmarking, oracle, analysis
├── scripts/               Training, evaluation, ablation runners
├── configs/               TOML configuration
├── specs/                 Feature specifications (P4-F001 through F007)
└── tests/python/          235 tests (unit + integration)
```

## References

- Lample & Charton (2019). Deep Learning for Symbolic Mathematics. arXiv:1912.01412
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864
- Press et al. (2022). ALiBi: Train Short, Test Long. ICLR 2022
- Platanios et al. (2019). Competence-based Curriculum Learning. NAACL 2019
- Barket et al. (2025). Tree-Based Deep Learning for Ranking Integration Algorithms. arXiv:2508.06383
- EGG-SR (2025). Embedding Symbolic Equivalence into Symbolic Regression. arXiv:2511.05849
- AlphaIntegrator (2024). Transformer Action Search for Symbolic Integration Proofs. arXiv:2410.02666
- BFS-Prover (2025). Scalable Best-First Tree Search for LLM Provers. ACL 2025
- CRANE (2025). Reasoning with Constrained LLM Generation. ICML 2025
