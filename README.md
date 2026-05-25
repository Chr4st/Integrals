# Neural Symbolic Integration

A neural network that learns to compute indefinite and definite integrals. Given an integrand like `sin(x) * cos(x)`, it predicts the antiderivative `sin(x)^2 / 2`, then verifies the answer is correct by differentiating it symbolically.

The core idea: mathematical expressions are trees, not strings. Instead of flattening `sin(x*y + 1)` into a token sequence and hoping a transformer rediscovers the tree structure, we process the expression tree directly with a graph neural network. This lets a 12M-parameter model match the accuracy of 95M-parameter sequence models from prior work (Lample & Charton, 2019).

## What the System Does

**Training data generation**: We don't scrape textbook integrals. Instead, we generate millions of verified pairs by working backwards: pick a random expression F(x), differentiate it to get f(x), and pair them as (f(x), F(x)). Since differentiation is exact, every pair is correct by construction. A Rust engine handles this at ~50,000 pairs/minute.

**Learning**: The model sees 1.5 million of these pairs across five types of integrals: univariate, multivariate, definite, parametric, and special-function. It learns patterns --- u-substitution looks like f(g(x))*g'(x), integration by parts has a polynomial times a transcendental, and so on.

**Inference**: Given a new integrand, the model generates 25 candidate antiderivatives. Each one is checked by differentiating it and comparing to the original integrand. If any candidate passes, we return it. This "sample-and-verify" strategy turns a model that gets individual predictions right ~30% of the time into a solver that succeeds >96% of the time.

## Why a Tree Architecture

Previous neural integration systems (Lample & Charton, 2019) serialize expressions into prefix notation --- `+ sin x cos x` --- and use a standard seq2seq transformer. This works, but the model has to spend capacity learning that `sin` takes one argument, that `+` takes two, and how parenthetical nesting maps to attention patterns. All of this is already encoded in the tree structure.

Our model reads the expression tree directly:

```
       +              The tree already tells us:
      / \             - "+" has two children
    sin   cos         - "sin" and "cos" each have one child
     |     |          - "x" is a leaf (no children)
     x     x          No flattening needed.
```

The result: 12.1M parameters vs 95M, trainable on a single GPU in ~15 hours, with comparable accuracy on univariate integrals and the ability to handle multivariate integrals (which prior work could not do symbolically).

## Architecture

The system has four main components: data generation, the neural model, inference search, and verification.

### 1. Data Generation (Rust)

All training data is generated synthetically via backward construction. The Rust core (`rust/core/src/`) handles this:

**Expression trees** (`expr/`): Every mathematical expression is represented as a tree. Leaf nodes are numbers, variables (`x`, `y`), or constants (`pi`, `e`). Internal nodes are operators: binary (`+`, `-`, `*`, `/`, `^`) or unary (`sin`, `cos`, `exp`, `log`, `sqrt`, and 20+ others including special functions like `erf` and Bessel functions).

**Random generation** (`gen.rs`): Builds random expression trees with configurable depth (up to 10) and node count (up to 30). Five generation modes control what kind of expressions appear:
- *Univariate*: expressions in `x` only
- *Multivariate*: expressions in `x` and `y` (or `x`, `y`, `z`)
- *Definite*: evaluated at sampled bounds F(b) - F(a)
- *Parametric*: includes symbolic parameters (alpha, beta) treated as constants
- *Special function*: includes erf, Bessel, elliptic integrals, etc.

Smart constructors simplify during construction: `0 * anything` becomes `0`, `x^1` becomes `x`, `--x` becomes `x`. This prevents degenerate expressions from polluting the dataset.

**Coverage-guaranteed generation** (`gen_coverage.rs`): Random generation tends to over-represent simple patterns and under-represent important structural families. To fix this, we enumerate 200+ structural skeletons across 15+ families (elementary, u-substitution, integration by parts variants, partial fractions, trig substitution, etc.) and ensure each family appears proportionally. The final dataset is 70% skeleton-based (uniform across families) and 30% random exploration.

**Differentiation** (`diff.rs`): Symbolic differentiation that applies chain rule, product rule, quotient rule, and all special-case rules. This runs during data generation (to produce integrands from random antiderivatives) and during verification (to check model outputs). Fused with algebraic simplification at construction time.

**Verification** (`verify.rs`): Two-stage checking of whether a candidate antiderivative is correct:
1. *Symbolic*: Differentiate the candidate, canonicalize both sides, compare. Resolves >97% of cases.
2. *Numerical fallback*: Evaluate both expressions at 20 random points. Accept if at least 18/20 match within tolerance 1e-8.

Verification runs at ~0.01ms per pair, parallelized via Rayon. This makes batch verification of all 25 candidates per test problem sub-second.

**Canonicalization** (`canonical.rs`): E-graph equality saturation using the `egg` crate. Applies 20 algebraic rewrite rules (commutativity, associativity, trig identities, log rules, etc.) to extract the simplest equivalent form of an expression. Used for deduplication during data generation and for producing multiple equivalent targets per training example.

**Feature extraction** (`features.rs`): Computes a 344-dimensional feature vector for each expression, used to inject structural information into the model. Includes: operator-type histograms at each depth level (264-dim), variable role features (16-dim), analytical signature classification (40-dim: gaussian, oscillatory, rational, singular, exponential), and complexity scalars (24-dim).

**The Rust-Python bridge**: All Rust code is exposed to Python via PyO3. The Python training code calls into Rust for data generation, differentiation, and verification. This gives 100-1000x speedup over the pure-Python SymPy fallback.

### 2. Neural Model (Python/PyTorch)

The model (`python/neurips/models/`) is a tree GNN encoder followed by a top-down tree decoder. Total: 12.1M parameters.

**Encoder (~5.6M parameters)**

*Node Embedding*: Each tree node is embedded as a 256-dimensional vector by concatenating three projected feature groups:
- Symbol embedding (256 token types -> 64-dim): What operator or operand this node represents
- Role MLP (12 features -> 64-dim): Is it a root? A leaf? Left child or right child? What's its arity and depth parity? The child-index feature is critical --- it tells the model that in `a - b`, the left child `a` is the minuend and the right child `b` is the subtrahend, information that would be lost by symmetric aggregation.
- Structural MLP (40 features -> 128-dim): Subtree size, depth, sibling count, local shape

These concatenate to 256 dimensions per node.

*Message Passing* (8 rounds): Information flows along tree edges in both directions. Each round:
1. Parent-to-child: each parent sends a learned projection of its embedding to all children
2. Child-to-parent: each child sends a learned projection upward; the parent aggregates via mean pooling
3. Update: each node's embedding is updated by an MLP applied to the concatenation of its current state and received messages [768 -> 512 -> 256], with residual connection and layer normalization

After 8 rounds, every node's embedding captures information from the entire tree, but propagated along tree edges rather than through all-pairs attention. This is linear in node count per round, vs quadratic for transformer self-attention.

*Variable-Aware Attention* (8 heads): For multivariate integrals like integrating f(x,y) with respect to x, the model needs to know which subtrees depend on x (and should be transformed) vs which depend only on y (and should pass through unchanged). A bottom-up pass computes a boolean dependency mask: a node is "x-dependent" if any descendant contains x. This mask becomes a pairwise attention bias --- nodes sharing the same dependency status attend more strongly to each other.

**Decoder (~6.5M parameters)**

The decoder generates the output expression tree top-down, level by level, in breadth-first order:
1. Start with a single root node (seed embedding from encoder)
2. At each level, every frontier node queries the encoder output via 8 cross-attention layers (pre-norm, 256-dim, 8 heads), each followed by a SwiGLU feedforward layer
3. A symbol head (linear projection to vocab size 256) predicts what operator or operand each frontier node should be
4. Non-leaf predictions (arity > 0) spawn child nodes, which become the next level's frontier
5. Decoding stops when all frontier nodes are leaves or 8 levels are reached

This generates structurally valid trees by construction, unlike sequence decoders that can produce unparseable outputs.

**Tree Positional Encoding** (pluggable, configurable in `configs/default.toml`):
- `depth_index`: Sinusoidal encoding of depth plus a learnable embedding for child position (root/left/right)
- `rwse`: Random-walk structural encoding --- k-step landing probabilities capture local graph topology
- `laplacian`: Eigenvectors of the normalized graph Laplacian, capturing global tree structure

**Grammar-Constrained Decoding** (`grammar.py`): An arity stack tracks remaining child slots during generation. At each step, the grammar mask restricts predictions: if the expression is complete, only end-of-sequence is allowed; if nesting exceeds depth 20, only leaves are allowed; otherwise all tokens are valid. This guarantees every output is a syntactically valid expression. Cost: O(1) per token.

### 3. Inference Search

At test time, the model doesn't just predict one answer. It uses a sample-and-verify strategy:

**Basic inference**: Generate 25 candidates using temperature sampling (T=0.7, top-p=0.95) with grammar constraints. Verify each via Rust CAS. Return the first correct one.

**MCTS for step-by-step integration** (`inference/mcts.py`): For harder problems, the system can decompose integration into a sequence of steps using Monte Carlo Tree Search with 4 actions:
- `substitute(u=g(x))`: Apply a u-substitution, choosing which subtree to substitute via a pointer attention head
- `integrate_by_parts(u, dv)`: Split the integrand into u and dv factors via dual pointer heads
- `partial_fractions`: Decompose a rational function
- `close(F)`: Declare the current expression as the final answer

MCTS uses the PUCT selection formula (Q + c*P*sqrt(N_parent)/(1+N_child), c=1.4) with neural value estimation (no random rollouts). A learned ValueHead (MLP -> Tanh) estimates the probability of solving from any state. An LRU cache (100K entries) deduplicates environment steps. Dirichlet noise at the root (alpha=0.3, frac=0.25) encourages exploration.

### 4. Training Pipeline

**Dataset**: 1.5M verified pairs split 80/20 by skeleton (not by example). This means all instantiations of the same structural template go into the same split, preventing the model from memorizing coefficient variations of seen structures.

**Loss function**: Equivalence-class cross-entropy. For each training integrand, there may be K algebraically equivalent antiderivatives (found via e-graph canonicalization). The loss is the minimum CE over all K targets: `loss = min_k CE(prediction, target_k)`. This lets the model learn any correct form rather than being penalized for producing a valid but different-looking answer.

**Curriculum** (4-phase, 90 epochs):
1. Epochs 1-10: Univariate only, easy/medium difficulty
2. Epochs 11-30: Add multivariate, up to hard
3. Epochs 31-60: All 5 task types, all difficulty tiers
4. Epochs 61-90: Uniform sampling

An optional competence-based adaptive curriculum (Platanios et al., 2019) adjusts difficulty based on model performance: `competence(t) = min(1, sqrt(t*(1-c0^2)/T + c0^2))`, starting at c0=0.1. After 30% warmup, it enters self-paced mode, up-weighting examples in the "learning zone" (loss trending down) and down-weighting mastered or stalled examples.

**Optimization**: AdamW (lr=3e-4, weight_decay=0.01), 5-epoch linear warmup, cosine annealing to 1e-6, SWA at 75% of training (lr=1e-5), gradient clipping at max norm 1.0, BF16 mixed precision on Ampere+ GPUs, torch.compile with max-autotune on CUDA.

**Self-play**: After initial training, the model improves through MCTS self-play. REINFORCE with MCTS-guided trajectory generation produces (state, action, reward) tuples that train the action policy and value network.

## Quickstart

```bash
# Build Rust core (requires Rust 1.70+)
cd rust/core && cargo build --release

# Install Python package
pip install -e ".[dev]"

# Generate training data (1.5M pairs, ~30 min on 8 cores)
python scripts/generate_data.py --config configs/default.toml --output data/

# Or generate coverage-guaranteed data (Rust FFI with Python fallback)
python scripts/generate_covered.py --total 1500000 --output data/covered/

# Train (~15 hours on a single A100)
python scripts/train.py --config configs/default.toml --data-dir data/

# Evaluate
python scripts/evaluate.py --model tree --checkpoint checkpoints/tree_best.pt
```

## Configuration

All hyperparameters live in `configs/default.toml`:

```toml
[data]
total_pairs = 1_500_000     # Number of training pairs to generate
train_ratio = 0.8           # 80/20 skeleton-stratified split

[model.tree_gnn]
node_dim = 256              # Embedding dimension per tree node
message_rounds = 8          # Rounds of bidirectional message passing
decoder_levels = 8          # Maximum depth of generated output trees
n_heads = 8                 # Attention heads in variable-aware attention + decoder
pe_type = "none"            # Tree positional encoding: none/depth_index/rwse/laplacian

[training]
batch_size = 256
lr = 3e-4
epochs = 90
patience = 15               # Early stopping patience

[training.curriculum]
type = "static"             # static (4-phase) or competence (adaptive)

[inference.mcts]
n_simulations = 100         # MCTS simulations per problem
puct_c = 1.4                # Exploration constant
```

## Project Structure

```
rust/core/src/              Symbolic algebra backend (~4K lines of Rust)
  expr/                       Expression tree types, parsing, serialization
  gen.rs                      Random expression tree generation (5 modes)
  gen_coverage.rs             Coverage-guaranteed skeleton enumeration (200+ skeletons)
  diff.rs                     Symbolic differentiation (chain/product/quotient rules)
  verify.rs                   Two-stage verification (symbolic + numerical)
  canonical.rs                E-graph equality saturation (egg crate, 20 rewrite rules)
  features.rs                 344-dim structural feature extraction
  env.rs                      4-action integration environment for MCTS
  actions.rs                  Substitute, IBP, partial fractions implementations
  lib.rs                      PyO3 bindings exposing everything to Python

python/neurips/
  models/
    tree_gnn.py               GNN encoder: node embedding + message passing + var-aware attention
    tree_decoder.py            Top-down autoregressive tree decoder
    tree_positional.py         Pluggable tree PE (depth_index, RWSE, Laplacian)
    grammar.py                 Arity-based grammar mask for constrained decoding
    action_policy.py           4-action policy head for step-by-step integration
    activations.py             SwiGLU activation
  data/
    tokenizer.py               Base-100 number encoding, prefix notation tokenization
    dataset.py                 Precomputed uint8 tensor cache (8x RAM reduction)
    split.py                   Skeleton-based train/test splitting (zero structural leakage)
    features.py                Python-side feature computation
    verify.py, verify_impl.py  SymPy verification fallback
    vocab.py, prefix.py        Vocabulary and prefix notation utilities
  training/
    trainer.py                 Main training loop with AMP, gradient accumulation, SWA
    loss.py                    Equivalence-class cross-entropy (min-over-K targets)
    curriculum.py              Static 4-phase + competence-based adaptive scheduling
    difficulty.py              Difficulty scoring for curriculum decisions
    self_play.py               MCTS self-play trajectory generation + REINFORCE
    auxiliary.py               Auxiliary losses (difficulty classification, depth regression)
    train.py                   High-level training orchestration
    checkpoint.py, dwa.py, pcgrad.py  Utilities
  inference/
    mcts.py                    PUCT-MCTS with neural value estimation
    verify_cache.py            LRU cache for deduplicating verification calls
    action_search.py           Action-space search coordination
  evaluation/
    benchmark.py               Evaluation pipeline (sample-and-verify, metrics)
    oracle.py                  Verification oracle interface
    analysis.py, ablations.py  Result analysis utilities

scripts/                     Entry points for training, evaluation, data generation
configs/                     TOML configuration files
tests/python/                Unit and integration tests
paper/                       LaTeX source for the paper
```

## Prior Work

This project builds on Lample & Charton (2019), who showed that encoder-decoder transformers can learn symbolic integration by treating it as sequence-to-sequence translation. Their 95M-parameter model trained on 40M pairs achieves >99% accuracy on univariate integrals with a random train/test split. Our tree-native approach achieves 96.1% on a harder skeleton-split evaluation (zero structural overlap between train and test) with 8x fewer parameters and 27x less training data, while extending to multivariate, definite, parametric, and special-function integrals.

## References

- Lample & Charton (2019). Deep Learning for Symbolic Mathematics. arXiv:1912.01412
- Platanios et al. (2019). Competence-based Curriculum Learning. NAACL 2019
- AlphaIntegrator (2024). Transformer Action Search for Symbolic Integration Proofs. arXiv:2410.02666
- Barket et al. (2025). Tree-Based Deep Learning for Ranking Integration Algorithms. arXiv:2508.06383
- EGG-SR (2025). Embedding Symbolic Equivalence into Symbolic Regression. arXiv:2511.05849
- BFS-Prover (2025). Scalable Best-First Tree Search for LLM Provers. ACL 2025
- CRANE (2025). Reasoning with Constrained LLM Generation. ICML 2025
