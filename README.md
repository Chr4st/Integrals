# Integral Prediction Engine

Transformer-based symbolic integral solver. Takes a math expression, returns its antiderivative in LaTeX.

```python
from integral_engine.solver import solve_integral

solve_integral("sin(x)")
# {"status": "solved_sympy", "latex": "- \\cos{x} + C", ...}

solve_integral("exp(sin(x)) * cos(x)")
# {"status": "solved_ml", "latex": "e^{\\sin{x}} + C", "confidence": 0.87, ...}
```

## Inference Pipeline

```
  "sin(x)*exp(cos(x))"
          |
          v
  +--- Stage 1: SymPy (4s timeout, subprocess) ---+
  |  sp.integrate(expr, x)                        |
  |  success? ──yes──> return LaTeX + C            |
  +------------------------------------------------+
          | no / timeout
          v
  +--- Stage 2: ML (15s timeout, in-process) -----+
  |                                                |
  |  1. Tokenize to prefix:  ["Mul","sin","x",    |
  |     "exp","cos","x"]  →  vocab indices         |
  |                                                |
  |  2. Extract depth features:  608-dim vector    |
  |                                                |
  |  3. Sample 40 candidates (T=0.7, top-p=0.95)  |
  |     for each candidate:                        |
  |       decode tokens → SymPy expr               |
  |       solve constants (C1, C2, ...)            |
  |       verify: diff(F, x) == f ?                |
  |       ──yes──> return LaTeX + C                |
  |                                                |
  |  4. Fallback: beam search (width=10)           |
  |     try each beam through constant solver      |
  |     verify by differentiation                  |
  |     ──yes──> return LaTeX + C                  |
  |                                                |
  +------------------------------------------------+
          | all failed
          v
    {"status": "failed"}
```

SymPy runs in a separate process (`ProcessPoolExecutor`) because it can hang on hard integrals and the only reliable kill is process termination. The ML model runs in a thread to keep weights in GPU VRAM.

**Key insight**: with a perfect verifier (SymPy differentiation), candidate diversity beats per-candidate quality. 40 independent samples at 60% individual accuracy gives 99.99% aggregate success probability: `1 - (1-p)^N`.

## Model Architecture

Pre-norm encoder-decoder Transformer. Default config (~44.8M params):

| Parameter | Default | Configurable range |
|---|---|---|
| `d_model` | 512 | 64 - 1024 |
| `nhead` | 8 | must divide d_model (head_dim = d_model/nhead) |
| `num_encoder_layers` | 6 | any (e.g. 4 for asymmetric) |
| `num_decoder_layers` | 6 | any (e.g. 12 for asymmetric) |
| `dim_feedforward` | 2048 | typically 4 * d_model |
| `dropout` | 0.1 | |
| `vocab_size` | 204 | fixed by vocabulary.py |
| `feature_dim` | 608 | 76 heads x 8 depth bins |

### Encoder

```
Input tokens: (batch, src_len) int indices from vocab of 204 tokens
                |
                v
  src_embedding: Embedding(204, 512, padding_idx=0)
  scale by sqrt(512) ≈ 22.6
                |                        Depth features: (batch, 608)
                v                                 |
  (batch, src_len, 512)                  feature_proj: Linear(608, 512)
                |                                 |
                v                                 v
            [ concat dim=1 ]  <───  (batch, 1, 512)   ← [FEAT] token
                |
                v
  (batch, src_len+1, 512)    ← +1 for prepended [FEAT]
                |
  sinusoidal positional encoding + dropout
                |
                v
  6x TransformerEncoderLayer (pre-norm):
    LayerNorm(512)
    MultiheadAttention(512, 8 heads → head_dim=64)
    residual + dropout
    LayerNorm(512)
    FFN: Linear(512,2048) → ReLU → Linear(2048,512)
    residual + dropout
                |
                v
  memory: (batch, src_len+1, 512)
```

The depth feature vector encodes **which mathematical functions appear and how deeply nested they are** relative to x. 76 function heads (sin, cos, exp, Pow, Add, Mul, ...) x 8 logarithmic depth bins = 608 floats. This is projected to 512 dims and prepended as a `[FEAT]` token, giving the encoder a structural summary before it processes the token sequence.

### Decoder

```
Target tokens: (batch, tgt_len) with BOS prepended
                |
                v
  tgt_embedding: Embedding(204, 512, padding_idx=0)
  scale by sqrt(512)
                |
  sinusoidal positional encoding + dropout
                |
                v
  (batch, tgt_len, 512)
                |
  causal mask: upper-triangular (tgt_len, tgt_len)
                |
                v
  6x TransformerDecoderLayer (pre-norm):
    LayerNorm(512)
    Masked self-attention (512, 8 heads)     ← causal
    residual + dropout
    LayerNorm(512)
    Cross-attention to encoder memory        ← (batch, src_len+1, 512)
    residual + dropout
    LayerNorm(512)
    FFN: Linear(512,2048) → ReLU → Linear(2048,512)
    residual + dropout
                |
                v
  output_proj: Linear(512, 204)
                |
                v
  logits: (batch, tgt_len, 204)
```

### Decoding Strategies

**Temperature sampling** (primary, 40 tries): At each step, scale logits by 1/T, sort by probability, mask tokens past cumulative probability `top_p`, sample from filtered distribution. Each sample is independently verified by differentiating the decoded expression.

**Beam search** (fallback, width 10): Maintains 10 hypotheses scored by length-normalized log probability using Wu et al. (2016) length penalty: `score / ((5 + len) / 6)^0.7`.

**MCTS** (optional, `mcts.py`): UCB1 tree search over the token vocabulary. Terminal reward = binary SymPy verification. Grammar mask prunes syntactically invalid expansions at each node.

### Constant Resolution

When the model predicts a template like `C1*sin(x) + C2`, the constant solver finds values for C1, C2 such that `diff(template, x) == integrand`:

1. **Symbolic** (3s): differentiate template, `sp.solve` the resulting system
2. **Numeric** (10s): Sobol quasi-random points on [-10, 10], multi-start L-BFGS-B optimization, `nsimplify` with cascading tolerances to rationalize results

## From Zero to Solving Integrals

### 1. Generate training data

```bash
python -m integral_engine.data_generator \
  --output data/processed/ --count 200000
```

Backward generation (Lample & Charton 2020): build random antiderivatives F(x), differentiate to get f(x), tokenize both. Always produces verified pairs.

### 2. Train

```bash
python -m integral_engine.train \
  --data data/processed/ \
  --checkpoint checkpoints/ \
  --epochs 50 \
  --batch-size 256
```

Optional enhancements via CLI flags:

| Flag | What it does |
|------|-------------|
| `--num-encoder-layers 4 --num-decoder-layers 12` | Asymmetric depth (Spec 2) |
| `--d-model 768 --nhead 12` | Scale up (Spec 8) |
| `--curriculum --curriculum-warmup-epochs 10` | Easy-to-hard training (Spec 11) |
| `--num-backward-decoder-layers 6 --consistency-weight 0.5` | Cycle consistency loss (Spec 4) |

### 3. Post-training (optional)

```bash
# Average last 5 checkpoints (+1-2% accuracy, free)
python -m integral_engine.average_checkpoints checkpoints/

# Self-training: infer on unlabeled pool, verify, retrain
python -m integral_engine.self_train \
  --checkpoint checkpoints/best.pt \
  --pool-size 10000 \
  --output data/self_train/
```

### 4. Solve

```python
from integral_engine.solver import solve_integral
result = solve_integral("x**2 * exp(x)")
```

Set `INTEGRAL_MODEL_PATH` to use a specific checkpoint. `INTEGRAL_N_SAMPLES` controls sampling budget (default 40).

## Module Map

```
solver.py            Public API: solve_integral()
model.py             Encoder-decoder Transformer
tokenizer.py         SymPy <-> prefix token sequences (base-100)
feature_extractor.py 608-dim depth-encoded structural features
constant_solver.py   Symbolic + numeric constant resolution
grammar_mask.py      Arity-based constrained decoding
vocabulary.py        Token/feature definitions, schema hashing

train.py             Training loop (curriculum, consistency, grad accum)
data_generator.py    Backward data generation
dataset.py           SIRD CSV preprocessing
augmentation.py      Expression rewriting (commutativity, trig, expand/factor)
curriculum.py        Difficulty-based sampling schedule
self_train.py        Inference-verify-retrain loop

rope.py              Rotary Position Embeddings (drop-in layer replacement)
mcts.py              Monte Carlo Tree Search decoder
value_network.py     MLP solvability estimator for MCTS
contrastive.py       Symbolic-numeric contrastive pre-training (InfoNCE)
router.py            Integration technique classifier (4-class routing)
average_checkpoints.py  Stochastic Weight Averaging
```

## Tests

```bash
python -m pytest integral_engine/tests/ -v
# 219 tests, ~5s
```
