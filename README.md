# Integral Prediction Engine

Transformer-based symbolic integral solver. Takes a math expression, returns its antiderivative in LaTeX.

```python
from integral_engine.solver import solve_integral

solve_integral("sin(x)")
# {"status": "solved_sympy", "latex": "- \\cos{x} + C", ...}

solve_integral("exp(sin(x)) * cos(x)")
# {"status": "solved_ml", "latex": "e^{\\sin{x}} + C", "confidence": 0.87, ...}
```

## Architecture

```
                      expression string
                             |
              +--------------+--------------+
              |                             |
        Stage 1: SymPy               Stage 2: ML
     (4s timeout, subprocess)     (15s timeout, in-process)
              |                             |
         sp.integrate              tokenize to prefix notation
              |                    extract 608-dim depth features
          success? ----yes----->   encode with [FEAT] prepend
              |                             |
              no                   sample 40 candidates (T=0.7)
              |                    verify each: diff(F,x) == f?
              +----------->                 |
                                   no hit? beam search (width=10)
                                   resolve constants (L-BFGS-B)
                                            |
                                    LaTeX + C  or  "failed"
```

**Core model**: Encoder-decoder Transformer (~44.8M params default). Pre-norm, 512-dim embeddings, 8 heads, 6+6 layers. Configurable to asymmetric depths (e.g. 4 enc + 12 dec) and larger dimensions (768/1024).

**Key insight**: With a perfect verifier (SymPy differentiation), candidate diversity beats per-candidate quality. 40 independent samples at 60% individual accuracy gives 99.99% aggregate success.

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
