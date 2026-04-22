# Integral Prediction Engine

Technical deep-dive into a ML-powered symbolic integral solver.

## What It Does

The integral engine takes a mathematical expression as a string (e.g. `"sin(x)"`) and returns its antiderivative in LaTeX. It uses a two-stage pipeline: first attempting direct symbolic integration via SymPy, then falling back to a custom Transformer model that generates diverse candidate antiderivatives, verifies them by differentiation, and resolves unknown constants.

```
solve_integral("sin(x)")
-> {"status": "solved_sympy", "latex": "- \\cos{x} + C", "method": "SymPy direct integration", "confidence": null}

solve_integral("exp(sin(x)) * cos(x)")
-> {"status": "solved_ml", "latex": "e^{\\sin{x}} + C", "method": "Transformer prediction with constant solving", "confidence": 0.87}
```

## Architecture

### Two-Stage Pipeline

```
Input: expression string
|
+- Stage 1: SymPy Direct Integration (persistent ProcessPoolExecutor, 4s timeout)
|    +- Success -> LaTeX + C -> return immediately
|    +- Timeout / Failure --+
|                           |
+- Stage 2: ML Inference <--+ (ThreadPoolExecutor, 15s timeout)
|    +- Tokenize expression to prefix notation (base-100)
|    +- Extract 608-dim depth feature vector
|    +- Encode with [FEAT] token prepend
|    |
|    +- Primary: Temperature sampling (T=0.7, top-p=0.95)
|    |   +- Generate 25 samples (batches of 5)
|    |   +- Grammar-constrained decoding (arity stack)
|    |   +- Verify each via SymPy differentiation
|    |   +- Return first verified result
|    |
|    +- Fallback: Beam search (width=10, length penalty alpha=0.7)
|    |   +- Try each candidate through constant solver
|    |   +- Verify by differentiation
|    |
|    +- Constant solving:
|        +- Symbolic: differentiate + sp.solve (3s)
|        +- Numeric: Sobol + multi-start L-BFGS-B + nsimplify (10s)
|
+- Both failed -> {"status": "failed"}
```

**Why two stages?** SymPy is exact and fast for textbook integrals but fails on non-elementary or complex compositions. The ML model handles a broader class of expressions but requires constant resolution and is slower. Running SymPy first avoids unnecessary ML inference for easy cases.

**Why sampling-first?** With a perfect verifier (SymPy differentiation), candidate *diversity* matters more than per-candidate quality. If per-sample accuracy is `p`, then `N` independent samples give success probability `1-(1-p)^N`. For `p=0.6, N=25`: 99.99%. This insight from Wang et al. (2023) "Self-Consistency" is even more powerful here because we have an exact verifier rather than majority voting.

**Why different executors?** SymPy runs in a separate process (`ProcessPoolExecutor`) because it can hang indefinitely on hard integrals and the only reliable way to kill it is process termination. The ML model runs in a thread (`ThreadPoolExecutor`) to keep model weights in GPU VRAM.

**Maximum latency**: 4s (SymPy) + 15s (ML) = 19s worst case.

### Module Dependency Graph

```
solver.py --------------- PUBLIC API: solve_integral()
+-- model.py              Pre-norm encoder-decoder Transformer (~44.8M params)
+-- tokenizer.py          SymPy <-> base-100 prefix token sequences
+-- feature_extractor.py  608-dim depth-encoded structural features
+-- constant_solver.py    Sobol + multi-start L-BFGS-B constant resolution
+-- grammar_mask.py       Prefix-notation constrained decoding
+-- vocabulary.py         Shared token/feature definitions

train.py ----------------- Training loop
+-- model.py
+-- dataset.py            SIRD CSV -> tokenized JSONL.gz shards
+-- data_generator.py     Backward data generation (random F -> diff -> f)
+-- augmentation.py       Expression rewriting augmentation
+-- tokenizer.py
+-- vocabulary.py

average_checkpoints.py --- Checkpoint weight averaging (post-training)
```

## Module Breakdown

### vocabulary.py -- Token Definitions

Two separate vocabularies serve different purposes:

**Feature vocabulary** (76 heads): Maps mathematical function names to indices for the depth feature vector. Includes standard functions (`sin`, `cos`, `exp`, `log`), special functions (`besselj`, `erf`, `gamma`), structural operators (`Add`, `Mul`, `Pow`), numeric types (`Integer`, `Rational`, `Float`), and specialized power heads: `Exp` (variable in exponent, e.g. `2^x`, `exp(x)`), `Tower` (variable in both base and exponent, e.g. `x^x`).

**Sequence vocabulary** (204 tokens): Used for prefix notation encoding. Contains 4 special tokens (`PAD`, `BOS`, `EOS`, `UNK`), sign markers (`INT+`, `INT-`), 20 constant placeholders (`C1`-`C20`), 70 operator/function tokens, and 100 base-100 number tokens (`0`-`99`).

**Depth binning**: 8 logarithmic bins with explicit boundaries `(1, 2, 3, 4, 6, 9, 15)`. Depths 1-4 get individual bins (preserving resolution at shallow depths). Deeper levels are grouped: 5-6, 7-9, 10-15, 16+.

**Feature schema hashing**: `feature_schema_hash()` computes a SHA256 hash of head names, indices, and depth config. Stored in checkpoints to detect incompatible feature schema changes at load time.

Key constants:
- `NUM_FEATURE_HEADS = 76`
- `NUM_DEPTH_BINS = 8`
- `FEATURE_DIM = 608` (76 x 8)
- `SEQ_VOCAB_SIZE = 204`
- `MAX_INPUT_LEN = 384`, `MAX_OUTPUT_LEN = 256`

### feature_extractor.py -- Structural Feature Extraction

Converts a SymPy expression tree into a 608-dimensional float vector encoding **which functions appear and how deeply nested they are** relative to the integration variable.

**Algorithm**: Depth-first walk of the expression tree. For each node, compute the minimum depth from that node to a leaf containing `x`. Map depth to a logarithmic bin via `depth_to_bin()`. Record `(function_head, depth_bin)` counts.

**Three-way power classification**:
- `Pow`: variable only in base (e.g. `x^2`)
- `Exp`: variable only in exponent (e.g. `2^x`, `exp(x)`)
- `Tower`: variable in both (e.g. `x^x`)

**Example**: `sin(log(x))` produces:
- `sin` at depth bin 1 (depth 2 -> bin 1)
- `log` at depth bin 0 (depth 1 -> bin 0)

**Vector layout**: Flat array where `vec[head_index * 8 + depth_bin] = count`.

This vector is injected into the Transformer encoder as a special `[FEAT]` token, giving the model a structural summary of the integrand before it sees the token sequence.

### tokenizer.py -- Prefix Notation Encoding

Bidirectional conversion between SymPy expressions and token sequences using **prefix (Polish) notation** with **base-100 integer encoding**.

**Encoding rules**:
| Expression | Prefix tokens |
|-----------|--------------|
| `x` | `["x"]` |
| `42` | `["INT+", "42"]` |
| `-3` | `["INT-", "3"]` |
| `123456` | `["INT+", "12", "34", "56"]` |
| `3/4` | `["Rational", "INT+", "3", "INT+", "4"]` |
| `sin(x)` | `["sin", "x"]` |
| `a + b + c` | `["Add", a, "Add", b, c]` (right-nested) |
| `x^2` | `["Pow", "x", "INT+", "2"]` |

Integers are encoded in **base-100 digit pairs** with a sign marker. N-ary operators (`Add`, `Mul`) are serialized as right-nested binary applications. Decoding uses base-100 arithmetic (`val = val * 100 + chunk`), not string concatenation.

`encode()` wraps tokens with `BOS`/`EOS` and maps to vocabulary indices. `decode()` reverses: strips special tokens, maps indices back to tokens, and parses with a recursive descent parser (`from_prefix`).

### model.py -- IntegralTransformer

Pre-norm encoder-decoder Transformer adapted for symbolic mathematics.

**Architecture**:
| Parameter | Value |
|-----------|-------|
| Total parameters | ~44.8M |
| Embedding dimension | 512 |
| Attention heads | 8 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Feedforward dimension | 2048 |
| Dropout | 0.1 |
| Vocabulary size | 204 |
| Feature dimension | 608 |
| Layer normalization | Pre-norm (`norm_first=True`) |
| Positional encoding | Sinusoidal (max_len=512) |
| Weight initialization | Xavier uniform |

**Encoder**: Embeds source tokens (scaled by `sqrt(d_model)`), projects the 608-dim depth feature vector to 512 dimensions via a linear layer, prepends it as a special `[FEAT]` token, applies sinusoidal positional encoding, then passes through 6 pre-norm Transformer encoder layers.

**Decoder**: Standard autoregressive decoder with causal masking and pre-norm. Cross-attends to encoder memory. Projects hidden states to vocabulary logits.

**Beam search** (`generate` method): Width=10 with Wu et al. (2016) length penalty (`alpha=0.7`). Maintains `beam_width` hypotheses, expands by top-k next tokens, keeps best candidates by length-normalized log probability.

**Temperature sampling** (`sample` method): Generates a single sample using temperature `T` and nucleus (top-p) filtering. At each step, scales logits by `1/T`, sorts by probability, masks tokens beyond cumulative probability `top_p`, then samples from the filtered distribution.

### constant_solver.py -- Resolving Template Constants

When the Transformer predicts a template like `C1*sin(x) + C2`, this module determines the values of `C1`, `C2`, ... such that the derivative of the filled template equals the original integrand.

**Phase 1 -- Symbolic solve** (3s timeout): Differentiate the template, set equal to the integrand, solve the resulting system with `sp.solve`. Works well for linear systems. Runs via persistent `ProcessPoolExecutor` singleton for timeout safety.

**Phase 2 -- Numeric solve** (10s timeout): When symbolic solving fails, fall back to optimization.
1. **Sobol sampling**: Generate quasi-random points on `[-10, 10]` using `scipy.stats.qmc.Sobol` for better space-filling coverage than uniform random
2. **Train/verify split**: 70% of points for optimization, 30% held out for validation
3. **Multi-start L-BFGS-B**: 3 random restarts (initial guesses: zeros, then uniform `[-5, 5]`)
4. **Acceptance**: Train MSE `<= 1e-6` AND verify max error `<= 1e-4`
5. **Cascading nsimplify**: Rationalize each constant with hints (`pi`, `E`, `sqrt(2)`, `sqrt(3)`, `log(2)`, `GoldenRatio`) at tolerances `[1e-10, 1e-8, 1e-6]`, falling back to `Rational.limit_denominator(10000)`
6. **Final verification**: Differentiate the filled result and check symbolic equality

### solver.py -- Public API

`solve_integral(expr_str, var="x")` is the single entry point.

**Return format**:
```python
{
    "status": "solved_sympy" | "solved_ml" | "failed",
    "latex": str | None,         # antiderivative in LaTeX with "+ C"
    "method": str,
    "confidence": float | None   # exp(beam log probability), ML only
}
```

**Inference strategy (sampling-first)**:
1. Generate 25 temperature samples (T=0.7, top-p=0.95), verify each by differentiation, return first verified result
2. If no sample verifies: fall back to beam search (width=10), try each candidate through constant solver
3. Both use persistent `ProcessPoolExecutor` singletons (not per-call creation)

The model is **lazily loaded** as a singleton on first ML call. Device auto-detection prefers CUDA > MPS > CPU. Prefers `checkpoints/averaged.pt` over `best.pt` when available. Checkpoint path overridable via `INTEGRAL_MODEL_PATH` env var.

### data_generator.py -- Backward Data Generation

Implements the Lample & Charton (ICLR 2020) backward generation technique: build random expression trees as antiderivatives `F(x)`, differentiate to get `f(x)`, tokenize both, output JSONL shards.

**Algorithm**:
1. Pick a difficulty tier: easy (depth 1-2), medium (depth 3-4), hard (depth 5-6)
2. Build a random expression tree using operators (`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `Add`, `Mul`, `Pow`) with domain guards (e.g. `log(|x|+1)`, `asin(sin(x))`)
3. Simplify to canonical form via `nsimplify(cancel(expand(...)))`
4. Differentiate: `f(x) = diff(F, x)` -- always succeeds and is fast
5. Reject if result is zero, contains `Piecewise`/`Integral`, or `x` not in free symbols
6. Tokenize, extract features, check length limits
7. Deduplicate by integrand hash (SHA256 of `srepr`)
8. Write to train/val/test JSONL.gz shards (80/10/10 split)

### augmentation.py -- Expression Rewriting Augmentation

Generates mathematically-equivalent variants of expressions to multiply effective dataset size by 3-4x:

- **Commutativity**: Randomly permute `Add`/`Mul` argument order
- **Expand/factor**: Randomly apply `sp.expand()` or `sp.factor()` to the expression or subexpressions
- **Trig rewrites**: Apply `trigsimp()`, `expand_trig()`, or rewrite in terms of `sin`/`cos`

All variants integrate to the same `F(x)` -- only the input representation changes.

### grammar_mask.py -- Constrained Decoding

`PrefixGrammarMask` tracks the arity stack during autoregressive generation and returns a boolean mask of valid next tokens at each step.

**Rules**:
- Binary ops (`Add`, `Mul`, `Pow`, `Rational`, ...): push 1 to the need counter (need 2 operands, consume 1 slot)
- Unary ops (`sin`, `cos`, `Neg`, ...): no change (need 1 operand, consume 1 slot)
- Terminals (`x`, `pi`, `E`, constants): decrement need counter
- `INT+`/`INT-`: enter integer mode, must see at least one number token
- `EOS`: valid only when need counter reaches zero (expression complete)

Eliminates syntactically invalid outputs, effectively increasing the usable candidate pool.

### average_checkpoints.py -- Checkpoint Weight Averaging

Implements Izmailov et al. (2018) "Stochastic Weight Averaging": averages the `state_dict` tensors of the last K checkpoints. Consistently yields +1-2% accuracy with no additional training.

Finds the K most recent `epoch_*.pt` files by modification time, averages all tensor values, preserves metadata (feature schema hash, config).

### dataset.py -- SIRD Data Preprocessing

Converts raw SIRD CSV data into tokenized JSONL.gz shards for training.

**Pipeline**:
1. Extract unique integrands from CSV (filters substitution markers `_u`, `_v`)
2. Integrate each with SymPy in parallel (`ProcessPoolExecutor`, 10s timeout)
3. Verify by differentiation; reject `Integral` or `Piecewise` results
4. Normalize antiderivatives so that `F(eval_point) = 0` (tries x=0, x=1, x=-1)
5. Tokenize integrand and antiderivative to prefix notation
6. Extract depth features
7. Filter by sequence length limits
8. Stratified split: 80% train, 10% val, 10% test
9. Shard into gzipped JSONL files (10,000 pairs per shard)

### train.py -- Training Loop

**Dataset loading**: `IntegralDataset` reads JSONL.gz shards, maps tokens to indices with BOS/EOS framing. Dynamic padding via custom `collate_fn`.

**Loss**: Cross-entropy with inverse-frequency class weights (capped at 10.0). PAD tokens excluded.

**Optimizer**: Adam with `betas=(0.9, 0.98)`, `eps=1e-9`.

**Learning rate schedule** (`WarmupInvSqrtScheduler`): Linear warmup over 4000 steps, then inverse square root decay.

**Training loop**: Teacher forcing with gradient clipping (`max_norm=1.0`). Supports gradient accumulation (`accumulation_steps`). Evaluates token accuracy and exact sequence match on validation set each epoch. Early stopping with patience=5 based on exact match.

**Checkpointing**: Saves rolling last-5 epoch checkpoints (for weight averaging) and best checkpoint. Each checkpoint includes `feature_schema_hash` for compatibility validation.

**Hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Batch size | 256 |
| Learning rate | 4e-5 |
| Warmup steps | 4000 |
| Early stopping patience | 5 |
| Gradient clip norm | 1.0 |
| Class weight cap | 10.0 |
| DataLoader workers | 2 |

## Training Data

| File | Records | Purpose |
|------|---------|---------|
| `data/expr_2109869.csv` | ~2.1M | Training source (integrand + rule name) |
| `data/expr_test_210976.csv` | ~210K | Test source |
| Backward-generated | Configurable | Synthetic pairs via differentiation |

After SymPy verification and tokenization filtering, roughly 9-10% of SIRD expressions produce valid training pairs (~180K). The backward generator can produce arbitrarily many additional verified pairs.

## Test Coverage

113 tests across 11 test files:

| Module | Test file | Tests |
|--------|-----------|-------|
| model.py | test_model.py | 16 (positional encoding, forward pass, beam search, temperature sampling) |
| feature_extractor.py | test_feature_extractor.py | 13 (depth binning, power classification, edge cases) |
| tokenizer.py | test_tokenizer.py | 15 (base-100 encoding, round trips, large integers) |
| data_generator.py | test_data_generator.py | 14 (tree generation, pair verification, dedup, sharding) |
| augmentation.py | test_augmentation.py | 11 (commutativity, expand/factor, trig, equivalence) |
| dataset.py | test_dataset.py | 10 (normalization, unique extraction, BOS/EOS, collation) |
| grammar_mask.py | test_grammar_mask.py | 9 (arity tracking, integer mode, nested expressions) |
| constant_solver.py | test_constant_solver.py | 7 (symbolic, numeric, Sobol, multi-start) |
| average_checkpoints.py | test_average_checkpoints.py | 7 (averaging, metadata preservation, edge cases) |
| solver.py | test_solver.py | 6 (SymPy stage, ML stage, error handling) |
| train.py | test_train.py | 5 (warmup scheduler, overfit smoke test, gradient accumulation) |

## Resolved Weaknesses

All 15 originally documented weaknesses have been fixed:

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | Critical | Silent error suppression in solver.py | Added structured logging (`structlog`) with per-solve trace IDs |
| 2 | Critical | No model checkpoint validation | `_validate_checkpoint()`: file existence, size, deserialization, key matching, shape matching, feature schema hash |
| 3 | Critical | Beam search scores not length-normalized | Wu et al. (2016) length penalty with `alpha=0.7` |
| 4 | Critical | Antiderivative normalization missing | `_normalize_antiderivative()`: evaluates at x=0, x=1, x=-1, subtracts constant |
| 5 | Major | Depth feature clipping at MAX_DEPTH=5 | 8 logarithmic bins with boundaries `(1,2,3,4,6,9,15)` |
| 6 | Major | Hardcoded constant limit C1-C9 | Expanded to C1-C20 |
| 7 | Major | Numeric solver fixed bounds [-2, 2] | Sobol quasi-random sampling on `[-10, 10]` |
| 8 | Major | No gradient accumulation | `accumulation_steps` parameter with proper loss scaling |
| 9 | Major | Inconsistent powerExpVar semantics | Three-way classification: Pow/Exp/Tower |
| 10 | Minor | Verbose integer tokenization | Base-100 encoding: `42`->`["INT+","42"]`, `123456`->`["INT+","12","34","56"]` |
| 11 | Minor | Fixed numeric solver seed | Configurable seed + train/verify split (70/30) |
| 12 | Minor | No tests for model/dataset/train | 113 tests across 11 files |
| 13 | Minor | Feature head ordering fragile | `feature_schema_hash()` detects mismatches at checkpoint load time |
| 14 | Minor | ProcessPoolExecutor overhead | Persistent singleton pools with `threading.Lock` + `atexit` cleanup |
| 15 | Minor | nsimplify tolerance too strict | Cascading tolerances `[1e-10, 1e-8, 1e-6]` with constant hints |

## Research Roadmap

Techniques not yet implemented, organized by expected impact. All entries cite published work with demonstrated accuracy gains on symbolic mathematics or closely related tasks.

### High Impact

**1. Tree Positional Encodings**
Replace sinusoidal PE with tree-aware encodings that reflect the AST structure of prefix-notation input. Prefix tokens are a pre-order tree traversal -- linear position indices lose parent-child and sibling relationships.

- Shiv & Quirk, "Novel Positional Encodings to Enable Tree-Based Transformers," NeurIPS 2019
- Wang et al., "Rethinking Positional Encoding in Tree Transformer for Code Representation," EMNLP 2022
- Implementation: Add depth + sibling-index embedding tables (~50 lines), modify tokenizer to emit tree metadata. Compatible with `nn.TransformerEncoder` (no architecture change needed).

**2. Asymmetric Encoder-Decoder Depth**
Use 4 encoder + 12 decoder layers instead of symmetric 6+6. The decoder task (generating the antiderivative step-by-step) is harder than the encoder task (understanding the integrand). Asymmetric architectures significantly outperformed symmetric ones in symbolic regression.

- d'Ascoli & Charton, "Deep Symbolic Regression for Recurrent Sequences," ICML 2022
- Implementation: Hyperparameter change only. ~44.8M -> ~60M params.

**3. Self-Training with Verification Loop**
Run inference on a large pool of unlabeled integrands, verify correct predictions by differentiation, add verified pairs to training data, retrain. The perfect verifier (SymPy differentiation) makes this especially powerful -- expert iteration improved automated theorem proving from 56.2% to 65.4%.

- Polu & Sutskever, "Generative Language Modeling for Automated Theorem Proving," 2020
- Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021

**4. Forward-Backward Consistency Training**
Train a secondary decoder for differentiation (F -> f). Use cycle consistency: for a predicted antiderivative F(x), run it through the differentiation decoder to check if it recovers f(x). Lample & Charton reported +2-5% from bidirectional training.

- Lample & Charton, "Deep Learning for Symbolic Mathematics," ICLR 2020
- He et al., "Dual Learning for Machine Translation," 2016

### Medium Impact

**5. MCTS-Guided Decoding**
Replace beam search with Monte Carlo Tree Search using a learned value function. Terminal reward = SymPy verification pass. Enables integrating non-differentiable feedback during generation. 10-100x compute cost.

- TPSR, "Transformer-based Planning for Symbolic Regression," NeurIPS 2023

**6. Contrastive Symbolic-Numeric Pre-training**
Dual encoders for symbolic and numeric representations with contrastive alignment. Pre-train on matching expressions to their numeric evaluations before fine-tuning on integration.

- SNIP, "Bridging Mathematical Symbolic and Numeric Realms with a Multimodal Foundation Model," ICLR 2024 Spotlight

**7. RoPE Relative Position Encodings**
Replace absolute sinusoidal PE with Rotary Position Embeddings for better relative position modeling and length generalization. Requires custom attention (not native `nn.TransformerEncoder`).

- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021

**8. Model Scaling (d_model=768-1024)**
Lample & Charton's best results used d_model=1024. Scaling laws for symbolic regression show clear power-law improvement with compute.

- Otte et al., "Towards Scaling Laws for Symbolic Regression," 2025

### Lower Impact

**9. Method-Routing Classifier**
Pre-classify integrands by integration technique (substitution, by-parts, partial fractions), route to specialized sub-models. Transformers outperformed human-crafted heuristics by up to 30% on method applicability prediction.

- Barket et al., "Transformers to Predict the Applicability of Symbolic Integration Routines," NeurIPS 2024 Workshop

**10. Tree Attention Masks**
Restrict encoder self-attention to structurally related tokens (parent-child, siblings in AST). Complementary to tree positional encodings. Passable as `src_mask` to `nn.TransformerEncoder`.

- Wang et al., "Tree Transformer: Integrating Tree Structures into Self-Attention," EMNLP 2019

**11. Curriculum Learning**
Train on easy expressions first (low depth), progressively add harder ones. Mixed evidence in the literature.

- Bengio et al., "Curriculum Learning," ICML 2009
- Saxton et al., "Analysing Mathematical Reasoning Abilities of Neural Models," 2019

### Explicitly Excluded

| Technique | Reason |
|-----------|--------|
| Label smoothing | Hurts exact-match accuracy (Muller et al. 2019) |
| Mixture of Experts | High complexity, uncertain gain at 44.8M param scale |
| Full iterative refinement | Requires second model or architecture changes |

## References

- Lample & Charton, "Deep Learning for Symbolic Mathematics," ICLR 2020
- Shiv & Quirk, "Novel Positional Encodings to Enable Tree-Based Transformers," NeurIPS 2019
- Wang et al., "Rethinking Positional Encoding in Tree Transformer," EMNLP 2022
- d'Ascoli & Charton, "Deep Symbolic Regression for Recurrent Sequences," ICML 2022
- Kamienny et al., "End-to-end Symbolic Regression with Transformers," NeurIPS 2022
- TPSR, "Transformer-based Planning for Symbolic Regression," NeurIPS 2023
- SNIP, "Bridging Mathematical Symbolic and Numeric Realms," ICLR 2024 Spotlight
- Barket et al., "Transformers to Predict Applicability of Symbolic Integration Routines," NeurIPS 2024 Workshop
- AlphaIntegrator, "Transformer Action Search for Symbolic Integration Proofs," arXiv 2410.02666
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," UAI 2018
- Xiong et al., "On Layer Normalization in the Transformer Architecture," ICML 2020
- Wang et al., "Self-Consistency Improves Chain of Thought Reasoning," ICLR 2023
- Polu & Sutskever, "Generative Language Modeling for Automated Theorem Proving," 2020
- Cobbe et al., "Training Verifiers to Solve Math Word Problems," 2021
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2021
- Otte et al., "Towards Scaling Laws for Symbolic Regression," 2025
- Wu et al., "Google's Neural Machine Translation System," 2016
