# Phase 7b: Structural Feature Extractor

## Goal
Build the feature extractor that computes a fixed-size numeric vector
describing the expression's structure. This vector helps the model
understand the integral BEFORE it starts decoding.

## What Features Are
Think of features as a fingerprint of the expression.
Instead of the model figuring out "this has nested sin/cos at depth 3"
from raw tokens, we pre-compute it and hand it to the model directly.
688 numbers that describe what the expression looks like.

## File: `python/neurips/data/features.py`

### Feature Vector: 688 dimensions total

```python
def extract_features(tree: ExprNode, int_var: str) -> np.ndarray:
    """Returns a 688-dim feature vector. Always the same size
    regardless of how big or small the expression tree is."""
```

**Part 1: Depth-encoded function heads (608-dim)**
76 function types × 8 depth bins = 608 numbers.

What this means: for each function (sin, cos, exp, add, mul, pow, ...),
count how many times it appears at each depth level.
- Depth bin 0: depth 0-1 (near root)
- Depth bin 1: depth 2-3
- ...
- Depth bin 7: depth 14+ (very deep)

Example: `sin(cos(x) + exp(x))` has:
- sin at depth 0 → bin 0 for sin gets +1
- cos at depth 1 → bin 0 for cos gets +1
- exp at depth 1 → bin 0 for exp gets +1
- add at depth 1 → bin 0 for add gets +1

This captures WHERE functions are, not just WHAT functions exist.
sin-at-root is very different from sin-buried-deep.

**Part 2: Variable role features (16-dim)**
- 5 bits: which variables appear? (x=1, y=0, z=0, w=0, t=0)
- 5 bits: which are integration targets? (x=1, rest=0)
- 3 floats: what fraction of nodes contain the integration variable?
  (measures how "spread out" x is in the tree)
- 3 floats: depth of deepest / shallowest / mean occurrence of int_var

**Part 3: Special function signatures (40-dim)**
8 bins × 5 signature types = 40 numbers.
- Gaussian decay: does the tree have exp(-x²) patterns? (signals erf)
- Oscillatory: nested sin/cos depth (signals Bessel, Fresnel)
- Rational: numerator/denominator polynomial degree (signals partial fractions)
- Algebraic singularity: 1/f(x) patterns (signals log, arctan answers)
- Exponential growth: exp(polynomial) patterns

**Part 4: Task + complexity features (24-dim)**
- Task type one-hot: [indef, def, param, special] (4 numbers, one is 1, rest 0)
- Total node count (normalized to 0-1 range)
- Max tree depth (normalized)
- Parameter count (0, 1, or 2)
- Bound type: [finite, semi-infinite, infinite, none] (4-dim one-hot)
- 13 reserved dimensions (set to 0, for future use)

### How Features Are Used
- **Sequence transformer**: features are projected 688→640 via a linear layer,
  then injected as the [FEAT] token embedding in the encoder input
- **Tree GNN**: features are concatenated to the root node's embedding
  (128-dim of the initial 256-dim node embedding comes from this)

## Verification
- Feature vector shape: always exactly (688,) — never varies with tree size
- All values in [0, 1] after normalization
- No NaN or inf in any feature
- Feature extraction is fast: < 1ms per tree (bottleneck is training, not features)
- Sanity: sin(x) and cos(x) have similar features (both unary trig at depth 0)
- Sanity: x^2 and x^10 have same depth features (differ only in constant)
