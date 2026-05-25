# Phase 5: CAS Verification + Special Functions (Python)

## Goal
Validate all generated pairs using SymPy as ground truth.
Filter out invalid, degenerate, or numerically unstable pairs.
Compute definite integral values for Mode 3 pairs.

## Why This Phase Exists
Rust's differentiator (Phase 3) does basic simplification but not full
algebraic simplification. Some generated pairs may have:
- Equivalent but visually different forms (x+x vs 2x)
- Expressions that simplify to trivial cases (0, constant)
- Division by zero or undefined regions
- Definite integrals that diverge
SymPy catches all of these. This phase is the quality gate.

## File: `python/neurips/data/verify.py`

### Core Verification Function
```python
def verify_pair(pair: dict) -> dict | None:
    """Verify one (f, F) pair. Return enriched pair or None if invalid."""
```

Steps:
1. Parse integrand f and antiderivative F from prefix notation to SymPy
2. Compute `sympy.diff(F, x)` (or partial diff for multivariate)
3. Check `sympy.simplify(diff_F - f) == 0`
   - If yes: pair is valid
   - If no: try `sympy.trigsimp`, `sympy.expand` (some need extra simplification)
   - If still no: discard the pair
4. Check for degenerate cases:
   - f == 0 (trivial integral) → discard
   - F is a constant (trivial) → discard
   - f or F contains zoo, nan, oo → discard
5. For definite integrals (Mode 3):
   - Evaluate `F(b) - F(a)` symbolically via SymPy
   - If symbolic eval fails, try `mpmath.quad(f, a, b, maxdegree=10)`
   - Require agreement to 1e-12 between symbolic and numeric
   - If integral diverges → discard
6. Compute difficulty metadata:
   - `difficulty_tier`: easy/medium/hard/very_hard based on depth + nodes
   - `has_special_fn`: does F contain erf, Bessel, etc.
   - `num_vars`: count of free variables
   - `num_params`: count of symbolic parameters

### Batch Verification
```python
def verify_batch(pairs: list[dict], n_workers: int = 8) -> list[dict]:
    """Verify pairs in parallel using multiprocessing.
    Each worker gets a 4-second timeout per pair (matches CAS stage).
    Returns only valid, enriched pairs."""
```

Use `multiprocessing.Pool` with `imap_unordered` for progress tracking.
Timeout: 4 seconds per pair (same as the CAS stage in the pipeline).
Pairs that timeout are discarded (if CAS can't verify in 4s, it's too hard
for the CAS stage anyway).

### Special Function Verification
For pairs with special functions in F:
```python
# SymPy can differentiate all standard special functions
sympy.diff(sympy.erf(x), x)        # → 2*exp(-x**2)/sqrt(pi)
sympy.diff(sympy.besselj(n, x), x) # → (besselj(n-1,x) - besselj(n+1,x))/2
```
Verify these symbolically. If symbolic diff fails (rare edge cases),
fall back to numerical verification:
- Evaluate f and dF/dx at 20 random points
- Check agreement to 1e-10 at each point

### Output
Write verified pairs to `data/verified/{mode}.jsonl`:
- One JSON object per line
- Add verification metadata: `verified: true`, `verify_method: symbolic|numeric`
- Add difficulty tier and feature flags

## Expected Yield
| Mode | Generated | After Verification | Yield |
|------|-----------|-------------------|-------|
| Univariate | 500K | ~450K | 90% |
| Multivariate | 300K | ~255K | 85% |
| Definite | 200K | ~160K | 80% |
| Parametric | 200K | ~170K | 85% |
| Special fn | 300K | ~240K | 80% |
| **Total** | **1.5M** | **~1.275M** | **85%** |

Lower yield for definite (divergent integrals) and special fn (CAS timeout).

## Verification
- Spot-check 100 verified pairs manually in Mathematica/WolframAlpha
- Zero false positives: every verified pair must be correct
- False negative rate < 5%: most valid pairs should pass verification
- Benchmark: full 1.5M verification in < 2 hours (8 workers, 4s timeout)
