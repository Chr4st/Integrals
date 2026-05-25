# Phase 3: Tree Operations (Rust)

## Goal
Implement symbolic differentiation, skeleton extraction, and tree comparison in Rust.
These are hot-path operations called millions of times during data generation and splitting.

## File: `rust/core/src/diff.rs` — Symbolic Differentiation

Implement `differentiate(expr: &ExprNode, var: VarName) -> ExprNode`.

This computes d/dx of any expression tree using standard rules:
- d/dx (constant) = 0
- d/dx (x) = 1, d/dx (y) = 0 (y is not the target variable)
- d/dx (f + g) = f' + g'            (sum rule)
- d/dx (f * g) = f'g + fg'          (product rule)
- d/dx (f / g) = (f'g - fg') / g^2  (quotient rule)
- d/dx (f^g) = f^g * (g' * ln(f) + g * f'/f)  (general power rule)
- d/dx sin(f) = cos(f) * f'         (chain rule for sin)
- d/dx cos(f) = -sin(f) * f'
- d/dx exp(f) = exp(f) * f'
- d/dx log(f) = f' / f
- d/dx erf(f) = 2/sqrt(pi) * exp(-f^2) * f'
- d/dx BesselJ(n, f) = (BesselJ(n-1, f) - BesselJ(n+1, f)) / 2 * f'
- (similar chain rule entries for all other special functions)

After differentiation, apply basic simplification:
- 0 + x → x, x + 0 → x
- 0 * x → 0, 1 * x → x
- x^0 → 1, x^1 → x
- Constant folding: 3 + 4 → 7

Do NOT implement full algebraic simplification (that's SymPy's job).
Just enough to keep trees from blowing up during backward generation.

## File: `rust/core/src/skeleton.rs` — Skeleton Extraction

Implement `skeleton(expr: &ExprNode) -> String`.

Replace all numeric leaves with a placeholder "C":
- `Num(_)` → "C"
- `Rational(_, _)` → "C"
- Everything else stays: `Var`, `Param`, operators, functions

Return a canonical string representation:
```
skeleton(parse("add mul 3 pow x 2 sin x"))
  → "(add (mul C (pow x C)) (sin x))"
```

Two expressions with the same skeleton differ only by constants.
This is used for train/test deduplication (Phase 6).

## File: `rust/core/src/compare.rs` — Tree Comparison

Implement `structural_eq(a: &ExprNode, b: &ExprNode) -> bool`.
Two trees are structurally equal if they have the same shape and same
operators/variables at every node (ignoring numeric values).
This is skeleton equality but comparing trees directly instead of strings.

Implement `exact_eq(a: &ExprNode, b: &ExprNode) -> bool`.
Same as structural_eq but also checks numeric values.

## PyO3 Bindings

Add to `PyExprTree`:
- `differentiate(var: str) -> PyExprTree`
- `skeleton() -> str`
- `structural_eq(other: PyExprTree) -> bool`

Add module-level function:
- `batch_differentiate(trees: list[PyExprTree], var: str) -> list[PyExprTree]`
  Uses rayon for parallel differentiation across many trees.

## Verification
- `d/dx (x^2) = 2x` (power rule)
- `d/dx (sin(x^2)) = 2x*cos(x^2)` (chain rule)
- `d/dy (x*sin(y)) = x*cos(y)` (partial derivative, x is constant)
- `d/dx (erf(x)) = 2/sqrt(pi) * exp(-x^2)` (special function)
- `skeleton("add mul 3 x mul 7 x") == skeleton("add mul 5 x mul 2 x")`
- Round-trip: `diff(F, x)` then verify via SymPy that result equals f
- Benchmark: differentiate 100K trees in < 5 seconds (Rust + rayon)
