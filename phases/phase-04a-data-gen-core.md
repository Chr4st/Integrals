# Phase 4a: Data Generation — Core Infrastructure (Rust)

## Goal
Build the random expression tree generator and batch generation pipeline.
This is the engine that produces 1.5M training pairs.

## How Backward Generation Works
You can't easily check if a random integral has a closed-form answer.
But you CAN always differentiate. So go backward:
1. Generate random antiderivative F
2. Compute f = dF/dx (using Phase 3 differentiator)
3. The pair (f, F) is guaranteed correct
The model trains on f as input and F as target.

## File: `rust/core/src/gen.rs`

### GenConfig (controls what kind of trees to generate)
```rust
pub struct GenConfig {
    pub max_depth: usize,       // deepest the tree can go (3-12)
    pub max_nodes: usize,       // cap on total nodes
    pub vars: Vec<VarName>,     // [x] for univariate, [x,y] for bivariate
    pub params: Vec<ParamName>, // [] or [a] or [a,b]
    pub allow_special_fns: bool,// include erf, Bessel, etc.
    pub num_range: (i64, i64),  // range for random constants (-10, 10)
}
```

### Random Tree Generator
`generate_random_tree(config: &GenConfig, rng: &mut impl Rng) -> ExprNode`

At each node, randomly pick what kind of node to create:
- 40% binary operator (add, mul, sub, div, pow)
- 25% unary function (sin, cos, exp, log, sqrt...)
- 5% special function (erf, BesselJ...) — only if allowed
- 30% leaf (variable, parameter, or integer)

Recurse until max_depth or max_nodes is hit.
If max_depth reached, force a leaf.

Weight distribution matters:
- Too many operators → trees explode exponentially
- Too many leaves → trees are trivially shallow
- These weights were tuned empirically

### Batch Generation with Rayon
```rust
pub fn generate_batch(
    n: usize,
    mode: GenMode,
    config: &GenConfig,
) -> Vec<Pair> {
    // Use rayon for parallel generation across CPU cores
    (0..n).into_par_iter().map(|_| {
        let mut rng = thread_rng();
        let f_tree = generate_random_tree(config, &mut rng);
        let integrand = differentiate(&f_tree, &config.vars[0]);
        Pair { integrand, antiderivative: f_tree, mode, ... }
    }).collect()
}
```
Target: 50K pairs/minute on 8 cores.

### Output Format
```json
{
  "integrand": "add mul x sin y pow x 2",
  "antiderivative": "add mul div pow x 2 2 neg cos y div pow x 3 3",
  "task": "INDEF",
  "var": "x",
  "free_vars": ["x", "y"],
  "params": [],
  "integrand_depth": 4,
  "integrand_nodes": 9
}
```

### PyO3 Bindings
- `generate_batch(n, mode, config) -> list[dict]`
- `GenConfig` exposed as Python class with builder pattern

## Verification
- Generate 1000 trees: all valid (no NaN, no div-by-zero)
- Tree depth distribution matches config (max_depth respected)
- Benchmark: 50K pairs/min on 8 cores
