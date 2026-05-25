# Phase 4b: Data Generation — Task-Specific Modes

## Goal
Implement the 5 generation modes, one per integral type.
Each mode configures the tree generator differently.

## File: `rust/core/src/gen.rs` (continued from Phase 4a)

### Mode 1 — Univariate Indefinite (the base case)
```
config = GenConfig { vars: [x], params: [], depth: 3-10, special_fns: false }
F = generate_random_tree(config)
f = differentiate(F, x)
pair = (f, F, task=INDEF, var=x)
```
This is the Lample & Charton setting. Single variable, elementary functions only.

### Mode 2 — Multivariate Indefinite (the novel contribution)
```
config = GenConfig { vars: [x, y], params: [], depth: 3-8 }
F = generate_random_tree(config)
f = differentiate(F, x)   // partial derivative w.r.t. x
pair = (f, F, task=INDEF, var=x)
```
The model must learn: ∫ f(x,y) dx = F(x,y) + g(y).
F has a +g(y) ambiguity — any function of y alone can be added.
The oracle (Phase 13) checks ∂F/∂x = f, which handles this.

Lower max_depth (8 vs 10) because bivariate trees are larger.

### Mode 3 — Definite Integrals
```
config = GenConfig { vars: [x], depth: 3-8 }
F = generate_random_tree(config)
f = differentiate(F, x)
bounds = random_choice([(0,1), (0,"pi"), (-1,1), (0,"inf"), ("-inf","inf")])
// Value computation deferred to Phase 5 (needs SymPy)
pair = (f, bounds, task=DEF, var=x, antiderivative=F)
```
Bounds are symbolic (pi, inf). The definite integral value F(b)-F(a)
is computed later in Python because it may need limit evaluation.

### Mode 4 — Parametric
```
config = GenConfig { vars: [x], params: [a], depth: 3-8 }
F = generate_random_tree(config)
f = differentiate(F, x)   // a treated as constant
pair = (f, F, task=INDEF, var=x, params=[a])
```
Parameters appear in the expression but are NOT differentiated.
The model must produce antiderivatives valid for ALL values of a.

### Mode 5 — Special Function Output
```
config = GenConfig { vars: [x], special_fns: true, depth: 3-6 }
F = generate_random_tree(config)
f = differentiate(F, x)   // chain rule through erf, Bessel, etc.
pair = (f, F, task=INDEF, var=x)
```
Lower max_depth (6) because special functions add complexity.
Differentiation rules for special functions are in Phase 3.

## Target Counts
| Mode | Count | Notes |
|------|-------|-------|
| Univariate | 500K | Largest set — base skill |
| Multivariate | 300K | Novel contribution |
| Definite | 200K | Values computed in Phase 5 |
| Parametric | 200K | 1-2 symbolic parameters |
| Special fn | 300K | erf, Bessel, elliptic in output |
| **Total** | **1.5M** | Before dedup and filtering |

## File: `scripts/generate_data.py`
CLI entry point:
```bash
python scripts/generate_data.py --mode all --output data/raw/
python scripts/generate_data.py --mode univariate --count 500000
python scripts/generate_data.py --mode multivariate --count 300000
```

## Verification
- 1000 pairs per mode: SymPy confirms diff(F, x) == f for each
- No NaN, no division by zero (filtered out during generation)
- Multivariate pairs: F contains both x and y in >80% of cases
- Parametric pairs: F contains parameter a in >90% of cases
- Special fn pairs: F contains at least one special function
- Full 1.5M generation completes in < 30 minutes
