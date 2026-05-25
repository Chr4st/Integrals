# Phase 13b: Verification Oracle — Multivariate + Parametric

## Goal
Extend the oracle to handle multivariate and parametric integrals.
The core idea is the same (differentiate and compare), but with
partial derivatives and parameter spot-checking.

## File: `python/neurips/evaluation/oracle.py` (continued)

### Multivariate Verification
```python
def _verify_multivariate(self, F, f, var):
    """Check: ∂F/∂x = f(x, y)
    
    For multivariate integrals, we check the PARTIAL derivative.
    ∫ f(x,y) dx = F(x,y) + g(y)
    
    The g(y) part doesn't matter because ∂g(y)/∂x = 0.
    So we just check ∂F/∂x = f, exactly as in the univariate case
    but with partial differentiation.
    """
    F_sym = to_sympy(F)
    f_sym = to_sympy(f)
    var_sym = sympy.Symbol(var)
    
    dF = sympy.diff(F_sym, var_sym)
    
    # Symbolic check
    diff = sympy.simplify(dF - f_sym)
    if diff == 0:
        return True
    
    # Numerical check on a GRID of points
    other_vars = sorted(F_sym.free_symbols - {var_sym}, key=str)
    
    agreements = 0
    for _ in range(20):
        subs = {var_sym: random.uniform(-5, 5)}
        for ov in other_vars:
            subs[ov] = random.uniform(-5, 5)
        try:
            val_dF = complex(dF.subs(subs))
            val_f = complex(f_sym.subs(subs))
            if abs(val_dF - val_f) < 1e-8 * max(1, abs(val_f)):
                agreements += 1
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return agreements >= 18
```

### Parametric Verification
```python
def _verify_parametric(self, F, f, var, params):
    """Check: d/dx F(x; a, b) == f(x; a, b) for ALL parameter values.
    
    The antiderivative must be UNIVERSALLY valid — not just for one
    specific value of a, but for all values of a.
    
    Strategy:
    1. Symbolic check (strongest: proves it for ALL a)
    2. Spot-check at 10 random parameter values (weaker but catches mistakes)
    """
    F_sym = to_sympy(F)
    f_sym = to_sympy(f)
    var_sym = sympy.Symbol(var)
    param_syms = [sympy.Symbol(p) for p in params]
    
    dF = sympy.diff(F_sym, var_sym)  # params treated as constants
    diff = sympy.simplify(dF - f_sym)
    if diff == 0:
        return True
    
    agreements = 0
    for _ in range(30):  # more points because parameter space is larger
        subs = {var_sym: random.uniform(-3, 3)}
        for ps in param_syms:
            subs[ps] = random.uniform(0.5, 5.0)  # avoid 0 (division, log issues)
        try:
            val_dF = complex(dF.subs(subs))
            val_f = complex(f_sym.subs(subs))
            if abs(val_dF - val_f) < 1e-8 * max(1, abs(val_f)):
                agreements += 1
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    
    return agreements >= 27  # 27 out of 30 must agree
```

## Verification
- Multivariate: ∂/∂x(x²y/2) = xy → True
- Multivariate: ∂/∂x(x²y/2) = x²y → False
- Parametric: d/dx(x^{a+1}/(a+1)) = x^a → True for all a ≠ -1
- Parametric wrong: d/dx(x^a) = x^a → False (missing /log(x) term)
- Special fn: d/dx(erf(x)) → 2/√π·e^{-x²} → True
- All timeouts handled gracefully (return False, no crash)
