# Phase 13a: Verification Oracle — Core

## Goal
Build the verification system that checks whether a candidate antiderivative
is correct. This is the most important component: it makes the entire
sampling strategy work.

## The Key Insight (Simply)
Finding an antiderivative is HARD (the whole point of the model).
Checking an antiderivative is EASY (just differentiate and compare).

If someone claims ∫ 2x dx = x², you check: d/dx(x²) = 2x. Correct.
If someone claims ∫ 2x dx = x³, you check: d/dx(x³) = 3x² ≠ 2x. Wrong.

The checker is essentially free — SymPy can differentiate anything instantly.
This asymmetry (hard to find, easy to check) is what makes sampling work.

## File: `python/neurips/evaluation/oracle.py`

### Main Entry Point
```python
class VerificationOracle:
    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout  # max seconds per verification
    
    def verify(self, candidate, integrand, task_info) -> bool:
        """Is this candidate a correct answer to the integral?"""
        task = task_info["task"]
        var = task_info["var"]
        
        if task == "INDEF":
            return self._verify_indefinite(candidate, integrand, var)
        elif task == "DEF":
            return self._verify_definite(candidate, integrand, task_info["bounds"])
```

### Indefinite Integral Verification
```python
def _verify_indefinite(self, F, f, var):
    """Check: d/dx F(x) == f(x)
    
    Three strategies, tried in order:
    1. Symbolic: differentiate and simplify → exact check
    2. Expanded symbolic: try harder simplification tricks
    3. Numerical: evaluate at random points → approximate check
    """
    F_sym = to_sympy(F)
    f_sym = to_sympy(f)
    x = sympy.Symbol(var)
    
    # Strategy 1: Direct symbolic check
    dF = sympy.diff(F_sym, x)
    diff = sympy.simplify(dF - f_sym)
    if diff == 0:
        return True
    
    # Strategy 3: Numerical check at 20 random points
    # If dF/dx and f agree to 1e-10 everywhere, probably correct
    test_points = [random.uniform(-5, 5) for _ in range(20)]
    agreements = 0
    for pt in test_points:
        try:
            val_dF = complex(dF.subs(x, pt))
            val_f = complex(f_sym.subs(x, pt))
            if abs(val_dF - val_f) < 1e-8 * max(1, abs(val_f)):
                agreements += 1
        except (ValueError, ZeroDivisionError, OverflowError):
            continue  # skip points where evaluation fails
    
    # Accept if 18+ out of 20 points agree (allow 2 for numerical noise)
    return agreements >= 18
```

### Definite Integral Verification
```python
def _verify_definite(self, predicted_value, f, bounds):
    """Check: predicted value ≈ ∫_a^b f(x) dx
    
    Uses high-precision numerical integration (mpmath) to compute
    the "true" value, then compares.
    """
    f_sym = to_sympy(f)
    a, b = bounds
    
    import mpmath
    mpmath.mp.dps = 50  # 50 digits of precision
    
    # Numerically integrate f from a to b
    f_func = sympy.lambdify("x", f_sym, modules=["mpmath"])
    try:
        numerical = mpmath.quad(f_func, [float(a), float(b)])
    except Exception:
        return False
    
    # Compare to predicted value
    pred_sym = to_sympy(predicted_value)
    try:
        predicted = mpmath.mpf(str(pred_sym.evalf(50)))
    except Exception:
        return False
