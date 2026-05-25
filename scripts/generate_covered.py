"""Coverage-guaranteed data generation.

Generates training data using skeleton enumeration to ensure every
structural family of integration-relevant expressions is represented.
Falls back to pure-Python implementation when Rust FFI is unavailable.

Usage:
    python scripts/generate_covered.py --total 1500000 --output data/covered/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import sympy as sp
from sympy import (
    Symbol, acos, asin, atan, cos, cosh, exp, log, sin, sinh, sqrt, tan, tanh,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

x = Symbol("x")

SKELETON_FAMILIES: dict[str, list[callable]] = {}


def _register(family: str):
    def decorator(fn):
        SKELETON_FAMILIES.setdefault(family, []).append(fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Skeleton definitions — each returns a SymPy antiderivative F(x).
# The integrand is d/dx F(x), computed symbolically.
# ---------------------------------------------------------------------------

# 1. Elementary functions
@_register("elementary")
def _elem_sin(): return sin(x)
@_register("elementary")
def _elem_cos(): return cos(x)
@_register("elementary")
def _elem_tan(): return -log(cos(x))
@_register("elementary")
def _elem_exp(): return exp(x)
@_register("elementary")
def _elem_log(): return x * log(x) - x
@_register("elementary")
def _elem_sqrt(): return sp.Rational(2, 3) * x**sp.Rational(3, 2)
@_register("elementary")
def _elem_sinh(): return cosh(x)
@_register("elementary")
def _elem_cosh(): return sinh(x)
@_register("elementary")
def _elem_tanh(): return log(cosh(x))
@_register("elementary")
def _elem_asin(): return x * asin(x) + sqrt(1 - x**2)
@_register("elementary")
def _elem_acos(): return x * acos(x) - sqrt(1 - x**2)
@_register("elementary")
def _elem_atan(): return x * atan(x) - log(1 + x**2) / 2

# 2. Linear substitution: f(ax+b)
@_register("linear_sub")
def _linsub_sin():
    a = random.choice([-5, -3, -2, -1, 2, 3, 5])
    b = random.randint(-5, 5)
    return -cos(a * x + b) / a

@_register("linear_sub")
def _linsub_exp():
    a = random.choice([-3, -2, -1, 2, 3])
    return exp(a * x) / a

@_register("linear_sub")
def _linsub_log():
    a = random.choice([2, 3, 5])
    b = random.randint(1, 5)
    return ((a * x + b) * log(a * x + b) - (a * x + b)) / a

@_register("linear_sub")
def _linsub_cos():
    a = random.choice([-4, -2, 2, 3, 4])
    b = random.randint(-3, 3)
    return sin(a * x + b) / a

@_register("linear_sub")
def _linsub_sqrt():
    a = random.choice([2, 3, 4])
    b = random.randint(0, 5)
    return sp.Rational(2, 3) * (a * x + b)**sp.Rational(3, 2) / a

# 3. Compositions: f(g(x))
@_register("composition")
def _comp_sin_exp(): return sin(exp(x))
@_register("composition")
def _comp_exp_sin(): return exp(sin(x))
@_register("composition")
def _comp_log_sin(): return log(sin(x))
@_register("composition")
def _comp_log_cos(): return log(cos(x))
@_register("composition")
def _comp_sqrt_exp(): return sqrt(exp(x))
@_register("composition")
def _comp_exp_log(): return x  # e^(ln(x)) = x
@_register("composition")
def _comp_sin_log(): return sin(log(x))
@_register("composition")
def _comp_cos_sqrt(): return cos(sqrt(x))

# 4. U-substitution: ∫f(g(x))·g'(x)dx = F(g(x))
@_register("u_sub")
def _usub_sin_x2(): return -cos(x**2) / 2
@_register("u_sub")
def _usub_exp_x2(): return exp(x**2) / 2
@_register("u_sub")
def _usub_log_sinx(): return log(sin(x))
@_register("u_sub")
def _usub_sqrt_1mx2(): return sqrt(1 - x**2)
@_register("u_sub")
def _usub_exp_sinx(): return exp(sin(x))
@_register("u_sub")
def _usub_sin_expx(): return sin(exp(x))
@_register("u_sub")
def _usub_log_x2p1(): return log(x**2 + 1) / 2
@_register("u_sub")
def _usub_atan_expx(): return atan(exp(x))

# 5. Integration by parts: x^n * f(x)
@_register("ibp_poly_exp")
def _ibp_x_exp(): return x * exp(x) - exp(x)
@_register("ibp_poly_exp")
def _ibp_x2_exp(): return (x**2 - 2*x + 2) * exp(x)
@_register("ibp_poly_exp")
def _ibp_x3_exp(): return (x**3 - 3*x**2 + 6*x - 6) * exp(x)

@_register("ibp_poly_trig")
def _ibp_x_sinx(): return sin(x) - x * cos(x)
@_register("ibp_poly_trig")
def _ibp_x_cosx(): return cos(x) + x * sin(x)
@_register("ibp_poly_trig")
def _ibp_x2_sinx(): return 2*x*sin(x) - (x**2 - 2)*cos(x)
@_register("ibp_poly_trig")
def _ibp_x2_cosx(): return 2*x*cos(x) + (x**2 - 2)*sin(x)

@_register("ibp_poly_log")
def _ibp_x_logx(): return x**2 * log(x) / 2 - x**2 / 4
@_register("ibp_poly_log")
def _ibp_x2_logx(): return x**3 * log(x) / 3 - x**3 / 9
@_register("ibp_poly_log")
def _ibp_logx(): return x * log(x) - x

@_register("ibp_exp_trig")
def _ibp_exp_sinx(): return exp(x) * (sin(x) - cos(x)) / 2
@_register("ibp_exp_trig")
def _ibp_exp_cosx(): return exp(x) * (sin(x) + cos(x)) / 2

@_register("ibp_inverse_trig")
def _ibp_arcsinx(): return x * asin(x) + sqrt(1 - x**2)
@_register("ibp_inverse_trig")
def _ibp_arctanx(): return x * atan(x) - log(1 + x**2) / 2

# 6. Rational functions (partial fractions)
@_register("rational")
def _rat_1_over_xpa():
    a = random.choice([1, 2, 3, -1, -2])
    return log(x + a)

@_register("rational")
def _rat_1_over_x2pa2():
    a = random.choice([1, 2, 3])
    return atan(x / a) / a

@_register("rational")
def _rat_partial_simple():
    a, b = 1, 2
    return log(x + a) - log(x + b)

@_register("rational")
def _rat_x_over_x2p1(): return log(x**2 + 1) / 2

@_register("rational")
def _rat_1_over_x2m1(): return log(abs(x - 1)) / 2 - log(abs(x + 1)) / 2

@_register("rational")
def _rat_quadratic():
    return x - 2 * log(x + 1) + log(x + 2)

# 7. Trig powers: sin^m(x) * cos^n(x)
@_register("trig_power")
def _trig_sin2(): return x / 2 - sin(2*x) / 4
@_register("trig_power")
def _trig_cos2(): return x / 2 + sin(2*x) / 4
@_register("trig_power")
def _trig_sin3(): return -cos(x) + cos(x)**3 / 3
@_register("trig_power")
def _trig_sin2_cos(): return sin(x)**3 / 3
@_register("trig_power")
def _trig_sin_cos2(): return -cos(x)**3 / 3
@_register("trig_power")
def _trig_sin4(): return sp.Rational(3, 8)*x - sin(2*x)/4 + sin(4*x)/32

# 8. Trig substitution
@_register("trig_sub")
def _trigsub_sqrt_a2mx2():
    a = random.choice([1, 2, 3])
    return (x * sqrt(a**2 - x**2) + a**2 * asin(x / a)) / 2

@_register("trig_sub")
def _trigsub_sqrt_x2pa2():
    a = random.choice([1, 2])
    return (x * sqrt(x**2 + a**2) + a**2 * log(x + sqrt(x**2 + a**2))) / 2

@_register("trig_sub")
def _trigsub_1_over_sqrt(): return asin(x)

# 9. Power rule
@_register("power_rule")
def _pow_n():
    n = random.choice([-5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8])
    return x**(n + 1) / (n + 1)

@_register("power_rule")
def _pow_half(): return 2 * sqrt(x)
@_register("power_rule")
def _pow_neg1(): return log(x)  # special case x^(-1)

# 10. Exponential chains
@_register("exp_chain")
def _exp_sin(): return exp(sin(x))
@_register("exp_chain")
def _exp_cos(): return exp(cos(x))
@_register("exp_chain")
def _exp_x2(): return exp(x**2)
@_register("exp_chain")
def _exp_sqrt(): return exp(sqrt(x))

# 11. Hyperbolic
@_register("hyperbolic")
def _hyp_sinh(): return cosh(x)
@_register("hyperbolic")
def _hyp_cosh(): return sinh(x)
@_register("hyperbolic")
def _hyp_tanh(): return log(cosh(x))
@_register("hyperbolic")
def _hyp_sinh_comp():
    a = random.choice([2, 3])
    return cosh(a * x) / a

# 12. Triple compositions
@_register("triple_comp")
def _triple_sin_exp_x(): return sin(exp(exp(x)))
@_register("triple_comp")
def _triple_exp_sin_x(): return exp(sin(log(x)))
@_register("triple_comp")
def _triple_log_sin_x(): return log(sin(exp(x)))

# 13. Polynomial products: x^n * f(x)
@_register("poly_product")
def _pp_x_sin(): return sin(x) - x * cos(x)
@_register("poly_product")
def _pp_x2_exp():  return (x**2 - 2*x + 2) * exp(x)
@_register("poly_product")
def _pp_x_sqrt(): return sp.Rational(2, 5) * x**sp.Rational(5, 2)


def generate_from_skeleton(family: str) -> tuple[sp.Expr, sp.Expr] | None:
    """Generate an (integrand, antiderivative) pair from a skeleton family."""
    generators = SKELETON_FAMILIES.get(family, [])
    if not generators:
        return None

    gen_fn = random.choice(generators)
    try:
        antideriv = gen_fn()
        integrand = sp.diff(antideriv, x)
        if integrand == 0 or integrand.has(sp.zoo, sp.nan, sp.oo):
            return None
        return integrand, antideriv
    except Exception:
        return None


def generate_covered_dataset(
    total: int,
    output_dir: Path,
    verify: bool = True,
) -> dict:
    """Generate a dataset with guaranteed coverage of all skeleton families.

    Distribution:
      - 70% from skeleton enumeration (uniform across families)
      - 30% from random SymPy expressions (exploration)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    families = list(SKELETON_FAMILIES.keys())
    n_families = len(families)
    logger.info("Skeleton families: %d", n_families)

    covered_budget = int(total * 0.70)
    random_budget = total - covered_budget
    per_family = max(covered_budget // n_families, 1)

    logger.info(
        "Budget: %d covered (%d/family × %d families) + %d random",
        covered_budget, per_family, n_families, random_budget,
    )

    pairs: list[dict] = []
    family_counts: Counter = Counter()
    failures = 0

    # Phase 1: Skeleton-based generation
    for family in families:
        generated = 0
        attempts = 0
        while generated < per_family and attempts < per_family * 5:
            attempts += 1
            result = generate_from_skeleton(family)
            if result is None:
                failures += 1
                continue
            integrand, antideriv = result
            pairs.append({
                "integrand": str(integrand),
                "antiderivative": str(antideriv),
                "family": family,
                "source": "skeleton",
            })
            generated += 1
            family_counts[family] += 1

    logger.info(
        "Phase 1 done: %d pairs from %d families (%d failures)",
        len(pairs), n_families, failures,
    )

    # Phase 2: Random exploration (random polynomial/trig/exp combinations)
    random_generated = 0
    random_attempts = 0
    while random_generated < random_budget and random_attempts < random_budget * 3:
        random_attempts += 1
        result = _random_antiderivative()
        if result is None:
            continue
        integrand, antideriv = result
        pairs.append({
            "integrand": str(integrand),
            "antiderivative": str(antideriv),
            "family": "random",
            "source": "exploration",
        })
        random_generated += 1
        family_counts["random"] += 1

    logger.info("Phase 2 done: %d random pairs", random_generated)
    logger.info("Total pairs: %d", len(pairs))

    # Save
    random.shuffle(pairs)
    out_path = output_dir / "covered_pairs.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    # Coverage report
    report = {
        "total_pairs": len(pairs),
        "n_families": n_families,
        "per_family_target": per_family,
        "family_counts": dict(family_counts),
        "failures": failures,
        "coverage_fraction": len([f for f in families if family_counts[f] > 0]) / n_families,
    }
    report_path = output_dir / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Coverage: %.1f%% of families represented", report["coverage_fraction"] * 100)
    logger.info("Saved to %s", out_path)
    return report


def _random_antiderivative() -> tuple[sp.Expr, sp.Expr] | None:
    """Generate a random antiderivative by composing operations."""
    try:
        depth = random.randint(1, 4)
        expr = x
        for _ in range(depth):
            op = random.choice([
                lambda e: sin(e),
                lambda e: cos(e),
                lambda e: exp(e),
                lambda e: log(abs(e) + 1),
                lambda e: e**random.randint(2, 4),
                lambda e: e * random.choice([2, 3, -1, -2]),
                lambda e: e + random.randint(-5, 5),
                lambda e: sqrt(abs(e) + 1),
                lambda e: e * sin(x),
                lambda e: e * exp(x),
            ])
            expr = op(expr)

        integrand = sp.diff(expr, x)
        if integrand == 0 or integrand.has(sp.zoo, sp.nan, sp.oo):
            return None
        if len(str(integrand)) > 500:
            return None
        return integrand, expr
    except Exception:
        return None


def _try_rust_generate(total: int, output_dir: Path) -> dict | None:
    """Try Rust FFI path — returns report dict or None if unavailable."""
    try:
        from neurips_core import GenConfig, generate_covered_batch
    except ImportError:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Using Rust FFI path (Rayon parallel differentiation)")

    config = GenConfig()
    pairs_raw = generate_covered_batch(total, config)

    pairs = []
    for integrand_tree, integral_tree in pairs_raw:
        pairs.append({
            "integrand": str(integrand_tree),
            "antiderivative": str(integral_tree),
            "family": "covered",
            "source": "rust_skeleton",
        })

    random.shuffle(pairs)
    out_path = output_dir / "covered_pairs.jsonl"
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    report = {
        "total_pairs": len(pairs),
        "backend": "rust",
    }
    report_path = output_dir / "coverage_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved %d pairs to %s", len(pairs), out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Coverage-guaranteed data generation")
    parser.add_argument("--total", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("data/covered"))
    parser.add_argument("--backend", choices=["auto", "rust", "python"], default="auto")
    args = parser.parse_args()

    t0 = time.time()

    report = None
    if args.backend in ("auto", "rust"):
        report = _try_rust_generate(args.total, args.output)
        if report is None and args.backend == "rust":
            logger.error("Rust backend requested but neurips_core not available")
            raise SystemExit(1)

    if report is None:
        logger.info("Using Python/SymPy fallback")
        report = generate_covered_dataset(args.total, args.output)

    elapsed = time.time() - t0
    backend = report.get("backend", "python")
    logger.info(
        "Completed in %.1fs (%.0f pairs/s) [%s]",
        elapsed, report["total_pairs"] / elapsed, backend,
    )


if __name__ == "__main__":
    main()
