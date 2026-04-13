"""Expression rewriting augmentation for training data diversity.

Applies mathematically-equivalent rewrites to integrand expressions:
  - Commutativity: randomly permute Add/Mul argument order
  - Expand/factor: randomly apply sp.expand() or sp.factor()
  - Distribution: a*(b+c) ↔ a*b + a*c
  - Trig identities: random bidirectional trig rewrites

Each variant integrates to the same F(x) (the target is unchanged).
"""
from __future__ import annotations

import random
from typing import List

import sympy as sp


def _permute_commutative_args(expr: sp.Basic, rng: random.Random) -> sp.Basic:
    """Randomly permute arguments of commutative operations (Add, Mul)."""
    if isinstance(expr, (sp.Add, sp.Mul)):
        args = list(expr.args)
        rng.shuffle(args)
        new_args = [_permute_commutative_args(a, rng) for a in args]
        return type(expr)(*new_args, evaluate=False)

    if hasattr(expr, "args") and expr.args:
        new_args = tuple(_permute_commutative_args(a, rng) for a in expr.args)
        return expr.func(*new_args)

    return expr


def _random_expand_or_factor(expr: sp.Basic, rng: random.Random) -> sp.Basic:
    """Randomly apply expand or factor to the expression or a subexpression."""
    choice = rng.random()
    try:
        if choice < 0.4:
            return sp.expand(expr)
        if choice < 0.7:
            return sp.factor(expr)
        # Apply expand to a random subexpression
        if hasattr(expr, "args") and len(expr.args) >= 2:
            idx = rng.randrange(len(expr.args))
            new_args = list(expr.args)
            new_args[idx] = sp.expand(new_args[idx])
            return expr.func(*new_args)
    except Exception:
        pass
    return expr


def _trig_rewrite(expr: sp.Basic, rng: random.Random) -> sp.Basic:
    """Apply random trig identity rewrites (staying in trig domain)."""
    try:
        choice = rng.random()
        if choice < 0.33:
            return sp.trigsimp(expr)
        if choice < 0.66:
            return sp.expand_trig(expr)
        return expr.rewrite(sp.sin, sp.cos)
    except Exception:
        return expr


def augment_expression(
    expr: sp.Basic,
    rng: random.Random,
    n_variants: int = 3,
) -> List[sp.Basic]:
    """Generate n_variants rewritten forms of expr.

    All returned expressions are mathematically equivalent to expr.
    Returns a list of up to n_variants distinct expressions (may be fewer
    if rewrites produce duplicates).
    """
    seen = {sp.srepr(expr)}
    variants: List[sp.Basic] = []

    strategies = [
        lambda e: _permute_commutative_args(e, rng),
        lambda e: _random_expand_or_factor(e, rng),
        lambda e: _trig_rewrite(e, rng),
        lambda e: _random_expand_or_factor(_permute_commutative_args(e, rng), rng),
        lambda e: _trig_rewrite(_permute_commutative_args(e, rng), rng),
    ]

    attempts = 0
    max_attempts = n_variants * 5

    while len(variants) < n_variants and attempts < max_attempts:
        attempts += 1
        strategy = rng.choice(strategies)
        try:
            rewritten = strategy(expr)
            # Simplify to canonical for dedup check, but keep the rewritten form
            key = sp.srepr(rewritten)
            if key not in seen:
                seen.add(key)
                variants.append(rewritten)
        except Exception:
            continue

    return variants
