use crate::expr::{BinaryOp, ExprNode, MathConst, UnaryOp, VarId};

/// Differentiate `expr` with respect to `var` (fused diff+simplify).
pub fn differentiate(expr: &ExprNode, var: &str) -> ExprNode {
    let vid = VarId::from_str(var).unwrap_or_else(|| panic!("unknown var: {var}"));
    diff(expr, vid)
}

// ---------------------------------------------------------------------------
// Smart constructors — simplify at construction time
// ---------------------------------------------------------------------------

#[inline]
fn smart_add(a: ExprNode, b: ExprNode) -> ExprNode {
    if a.is_zero() { return b; }
    if b.is_zero() { return a; }
    if let (ExprNode::Num(x), ExprNode::Num(y)) = (&a, &b) {
        if let Some(v) = x.checked_add(*y) {
            return ExprNode::Num(v);
        }
    }
    ExprNode::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
}

#[inline]
fn smart_sub(a: ExprNode, b: ExprNode) -> ExprNode {
    if b.is_zero() { return a; }
    if a.is_zero() { return smart_neg(b); }
    if let (ExprNode::Num(x), ExprNode::Num(y)) = (&a, &b) {
        if let Some(v) = x.checked_sub(*y) {
            return ExprNode::Num(v);
        }
    }
    ExprNode::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
}

#[inline]
fn smart_mul(a: ExprNode, b: ExprNode) -> ExprNode {
    if a.is_zero() || b.is_zero() { return ExprNode::num(0); }
    if a.is_one() { return b; }
    if b.is_one() { return a; }
    if let ExprNode::Num(-1) = &a { return smart_neg(b); }
    if let ExprNode::Num(-1) = &b { return smart_neg(a); }
    if let (ExprNode::Num(x), ExprNode::Num(y)) = (&a, &b) {
        if let Some(v) = x.checked_mul(*y) {
            return ExprNode::Num(v);
        }
    }
    ExprNode::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
}

#[inline]
fn smart_div(a: ExprNode, b: ExprNode) -> ExprNode {
    if a.is_zero() { return ExprNode::num(0); }
    if b.is_one() { return a; }
    if let (ExprNode::Num(x), ExprNode::Num(y)) = (&a, &b) {
        if *y != 0 && x % y == 0 {
            return ExprNode::Num(x / y);
        }
    }
    ExprNode::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
}

#[inline]
fn smart_pow(base: ExprNode, exp: ExprNode) -> ExprNode {
    if exp.is_zero() { return ExprNode::num(1); }
    if exp.is_one() { return base; }
    if let (ExprNode::Num(x), ExprNode::Num(y)) = (&base, &exp) {
        if *y >= 0 && *y <= 10 {
            if let Some(v) = x.checked_pow(*y as u32) {
                return ExprNode::Num(v);
            }
        }
    }
    ExprNode::Binary(BinaryOp::Pow, Box::new(base), Box::new(exp))
}

#[inline]
fn smart_neg(a: ExprNode) -> ExprNode {
    if a.is_zero() { return ExprNode::num(0); }
    if let ExprNode::Unary(UnaryOp::Neg, inner) = a {
        return *inner;
    }
    if let ExprNode::Num(n) = &a {
        if let Some(v) = n.checked_neg() {
            return ExprNode::Num(v);
        }
    }
    ExprNode::Unary(UnaryOp::Neg, Box::new(a))
}

// ---------------------------------------------------------------------------
// Fused differentiation (simplifies as it builds)
// ---------------------------------------------------------------------------

fn diff(expr: &ExprNode, var: VarId) -> ExprNode {
    match expr {
        // Leaves
        ExprNode::Var(v) if *v == var => ExprNode::num(1),
        ExprNode::Var(_)
        | ExprNode::Param(_)
        | ExprNode::Num(_)
        | ExprNode::Rational(_, _)
        | ExprNode::Const(_) => ExprNode::num(0),

        // Unary: chain rule  d/dx f(g(x)) = f'(g(x)) * g'(x)
        ExprNode::Unary(op, inner) => {
            let g = inner.as_ref();
            let dg = diff(g, var);
            let fprime_g = diff_unary(op, g);
            smart_mul(fprime_g, dg)
        }

        // Binary
        ExprNode::Binary(op, l, r) => diff_binary(op, l, r, var),

        // Hyp2F1 — not differentiable in this engine
        ExprNode::Hyp2F1(..) => ExprNode::num(0),
    }
}

/// Derivative of unary op applied to `g`: returns f'(g).
#[inline]
fn diff_unary(op: &UnaryOp, g: &ExprNode) -> ExprNode {
    match op {
        UnaryOp::Neg => ExprNode::num(-1),
        UnaryOp::Sin => ExprNode::cos(g.clone()),
        UnaryOp::Cos => smart_neg(ExprNode::sin(g.clone())),
        UnaryOp::Tan => {
            smart_add(
                ExprNode::num(1),
                smart_pow(
                    ExprNode::Unary(UnaryOp::Tan, Box::new(g.clone())),
                    ExprNode::num(2),
                ),
            )
        }
        UnaryOp::Exp => ExprNode::exp(g.clone()),
        UnaryOp::Log => smart_div(ExprNode::num(1), g.clone()),
        UnaryOp::Sqrt => {
            smart_div(
                ExprNode::num(1),
                smart_mul(ExprNode::num(2), ExprNode::sqrt(g.clone())),
            )
        }
        UnaryOp::Asin => {
            smart_div(
                ExprNode::num(1),
                ExprNode::sqrt(smart_sub(
                    ExprNode::num(1),
                    smart_pow(g.clone(), ExprNode::num(2)),
                )),
            )
        }
        UnaryOp::Acos => {
            smart_neg(smart_div(
                ExprNode::num(1),
                ExprNode::sqrt(smart_sub(
                    ExprNode::num(1),
                    smart_pow(g.clone(), ExprNode::num(2)),
                )),
            ))
        }
        UnaryOp::Atan => {
            smart_div(
                ExprNode::num(1),
                smart_add(
                    ExprNode::num(1),
                    smart_pow(g.clone(), ExprNode::num(2)),
                ),
            )
        }
        UnaryOp::Sinh => ExprNode::Unary(UnaryOp::Cosh, Box::new(g.clone())),
        UnaryOp::Cosh => ExprNode::Unary(UnaryOp::Sinh, Box::new(g.clone())),
        UnaryOp::Tanh => {
            smart_sub(
                ExprNode::num(1),
                smart_pow(
                    ExprNode::Unary(UnaryOp::Tanh, Box::new(g.clone())),
                    ExprNode::num(2),
                ),
            )
        }
        UnaryOp::Erf => {
            smart_mul(
                smart_div(
                    ExprNode::num(2),
                    ExprNode::sqrt(ExprNode::Const(MathConst::Pi)),
                ),
                ExprNode::exp(smart_neg(
                    smart_pow(g.clone(), ExprNode::num(2)),
                )),
            )
        }
        // Remaining special functions: preserve unevaluated
        _ => ExprNode::Unary(*op, Box::new(g.clone())),
    }
}

/// Differentiate binary operations.
#[inline]
fn diff_binary(op: &BinaryOp, l: &ExprNode, r: &ExprNode, var: VarId) -> ExprNode {
    let dl = diff(l, var);
    let dr = diff(r, var);
    match op {
        BinaryOp::Add => smart_add(dl, dr),
        BinaryOp::Sub => smart_sub(dl, dr),
        BinaryOp::Mul => {
            // product rule: l'*r + l*r'
            smart_add(
                smart_mul(dl, r.clone()),
                smart_mul(l.clone(), dr),
            )
        }
        BinaryOp::Div => {
            // quotient rule: (l'*r - l*r') / r^2
            smart_div(
                smart_sub(
                    smart_mul(dl, r.clone()),
                    smart_mul(l.clone(), dr),
                ),
                smart_pow(r.clone(), ExprNode::num(2)),
            )
        }
        BinaryOp::Pow => {
            let l_has = l.contains_var(var);
            let r_has = r.contains_var(var);
            if !r_has {
                // f(x)^c => c * f^(c-1) * f'
                smart_mul(
                    smart_mul(
                        r.clone(),
                        smart_pow(l.clone(), smart_sub(r.clone(), ExprNode::num(1))),
                    ),
                    dl,
                )
            } else if !l_has {
                // c^g(x) => c^g * ln(c) * g'
                smart_mul(
                    smart_mul(
                        smart_pow(l.clone(), r.clone()),
                        ExprNode::log(l.clone()),
                    ),
                    dr,
                )
            } else {
                // f^g => f^g * (g'*ln(f) + g*f'/f)
                smart_mul(
                    smart_pow(l.clone(), r.clone()),
                    smart_add(
                        smart_mul(dr, ExprNode::log(l.clone())),
                        smart_mul(r.clone(), smart_div(dl, l.clone())),
                    ),
                )
            }
        }
        BinaryOp::BesselJ => {
            // d/dx J_n(f) = (J_{n-1}(f) - J_{n+1}(f)) / 2 * f'
            smart_mul(
                smart_div(
                    smart_sub(
                        ExprNode::Binary(
                            BinaryOp::BesselJ,
                            Box::new(smart_sub(l.clone(), ExprNode::num(1))),
                            Box::new(r.clone()),
                        ),
                        ExprNode::Binary(
                            BinaryOp::BesselJ,
                            Box::new(smart_add(l.clone(), ExprNode::num(1))),
                            Box::new(r.clone()),
                        ),
                    ),
                    ExprNode::num(2),
                ),
                dr,
            )
        }
        // Fallback: preserve unevaluated
        _ => ExprNode::Binary(*op, Box::new(l.clone()), Box::new(r.clone())),
    }
}

// ---------------------------------------------------------------------------
// PyO3: batch differentiation via rayon
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::py_batch_differentiate;

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use rayon::prelude::*;
    use super::differentiate;
    use crate::expr::PyExprTree;

    #[pyfunction]
    #[pyo3(name = "batch_differentiate")]
    pub fn py_batch_differentiate(
        trees: Vec<PyExprTree>,
        var: &str,
    ) -> Vec<PyExprTree> {
        let var = var.to_string();
        trees
            .par_iter()
            .map(|t| PyExprTree {
                inner: differentiate(&t.inner, &var),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_constant() {
        let e = ExprNode::num(5);
        assert_eq!(differentiate(&e, "x"), ExprNode::num(0));
    }

    #[test]
    fn diff_var() {
        let e = ExprNode::var("x");
        assert_eq!(differentiate(&e, "x"), ExprNode::num(1));
        assert_eq!(differentiate(&e, "y"), ExprNode::num(0));
    }

    #[test]
    fn diff_add() {
        // d/dx (x + 3) = 1
        let e = ExprNode::add(ExprNode::var("x"), ExprNode::num(3));
        assert_eq!(differentiate(&e, "x"), ExprNode::num(1));
    }

    #[test]
    fn diff_mul() {
        // d/dx (x * x) = x + x (after simplification)
        let e = ExprNode::mul(ExprNode::var("x"), ExprNode::var("x"));
        let d = differentiate(&e, "x");
        assert_eq!(
            d,
            ExprNode::add(ExprNode::var("x"), ExprNode::var("x"))
        );
    }

    #[test]
    fn diff_sin() {
        // d/dx sin(x) = cos(x)
        let e = ExprNode::sin(ExprNode::var("x"));
        let d = differentiate(&e, "x");
        assert_eq!(d, ExprNode::cos(ExprNode::var("x")));
    }

    #[test]
    fn diff_exp() {
        // d/dx exp(x) = exp(x)
        let e = ExprNode::exp(ExprNode::var("x"));
        let d = differentiate(&e, "x");
        assert_eq!(d, ExprNode::exp(ExprNode::var("x")));
    }

    #[test]
    fn diff_power_const_exp() {
        // d/dx x^3 = 3 * x^2
        let e = ExprNode::pow(ExprNode::var("x"), ExprNode::num(3));
        let d = differentiate(&e, "x");
        assert_eq!(
            d,
            ExprNode::mul(
                ExprNode::num(3),
                ExprNode::pow(ExprNode::var("x"), ExprNode::num(2)),
            )
        );
    }

    #[test]
    fn constant_folding() {
        // d/dx (3 + 4 + x) = 1
        let e = ExprNode::add(
            ExprNode::add(ExprNode::num(3), ExprNode::num(4)),
            ExprNode::var("x"),
        );
        let d = differentiate(&e, "x");
        assert_eq!(d, ExprNode::num(1));
    }

    #[test]
    fn diff_nested_deep() {
        // d/dx sin(cos(exp(x))) should not blow up
        let e = ExprNode::sin(ExprNode::cos(ExprNode::exp(ExprNode::var("x"))));
        let d = differentiate(&e, "x");
        // Result should have reasonable size (fused prevents blowup)
        assert!(d.node_count() < 30);
    }
}
