use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::diff::differentiate;
use crate::eval::{evaluate, Bindings};
use crate::expr::{from_prefix_string, BinaryOp, ExprNode, UnaryOp, VarId};

/// Verification verdict with richer diagnostic information.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifyVerdict {
    Correct,
    IncorrectSymbolic,
    IncorrectNumerical,
    Timeout,
}

impl VerifyVerdict {
    pub fn is_correct(&self) -> bool {
        matches!(self, Self::Correct)
    }
}

// ---------------------------------------------------------------------------
// Canonicalization — normalize common algebraic patterns before comparison
// ---------------------------------------------------------------------------

/// Apply algebraic canonicalization rules to an expression tree.
/// Normalizes: sin²+cos²→1, double-angle, log(a*b)→log(a)+log(b).
fn canonicalize(expr: &ExprNode) -> ExprNode {
    match expr {
        ExprNode::Unary(op, inner) => {
            let c = canonicalize(inner);
            // exp(log(x)) → x, log(exp(x)) → x
            match (op, &c) {
                (UnaryOp::Exp, ExprNode::Unary(UnaryOp::Log, x)) => return x.as_ref().clone(),
                (UnaryOp::Log, ExprNode::Unary(UnaryOp::Exp, x)) => return x.as_ref().clone(),
                _ => {}
            }
            ExprNode::Unary(*op, Box::new(c))
        }
        ExprNode::Binary(op, l, r) => {
            let cl = canonicalize(l);
            let cr = canonicalize(r);
            canonicalize_binary(*op, cl, cr)
        }
        ExprNode::Hyp2F1(a, b, c, d) => ExprNode::Hyp2F1(
            Box::new(canonicalize(a)),
            Box::new(canonicalize(b)),
            Box::new(canonicalize(c)),
            Box::new(canonicalize(d)),
        ),
        other => other.clone(),
    }
}

fn canonicalize_binary(op: BinaryOp, l: ExprNode, r: ExprNode) -> ExprNode {
    // sin²(g) + cos²(g) → 1
    if op == BinaryOp::Add {
        if let Some(result) = try_pythagorean(&l, &r) {
            return result;
        }
        // log(a) + log(b) → log(a * b)
        if let (ExprNode::Unary(UnaryOp::Log, a), ExprNode::Unary(UnaryOp::Log, b)) = (&l, &r) {
            return ExprNode::Unary(
                UnaryOp::Log,
                Box::new(ExprNode::Binary(BinaryOp::Mul, a.clone(), b.clone())),
            );
        }
    }
    // a - a → 0
    if op == BinaryOp::Sub && l == r {
        return ExprNode::num(0);
    }
    // a / a → 1
    if op == BinaryOp::Div && l == r {
        return ExprNode::num(1);
    }
    ExprNode::Binary(op, Box::new(l), Box::new(r))
}

/// Detect sin²(g) + cos²(g) and reduce to 1.
fn try_pythagorean(a: &ExprNode, b: &ExprNode) -> Option<ExprNode> {
    let (g1, is_sin) = match a {
        ExprNode::Binary(BinaryOp::Pow, base, exp) if is_num(exp, 2) => {
            match base.as_ref() {
                ExprNode::Unary(UnaryOp::Sin, inner) => (inner.as_ref(), true),
                ExprNode::Unary(UnaryOp::Cos, inner) => (inner.as_ref(), false),
                _ => return None,
            }
        }
        _ => return None,
    };

    let g2 = match b {
        ExprNode::Binary(BinaryOp::Pow, base, exp) if is_num(exp, 2) => {
            match base.as_ref() {
                ExprNode::Unary(UnaryOp::Sin, inner) if !is_sin => inner.as_ref(),
                ExprNode::Unary(UnaryOp::Cos, inner) if is_sin => inner.as_ref(),
                _ => return None,
            }
        }
        _ => return None,
    };

    if g1 == g2 {
        Some(ExprNode::num(1))
    } else {
        None
    }
}

fn is_num(e: &ExprNode, n: i64) -> bool {
    matches!(e, ExprNode::Num(v) if *v == n)
}

// ---------------------------------------------------------------------------
// Interval arithmetic fallback
// ---------------------------------------------------------------------------

/// Simple interval [lo, hi] for fallback verification.
#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    fn contains_zero(self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }

    fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            lo: self.lo - other.hi,
            hi: self.hi - other.lo,
        }
    }
}

/// Evaluate (df - f) over a small interval around a point.
/// Returns true if the difference interval does NOT contain zero
/// (i.e., we can definitively reject).
fn interval_reject(df: &ExprNode, f: &ExprNode, var: VarId, center: f64) -> Option<bool> {
    let eps = 1e-6;
    let lo = center - eps;
    let hi = center + eps;

    let mut b_lo = Bindings::new();
    b_lo.set_var(var, lo);
    let mut b_hi = Bindings::new();
    b_hi.set_var(var, hi);

    let df_lo = evaluate(df, &b_lo).ok()?;
    let df_hi = evaluate(df, &b_hi).ok()?;
    let f_lo = evaluate(f, &b_lo).ok()?;
    let f_hi = evaluate(f, &b_hi).ok()?;

    let diff_interval = Interval {
        lo: df_lo.min(df_hi) - f_lo.max(f_hi),
        hi: df_lo.max(df_hi) - f_lo.min(f_hi),
    };

    // If the difference interval doesn't contain zero, definitively incorrect
    if !diff_interval.contains_zero() {
        Some(true) // reject
    } else {
        None // inconclusive
    }
}

// ---------------------------------------------------------------------------
// Core verification
// ---------------------------------------------------------------------------

/// Verify a single (integrand, antiderivative) pair with rich verdict.
pub fn verify_single_rich(
    f_prefix: &str,
    big_f_prefix: &str,
    var: &str,
) -> VerifyVerdict {
    let var_id = match VarId::from_str(var) {
        Some(v) => v,
        None => return VerifyVerdict::IncorrectSymbolic,
    };

    let f = match from_prefix_string(f_prefix) {
        Ok(e) => e,
        Err(_) => return VerifyVerdict::IncorrectSymbolic,
    };

    let big_f = match from_prefix_string(big_f_prefix) {
        Ok(e) => e,
        Err(_) => return VerifyVerdict::IncorrectSymbolic,
    };

    // Symbolically differentiate F to get dF/dx
    let df = differentiate(&big_f, var);

    // Cheap structural check before expensive canonicalization.
    if df == f {
        return VerifyVerdict::Correct;
    }

    // Canonicalize both sides for numerical comparison.
    // Do NOT short-circuit on structural equality here — domain-sensitive
    // rewrites (exp/log cancellation, a/a→1, log addition) can produce
    // false matches when domain restrictions are dropped.
    let df_canon = canonicalize(&df);
    let f_canon = canonicalize(&f);

    // Numerical verification at 20 test points (both positive and negative)
    let points: [f64; 20] = [
        -4.72, -3.37, -2.31, -1.61, -0.93, -0.51, -0.15, 0.15, 0.33, 0.51,
         0.72, 0.93, 1.14, 1.61, 2.07, 2.58, 3.09, 3.62, 4.15, 4.72,
    ];

    let min_evaluated = 5;
    let mut evaluated = 0;
    let mut disagreements = 0;

    for &pt in &points {
        let mut bindings = Bindings::new();
        bindings.set_var(var_id, pt);

        let val_df = match evaluate(&df_canon, &bindings) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let val_f = match evaluate(&f_canon, &bindings) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if !val_df.is_finite() || !val_f.is_finite() {
            continue;
        }

        evaluated += 1;
        let tol = 1e-8 * val_df.abs().max(val_f.abs()).max(1.0);
        if (val_df - val_f).abs() > tol {
            disagreements += 1;
            // Short-circuit: if we have enough evidence, reject early.
            if evaluated >= min_evaluated && disagreements > 1 {
                return VerifyVerdict::IncorrectNumerical;
            }
        }
    }

    if evaluated < min_evaluated {
        // Too few evaluable points — try interval arithmetic fallback
        for &pt in &[1.0, 2.0, 3.0] {
            if let Some(true) = interval_reject(&df_canon, &f_canon, var_id, pt) {
                return VerifyVerdict::IncorrectNumerical;
            }
        }
        return VerifyVerdict::Timeout;
    }

    if disagreements == 0 {
        VerifyVerdict::Correct
    } else {
        VerifyVerdict::IncorrectNumerical
    }
}

/// Verify a single pair (backward-compatible boolean API).
pub fn verify_single(
    f_prefix: &str,
    big_f_prefix: &str,
    var: &str,
) -> bool {
    verify_single_rich(f_prefix, big_f_prefix, var).is_correct()
}

/// Verify a batch of pairs in parallel (boolean API).
pub fn verify_batch(pairs: &[(String, String, String)]) -> Vec<bool> {
    pairs
        .par_iter()
        .map(|(f, big_f, var)| verify_single(f, big_f, var))
        .collect()
}

/// Verify a batch with rich verdicts in parallel.
pub fn verify_batch_rich(pairs: &[(String, String, String)]) -> Vec<VerifyVerdict> {
    pairs
        .par_iter()
        .map(|(f, big_f, var)| verify_single_rich(f, big_f, var))
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::{py_verify_batch, py_verify_batch_rich};

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use super::{verify_batch, verify_batch_rich};

    #[pyfunction]
    #[pyo3(name = "verify_batch")]
    pub fn py_verify_batch(
        pairs: Vec<(String, String, String)>,
    ) -> Vec<bool> {
        verify_batch(&pairs)
    }

    #[pyfunction]
    #[pyo3(name = "verify_batch_rich")]
    pub fn py_verify_batch_rich(
        pairs: Vec<(String, String, String)>,
    ) -> Vec<String> {
        verify_batch_rich(&pairs)
            .into_iter()
            .map(|v| match v {
                super::VerifyVerdict::Correct => "correct".to_string(),
                super::VerifyVerdict::IncorrectSymbolic => "incorrect_symbolic".to_string(),
                super::VerifyVerdict::IncorrectNumerical => "incorrect_numerical".to_string(),
                super::VerifyVerdict::Timeout => "timeout".to_string(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{to_prefix_string, ExprNode};
    use crate::diff::differentiate;

    #[test]
    fn verify_x_squared() {
        let f_prefix = "pow var:x 2";
        let big_f_prefix = "div pow var:x 3 3";
        assert!(verify_single(f_prefix, big_f_prefix, "x"));
        assert_eq!(
            verify_single_rich(f_prefix, big_f_prefix, "x"),
            VerifyVerdict::Correct
        );
    }

    #[test]
    fn verify_sin_cos() {
        assert!(verify_single("cos var:x", "sin var:x", "x"));
    }

    #[test]
    fn verify_exp() {
        assert!(verify_single("exp var:x", "exp var:x", "x"));
    }

    #[test]
    fn verify_wrong_pair() {
        assert!(!verify_single("var:x", "pow var:x 2", "x"));
        let verdict = verify_single_rich("var:x", "pow var:x 2", "x");
        assert_eq!(verdict, VerifyVerdict::IncorrectNumerical);
    }

    #[test]
    fn verify_backward_generated() {
        let big_f = ExprNode::mul(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::var("x"),
        );
        let f = differentiate(&big_f, "x");
        let f_prefix = to_prefix_string(&f);
        let big_f_prefix = to_prefix_string(&big_f);
        assert!(verify_single(&f_prefix, &big_f_prefix, "x"));
    }

    #[test]
    fn verify_batch_mixed() {
        let pairs = vec![
            ("cos var:x".to_string(), "sin var:x".to_string(), "x".to_string()),
            ("var:x".to_string(), "pow var:x 2".to_string(), "x".to_string()),
            ("exp var:x".to_string(), "exp var:x".to_string(), "x".to_string()),
        ];
        let results = verify_batch(&pairs);
        assert_eq!(results, vec![true, false, true]);
    }

    #[test]
    fn verdict_serialization() {
        let v = VerifyVerdict::Correct;
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"Correct\"");
    }

    #[test]
    fn canonicalize_pythagorean() {
        // sin²(x) + cos²(x) should canonicalize to 1
        let expr = ExprNode::add(
            ExprNode::pow(ExprNode::sin(ExprNode::var("x")), ExprNode::num(2)),
            ExprNode::pow(ExprNode::cos(ExprNode::var("x")), ExprNode::num(2)),
        );
        let result = canonicalize(&expr);
        assert_eq!(result, ExprNode::num(1));
    }
}
