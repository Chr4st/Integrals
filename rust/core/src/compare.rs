use crate::eval::{evaluate, Bindings};
use crate::expr::{ExprNode, VarId};

/// Same tree shape and operators, ignoring numeric leaf values.
/// Variables, parameters, and constants must still match exactly.
pub fn structural_eq(a: &ExprNode, b: &ExprNode) -> bool {
    match (a, b) {
        (ExprNode::Num(_), ExprNode::Num(_)) => true,
        (ExprNode::Rational(_, _), ExprNode::Rational(_, _)) => true,
        (ExprNode::Var(va), ExprNode::Var(vb)) => va == vb,
        (ExprNode::Param(pa), ExprNode::Param(pb)) => pa == pb,
        (ExprNode::Const(ca), ExprNode::Const(cb)) => ca == cb,
        (ExprNode::Unary(oa, ca), ExprNode::Unary(ob, cb)) => {
            oa == ob && structural_eq(ca, cb)
        }
        (ExprNode::Binary(oa, la, ra), ExprNode::Binary(ob, lb, rb)) => {
            oa == ob && structural_eq(la, lb) && structural_eq(ra, rb)
        }
        (
            ExprNode::Hyp2F1(a1, a2, a3, a4),
            ExprNode::Hyp2F1(b1, b2, b3, b4),
        ) => {
            structural_eq(a1, b1)
                && structural_eq(a2, b2)
                && structural_eq(a3, b3)
                && structural_eq(a4, b4)
        }
        _ => false,
    }
}

/// Fully identical trees, including numeric values.
pub fn exact_eq(a: &ExprNode, b: &ExprNode) -> bool {
    a == b
}

/// Check whether a candidate matches any of K equivalent targets.
///
/// Two expressions are considered equivalent if they agree numerically
/// at multiple test points (up to a constant offset for antiderivatives).
pub fn compare_equivalence_class(
    targets: &[ExprNode],
    candidate: &ExprNode,
    var: VarId,
) -> bool {
    if targets.is_empty() {
        return false;
    }

    // Quick exact-match check first
    for target in targets {
        if exact_eq(target, candidate) {
            return true;
        }
    }

    // Numerical equivalence: candidate ≡ target + C for some constant C
    let points: [f64; 10] = [
        0.21, 0.47, 0.89, 1.23, 1.67, 2.11, 2.53, 3.01, 3.47, 4.02,
    ];

    for target in targets {
        if numerical_equiv(target, candidate, var, &points) {
            return true;
        }
    }

    false
}

/// Check if two expressions differ by at most a constant.
/// Computes candidate(x) - target(x) at each point and checks
/// whether all differences are equal (within tolerance).
fn numerical_equiv(
    target: &ExprNode,
    candidate: &ExprNode,
    var: VarId,
    points: &[f64],
) -> bool {
    let mut diffs: Vec<f64> = Vec::with_capacity(points.len());

    for &pt in points {
        let mut bindings = Bindings::new();
        bindings.set_var(var, pt);

        let val_target = match evaluate(target, &bindings) {
            Ok(v) if v.is_finite() => v,
            _ => continue,
        };
        let val_candidate = match evaluate(candidate, &bindings) {
            Ok(v) if v.is_finite() => v,
            _ => continue,
        };

        diffs.push(val_candidate - val_target);
    }

    if diffs.len() < 5 {
        return false;
    }

    // Use the median diff as the reference constant (robust to outliers)
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let c = diffs[diffs.len() / 2];
    let tol = 1e-6 * c.abs().max(1.0);
    diffs.iter().all(|&d| (d - c).abs() < tol)
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::py_compare_equivalence_class;

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use crate::expr::{from_prefix_string, VarId};

    #[pyfunction]
    #[pyo3(name = "compare_equivalence_class")]
    pub fn py_compare_equivalence_class(
        target_prefixes: Vec<String>,
        candidate_prefix: &str,
        var: &str,
    ) -> PyResult<bool> {
        let var_id = VarId::from_str(var)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("unknown variable: {var}"),
            ))?;
        let targets: Vec<_> = target_prefixes
            .iter()
            .filter_map(|s| from_prefix_string(s).ok())
            .collect();
        let candidate = from_prefix_string(candidate_prefix)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(super::compare_equivalence_class(&targets, &candidate, var_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structural_eq_ignores_numbers() {
        let a = ExprNode::add(ExprNode::num(3), ExprNode::var("x"));
        let b = ExprNode::add(ExprNode::num(99), ExprNode::var("x"));
        assert!(structural_eq(&a, &b));
        assert!(!exact_eq(&a, &b));
    }

    #[test]
    fn structural_eq_requires_same_shape() {
        let a = ExprNode::add(ExprNode::var("x"), ExprNode::num(1));
        let b = ExprNode::mul(ExprNode::var("x"), ExprNode::num(1));
        assert!(!structural_eq(&a, &b));
    }

    #[test]
    fn structural_eq_checks_var_names() {
        let a = ExprNode::sin(ExprNode::var("x"));
        let b = ExprNode::sin(ExprNode::var("y"));
        assert!(!structural_eq(&a, &b));
    }

    #[test]
    fn exact_eq_identical() {
        let a = ExprNode::mul(ExprNode::num(2), ExprNode::var("x"));
        let b = ExprNode::mul(ExprNode::num(2), ExprNode::var("x"));
        assert!(exact_eq(&a, &b));
    }

    #[test]
    fn structural_eq_hyp2f1() {
        let a = ExprNode::Hyp2F1(
            Box::new(ExprNode::num(1)),
            Box::new(ExprNode::num(2)),
            Box::new(ExprNode::num(3)),
            Box::new(ExprNode::var("x")),
        );
        let b = ExprNode::Hyp2F1(
            Box::new(ExprNode::num(10)),
            Box::new(ExprNode::num(20)),
            Box::new(ExprNode::num(30)),
            Box::new(ExprNode::var("x")),
        );
        assert!(structural_eq(&a, &b));
    }

    #[test]
    fn equivalence_class_exact() {
        let t1 = ExprNode::sin(ExprNode::var("x"));
        let t2 = ExprNode::cos(ExprNode::var("x"));
        let candidate = ExprNode::sin(ExprNode::var("x"));
        assert!(compare_equivalence_class(&[t1, t2], &candidate, VarId::X));
    }

    #[test]
    fn equivalence_class_constant_offset() {
        // sin(x) and sin(x) + 5 differ by a constant
        let target = ExprNode::sin(ExprNode::var("x"));
        let candidate = ExprNode::add(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::num(5),
        );
        assert!(compare_equivalence_class(&[target], &candidate, VarId::X));
    }

    #[test]
    fn equivalence_class_no_match() {
        let target = ExprNode::sin(ExprNode::var("x"));
        let candidate = ExprNode::cos(ExprNode::var("x"));
        assert!(!compare_equivalence_class(&[target], &candidate, VarId::X));
    }
}
