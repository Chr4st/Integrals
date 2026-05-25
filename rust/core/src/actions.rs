//! Action-space for step-by-step integration (Sprint 3).
//!
//! Four typed actions that transform an integrand state:
//! - Substitute(u = g(x))
//! - IntegrateByParts(u, dv)
//! - PartialFractions
//! - Close(F)

use serde::{Deserialize, Serialize};

use crate::diff::differentiate;
use crate::expr::{BinaryOp, ExprNode, UnaryOp, VarId};

/// An integration action that can be applied to a state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// u-substitution: replace g(x) with u, adjust dx → du/g'(x).
    Substitute {
        /// The inner function g(x) to substitute.
        inner_fn: ExprNode,
    },
    /// Integration by parts: ∫ u dv = uv - ∫ v du.
    IntegrateByParts {
        /// The "u" factor (will be differentiated).
        u_factor: ExprNode,
        /// The "dv" factor (will be integrated / left symbolic).
        dv_factor: ExprNode,
    },
    /// Partial fraction decomposition of a rational integrand.
    PartialFractions,
    /// Close the integration: declare F as the antiderivative.
    Close {
        antiderivative: ExprNode,
    },
}

/// Errors arising from ill-typed or invalid action applications.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActionError {
    /// Action cannot apply to the current state type.
    IllTyped(String),
    /// Parameters are invalid (e.g., u not a subexpression).
    InvalidParams(String),
    /// The per-step verifier rejected the transformation.
    VerifyFailed,
}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllTyped(msg) => write!(f, "ill-typed action: {msg}"),
            Self::InvalidParams(msg) => write!(f, "invalid params: {msg}"),
            Self::VerifyFailed => write!(f, "per-step verification failed"),
        }
    }
}

/// Apply a substitution u = g(x) to an expression.
///
/// Replaces all occurrences of `inner_fn` in `expr` with `Var(u_var)`,
/// and divides by g'(x) to account for the change of variable.
pub fn apply_substitute(
    expr: &ExprNode,
    inner_fn: &ExprNode,
    var: &str,
    u_var: VarId,
) -> Result<ExprNode, ActionError> {
    // Check that inner_fn actually appears in expr
    if !contains_subtree(expr, inner_fn) {
        return Err(ActionError::InvalidParams(
            "inner function not found in expression".to_string(),
        ));
    }

    let g_prime = differentiate(inner_fn, var);

    // Replace inner_fn with u_var in the expression
    let substituted = replace_subtree(expr, inner_fn, &ExprNode::Var(u_var));

    // Divide by g'(x) — in the substituted form, this becomes part of
    // the transformed integrand
    Ok(ExprNode::div(substituted, g_prime))
}

/// Apply integration by parts: ∫ u dv = uv - ∫ v du.
///
/// `dv_factor` must be the antiderivative v (not dv itself), since we
/// lack a general symbolic integrator. The policy network is expected
/// to supply v directly. Returns (closed_term = u*v, remainder = v*du).
pub fn apply_ibp(
    u_factor: &ExprNode,
    v_factor: &ExprNode,
    var: &str,
) -> Result<(ExprNode, ExprNode), ActionError> {
    if u_factor.is_zero() || v_factor.is_zero() {
        return Err(ActionError::IllTyped(
            "IBP requires non-zero u and v".to_string(),
        ));
    }

    let du = differentiate(u_factor, var);

    // ∫ u dv = u*v - ∫ v du
    let closed_term = ExprNode::mul(u_factor.clone(), v_factor.clone());
    let remainder = ExprNode::mul(v_factor.clone(), du);

    Ok((closed_term, remainder))
}

/// Check whether `needle` appears as a subtree of `haystack`.
pub fn contains_subtree(haystack: &ExprNode, needle: &ExprNode) -> bool {
    if haystack == needle {
        return true;
    }
    haystack
        .children()
        .iter()
        .any(|child| contains_subtree(child, needle))
}

/// Replace all occurrences of `old` with `new` in `expr`.
pub fn replace_subtree(
    expr: &ExprNode,
    old: &ExprNode,
    new: &ExprNode,
) -> ExprNode {
    if expr == old {
        return new.clone();
    }
    match expr {
        ExprNode::Unary(op, inner) => {
            ExprNode::Unary(*op, Box::new(replace_subtree(inner, old, new)))
        }
        ExprNode::Binary(op, l, r) => ExprNode::Binary(
            *op,
            Box::new(replace_subtree(l, old, new)),
            Box::new(replace_subtree(r, old, new)),
        ),
        ExprNode::Hyp2F1(a, b, c, d) => ExprNode::Hyp2F1(
            Box::new(replace_subtree(a, old, new)),
            Box::new(replace_subtree(b, old, new)),
            Box::new(replace_subtree(c, old, new)),
            Box::new(replace_subtree(d, old, new)),
        ),
        other => other.clone(),
    }
}

/// Action ID for the policy network (maps action type to index).
pub const ACTION_SUBSTITUTE: u8 = 0;
pub const ACTION_IBP: u8 = 1;
pub const ACTION_PARTIAL_FRACTIONS: u8 = 2;
pub const ACTION_CLOSE: u8 = 3;
pub const NUM_ACTIONS: usize = 4;

impl Action {
    pub fn action_id(&self) -> u8 {
        match self {
            Self::Substitute { .. } => ACTION_SUBSTITUTE,
            Self::IntegrateByParts { .. } => ACTION_IBP,
            Self::PartialFractions => ACTION_PARTIAL_FRACTIONS,
            Self::Close { .. } => ACTION_CLOSE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_simple() {
        // ∫ 2x * exp(x²) dx, substitute u = x²
        let expr = ExprNode::mul(
            ExprNode::mul(ExprNode::num(2), ExprNode::var("x")),
            ExprNode::exp(ExprNode::pow(ExprNode::var("x"), ExprNode::num(2))),
        );
        let inner = ExprNode::pow(ExprNode::var("x"), ExprNode::num(2));
        let result = apply_substitute(&expr, &inner, "x", VarId::Y);
        assert!(result.is_ok());
    }

    #[test]
    fn substitute_not_found() {
        let expr = ExprNode::sin(ExprNode::var("x"));
        let inner = ExprNode::cos(ExprNode::var("x"));
        let result = apply_substitute(&expr, &inner, "x", VarId::Y);
        assert!(matches!(result, Err(ActionError::InvalidParams(_))));
    }

    #[test]
    fn ibp_basic() {
        // ∫ x * exp(x) dx — u = x, dv = exp(x)
        let u = ExprNode::var("x");
        let dv = ExprNode::exp(ExprNode::var("x"));
        let result = apply_ibp(&u, &dv, "x");
        assert!(result.is_ok());
    }

    #[test]
    fn contains_subtree_basic() {
        let expr = ExprNode::add(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::num(1),
        );
        assert!(contains_subtree(&expr, &ExprNode::sin(ExprNode::var("x"))));
        assert!(contains_subtree(&expr, &ExprNode::var("x")));
        assert!(!contains_subtree(&expr, &ExprNode::cos(ExprNode::var("x"))));
    }

    #[test]
    fn replace_subtree_basic() {
        let expr = ExprNode::add(ExprNode::var("x"), ExprNode::var("x"));
        let result = replace_subtree(
            &expr,
            &ExprNode::var("x"),
            &ExprNode::var("y"),
        );
        assert_eq!(
            result,
            ExprNode::add(ExprNode::var("y"), ExprNode::var("y"))
        );
    }

    #[test]
    fn action_ids() {
        assert_eq!(Action::PartialFractions.action_id(), ACTION_PARTIAL_FRACTIONS);
        assert_eq!(
            Action::Close { antiderivative: ExprNode::num(0) }.action_id(),
            ACTION_CLOSE
        );
    }
}
