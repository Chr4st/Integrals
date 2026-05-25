use crate::expr::{BinaryOp, ExprNode, UnaryOp, VarId};

const N_FUNC_TYPES: usize = 33;
const N_DEPTH_BINS: usize = 8;
// 264 (func heads: 33*8) + 16 (var role) + 40 (signature) + 24 (task) = 344
pub const FEATURE_DIM: usize = 344;

// 33 function names sorted alphabetically (must match Python FUNCTION_NAMES)
const FUNCTION_NAMES: [&str; 33] = [
    "BesselJ", "BesselY", "Ci", "Ei", "EllipticE", "EllipticK",
    "FresnelC", "FresnelS", "Gamma", "Hyp2F1", "Li", "Si",
    "acos", "add", "asin", "atan", "cos", "cosh", "digamma",
    "div", "erf", "exp", "log", "mul", "neg", "polylog", "pow",
    "sin", "sinh", "sqrt", "sub", "tan", "tanh",
];

// Signature types and their indicator functions
const SIG_GAUSSIAN_DECAY: [&str; 2] = ["exp", "erf"];
const SIG_OSCILLATORY: [&str; 5] = ["sin", "cos", "tan", "FresnelS", "FresnelC"];
const SIG_RATIONAL: [&str; 3] = ["div", "log", "polylog"];
const SIG_ALGEBRAIC: [&str; 2] = ["sqrt", "pow"];
const SIG_EXPONENTIAL: [&str; 3] = ["exp", "sinh", "cosh"];

/// Collected node info for feature extraction (zero-alloc tokens).
struct NodeInfo<'a> {
    token: &'a str,
    depth: usize,
}

fn collect_nodes<'a>(expr: &'a ExprNode, depth: usize, out: &mut Vec<NodeInfo<'a>>) {
    let token: &str = match expr {
        ExprNode::Var(v) => v.as_str(),
        ExprNode::Param(p) => p.as_str(),
        ExprNode::Num(_) => return,
        ExprNode::Rational(_, _) => return,
        ExprNode::Const(c) => c.as_str(),
        ExprNode::Unary(op, child) => {
            out.push(NodeInfo { token: op.as_str(), depth });
            collect_nodes(child, depth + 1, out);
            return;
        }
        ExprNode::Binary(op, l, r) => {
            out.push(NodeInfo { token: op.as_str(), depth });
            collect_nodes(l, depth + 1, out);
            collect_nodes(r, depth + 1, out);
            return;
        }
        ExprNode::Hyp2F1(a, b, c, d) => {
            out.push(NodeInfo { token: "Hyp2F1", depth });
            collect_nodes(a, depth + 1, out);
            collect_nodes(b, depth + 1, out);
            collect_nodes(c, depth + 1, out);
            collect_nodes(d, depth + 1, out);
            return;
        }
    };
    out.push(NodeInfo { token, depth });
}

fn depth_bin(depth: usize) -> usize {
    (depth / 2).min(N_DEPTH_BINS - 1)
}

fn func_index(name: &str) -> Option<usize> {
    FUNCTION_NAMES.iter().position(|&n| n == name)
}

/// Extract a 344-dimensional feature vector from an expression tree.
///
/// Layout:
///   [0..264)    depth-encoded function heads (33 × 8)
///   [264..280)  variable role features (16)
///   [280..320)  signature features (5 types × 8 bins)
///   [320..344)  task/complexity features (24)
pub fn extract_features(
    expr: &ExprNode,
    int_var: VarId,
    task: &str,
    bound_type: &str,
    param_count: usize,
) -> Vec<f32> {
    let mut nodes = Vec::with_capacity(64);
    collect_nodes(expr, 0, &mut nodes);

    let mut out = vec![0.0f32; FEATURE_DIM];

    // --- Single-pass: function heads (264), variable roles, signature features ---
    let base = N_FUNC_TYPES * N_DEPTH_BINS; // 264
    let sig_base = base + 16; // 280
    let var_strs = ["x", "y", "z", "w", "t"];
    let all_vars = [VarId::X, VarId::Y, VarId::Z, VarId::W, VarId::T];
    let int_var_str = int_var.as_str();

    let sig_groups: [&[&str]; 5] = [
        &SIG_GAUSSIAN_DECAY,
        &SIG_OSCILLATORY,
        &SIG_RATIONAL,
        &SIG_ALGEBRAIC,
        &SIG_EXPONENTIAL,
    ];

    let mut present_vars = [false; 5];
    let mut int_var_depths: Vec<usize> = Vec::new();

    for node in &nodes {
        let b = depth_bin(node.depth);
        let tok = node.token;

        // Function head features (264 = 33 types × 8 bins)
        if let Some(idx) = func_index(tok) {
            out[idx * N_DEPTH_BINS + b] += 1.0;
        }

        // Variable presence tracking
        for (i, vs) in var_strs.iter().enumerate() {
            if tok == *vs {
                present_vars[i] = true;
            }
        }
        if tok == int_var_str {
            int_var_depths.push(node.depth);
        }

        // Signature features (5 groups × 8 bins)
        for (sig_idx, indicators) in sig_groups.iter().enumerate() {
            if indicators.contains(&tok) {
                out[sig_base + sig_idx * N_DEPTH_BINS + b] += 1.0;
            }
        }
    }

    // Finalize variable role features (16)
    for (i, _) in all_vars.iter().enumerate() {
        if present_vars[i] {
            out[base + i] = 1.0;
        }
        if all_vars[i] == int_var {
            out[base + 5 + i] = 1.0;
        }
    }

    let total = if nodes.is_empty() { 1 } else { nodes.len() };
    out[base + 10] = int_var_depths.len() as f32 / total as f32;

    if !int_var_depths.is_empty() {
        let min_d = *int_var_depths.iter().min().unwrap();
        let max_d = *int_var_depths.iter().max().unwrap();
        let avg_d: f32 = int_var_depths.iter().sum::<usize>() as f32
            / int_var_depths.len() as f32;
        out[base + 11] = min_d as f32 / 20.0;
        out[base + 12] = max_d as f32 / 20.0;
        out[base + 13] = avg_d / 20.0;
    }

    // --- Task/complexity features (24) ---
    let task_base = sig_base + 40; // 320
    let task_types = ["def", "indef", "lower", "var"];
    for (i, t) in task_types.iter().enumerate() {
        if *t == task {
            out[task_base + i] = 1.0;
        }
    }

    let node_count = expr.node_count();
    let max_depth = expr.depth();
    out[task_base + 4] = (node_count as f32 / 50.0).min(1.0);
    out[task_base + 5] = (max_depth as f32 / 15.0).min(1.0);
    out[task_base + 6] = (param_count as f32 / 5.0).min(1.0);

    let bound_types = ["none", "finite", "semi_inf", "inf"];
    for (i, bt) in bound_types.iter().enumerate() {
        if *bt == bound_type {
            out[task_base + 7 + i] = 1.0;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Simplicity score — ranks antiderivatives by compactness
// ---------------------------------------------------------------------------

/// Operator penalty table for simplicity scoring.
/// Higher penalty = less "simple" in a human sense.
fn operator_penalty(expr: &ExprNode) -> f32 {
    match expr {
        ExprNode::Var(_) | ExprNode::Param(_) | ExprNode::Num(_)
        | ExprNode::Rational(_, _) | ExprNode::Const(_) => 0.0,
        ExprNode::Unary(op, _) => match op {
            UnaryOp::Neg => 0.1,
            UnaryOp::Sin | UnaryOp::Cos | UnaryOp::Exp | UnaryOp::Log => 0.5,
            UnaryOp::Tan | UnaryOp::Sqrt => 0.6,
            UnaryOp::Asin | UnaryOp::Acos | UnaryOp::Atan => 0.8,
            UnaryOp::Sinh | UnaryOp::Cosh | UnaryOp::Tanh => 0.7,
            // Special functions are expensive
            _ => 1.5,
        },
        ExprNode::Binary(op, _, _) => match op {
            BinaryOp::Add | BinaryOp::Sub => 0.2,
            BinaryOp::Mul => 0.3,
            BinaryOp::Div => 0.5,
            BinaryOp::Pow => 0.6,
            _ => 1.5, // Bessel, Polylog
        },
        ExprNode::Hyp2F1(..) => 2.0,
    }
}

/// Compute a simplicity score for an expression.
/// Lower score = simpler expression.
/// Score = tree_size + sum of operator penalties.
pub fn simplicity_score(expr: &ExprNode) -> f32 {
    let size = expr.node_count() as f32;
    let penalty = simplicity_penalty_recursive(expr);
    size + penalty
}

fn simplicity_penalty_recursive(expr: &ExprNode) -> f32 {
    let self_penalty = operator_penalty(expr);
    let child_penalty: f32 = expr
        .children()
        .iter()
        .map(|c| simplicity_penalty_recursive(c))
        .sum();
    self_penalty + child_penalty
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::{py_extract_features, py_simplicity_score};

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use crate::expr::{from_prefix_string, VarId};
    use super::{extract_features, simplicity_score};

    #[pyfunction]
    #[pyo3(name = "extract_features")]
    pub fn py_extract_features(
        prefix_str: &str,
        int_var: &str,
        task: &str,
        bound_type: &str,
        param_count: usize,
    ) -> PyResult<Vec<f32>> {
        let expr = from_prefix_string(prefix_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let var_id = VarId::from_str(int_var)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("unknown variable: {int_var}"),
            ))?;
        Ok(extract_features(&expr, var_id, task, bound_type, param_count))
    }

    #[pyfunction]
    #[pyo3(name = "simplicity_score")]
    pub fn py_simplicity_score(prefix_str: &str) -> PyResult<f32> {
        let expr = from_prefix_string(prefix_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(simplicity_score(&expr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_dim() {
        let e = ExprNode::sin(ExprNode::var("x"));
        let feats = extract_features(&e, VarId::X, "indef", "none", 0);
        assert_eq!(feats.len(), FEATURE_DIM);
    }

    #[test]
    fn feature_nonzero() {
        let e = ExprNode::add(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::exp(ExprNode::var("x")),
        );
        let feats = extract_features(&e, VarId::X, "indef", "none", 0);
        // Should have some non-zero entries
        let nnz = feats.iter().filter(|&&v| v > 0.0).count();
        assert!(nnz > 0);
    }

    #[test]
    fn simplicity_score_ordering() {
        // x should be simpler than sin(x)
        let simple = ExprNode::var("x");
        let complex = ExprNode::sin(ExprNode::var("x"));
        assert!(simplicity_score(&simple) < simplicity_score(&complex));
    }

    #[test]
    fn simplicity_score_special_fn_penalty() {
        // erf(x) should have higher penalty than sin(x)
        let trig = ExprNode::sin(ExprNode::var("x"));
        let special = ExprNode::Unary(UnaryOp::Erf, Box::new(ExprNode::var("x")));
        assert!(simplicity_score(&trig) < simplicity_score(&special));
    }

    #[test]
    fn feature_task_onehot() {
        let e = ExprNode::var("x");
        let feats = extract_features(&e, VarId::X, "def", "finite", 2);
        // task_base = 320, "def" is index 0
        assert_eq!(feats[320], 1.0);
        // "finite" is index 1 at task_base+7
        assert_eq!(feats[320 + 7 + 1], 1.0);
        // param_count normalized
        assert!((feats[320 + 6] - 0.4).abs() < 1e-6); // 2/5 = 0.4
    }
}
