use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::diff::differentiate;
use crate::expr::{BinaryOp, ExprNode, MathConst, ParamId, UnaryOp, VarId};

/// Generation mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GenMode {
    Univariate,
    Multivariate,
    Definite,
    Parametric,
    SpecialFn,
}

/// Configuration for random tree generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenConfig {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub vars: Vec<VarId>,
    pub params: Vec<ParamId>,
    pub allow_special_fns: bool,
    pub num_range: (i64, i64),
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            max_depth: 6,
            max_nodes: 30,
            vars: vec![VarId::X],
            params: vec![],
            allow_special_fns: false,
            num_range: (-5, 5),
        }
    }
}

/// An (integrand, integral) pair.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pair {
    pub integrand: ExprNode,
    pub integral: ExprNode,
}

// ---------------------------------------------------------------------------
// Random tree generation
// ---------------------------------------------------------------------------
pub fn generate_random_tree(
    config: &GenConfig,
    rng: &mut impl Rng,
) -> ExprNode {
    gen_node(config, rng, 1, &mut 0)
}

fn gen_node(
    config: &GenConfig,
    rng: &mut impl Rng,
    depth: usize,
    count: &mut usize,
) -> ExprNode {
    *count += 1;
    // Force leaf at max depth or max nodes
    if depth >= config.max_depth || *count >= config.max_nodes {
        return gen_leaf(config, rng);
    }

    // Node type weights: 40% binary, 25% unary, 5% special fn, 30% leaf
    let roll: f64 = rng.gen();
    let threshold_binary = 0.40;
    let threshold_unary = 0.65;
    let threshold_special = if config.allow_special_fns { 0.70 } else { 0.65 };

    if roll < threshold_binary {
        let op = pick_binary_op(rng, config.allow_special_fns);
        let l = gen_node(config, rng, depth + 1, count);
        let r = gen_node(config, rng, depth + 1, count);
        ExprNode::Binary(op, Box::new(l), Box::new(r))
    } else if roll < threshold_unary {
        let op = pick_unary_op(rng, false);
        let c = gen_node(config, rng, depth + 1, count);
        ExprNode::Unary(op, Box::new(c))
    } else if roll < threshold_special && config.allow_special_fns {
        let op = pick_unary_op(rng, true);
        let c = gen_node(config, rng, depth + 1, count);
        ExprNode::Unary(op, Box::new(c))
    } else {
        gen_leaf(config, rng)
    }
}

fn gen_leaf(config: &GenConfig, rng: &mut impl Rng) -> ExprNode {
    // Distribution: 40% var, 20% param (if any), 30% num, 10% const
    let has_params = !config.params.is_empty();
    let roll: f64 = rng.gen();
    if roll < 0.40 && !config.vars.is_empty() {
        let idx = rng.gen_range(0..config.vars.len());
        ExprNode::Var(config.vars[idx])
    } else if roll < 0.60 && has_params {
        let idx = rng.gen_range(0..config.params.len());
        ExprNode::Param(config.params[idx])
    } else if roll < 0.90 {
        let n = rng.gen_range(config.num_range.0..=config.num_range.1);
        // Avoid zero to prevent degenerate trees
        if n == 0 { ExprNode::Num(1) } else { ExprNode::Num(n) }
    } else {
        let consts = [MathConst::Pi, MathConst::EulerGamma, MathConst::Catalan];
        let idx = rng.gen_range(0..consts.len());
        ExprNode::Const(consts[idx].clone())
    }
}

fn pick_binary_op(rng: &mut impl Rng, allow_special: bool) -> BinaryOp {
    let standard = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Pow,
    ];
    let special = [BinaryOp::BesselJ, BinaryOp::BesselY, BinaryOp::Polylog];
    if allow_special && rng.gen_bool(0.15) {
        let idx = rng.gen_range(0..special.len());
        special[idx].clone()
    } else {
        let idx = rng.gen_range(0..standard.len());
        standard[idx].clone()
    }
}

fn pick_unary_op(rng: &mut impl Rng, special_only: bool) -> UnaryOp {
    let standard = [
        UnaryOp::Neg, UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Tan,
        UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sqrt,
        UnaryOp::Asin, UnaryOp::Acos, UnaryOp::Atan,
        UnaryOp::Sinh, UnaryOp::Cosh, UnaryOp::Tanh,
    ];
    let special = [
        UnaryOp::Erf, UnaryOp::Ei, UnaryOp::Si, UnaryOp::Ci,
        UnaryOp::Li, UnaryOp::Gamma, UnaryOp::Digamma,
        UnaryOp::FresnelS, UnaryOp::FresnelC,
        UnaryOp::EllipticK, UnaryOp::EllipticE,
    ];
    if special_only {
        let idx = rng.gen_range(0..special.len());
        special[idx].clone()
    } else {
        let idx = rng.gen_range(0..standard.len());
        standard[idx].clone()
    }
}

// ---------------------------------------------------------------------------
// Batch generation: differentiate integral to get integrand
// ---------------------------------------------------------------------------
fn config_for_mode(mode: &GenMode, base: &GenConfig) -> GenConfig {
    let mut cfg = base.clone();
    match mode {
        GenMode::Univariate => {
            cfg.vars = vec![VarId::X];
            cfg.params = vec![];
            cfg.allow_special_fns = false;
        }
        GenMode::Multivariate => {
            if cfg.vars.len() < 2 {
                cfg.vars = vec![VarId::X, VarId::Y];
            }
            cfg.params = vec![];
        }
        GenMode::Definite => {
            cfg.vars = vec![VarId::X];
            cfg.params = vec![];
        }
        GenMode::Parametric => {
            cfg.vars = vec![VarId::X];
            if cfg.params.is_empty() {
                cfg.params = vec![ParamId::A, ParamId::B];
            }
        }
        GenMode::SpecialFn => {
            cfg.vars = vec![VarId::X];
            cfg.allow_special_fns = true;
        }
    }
    cfg
}

pub fn generate_batch(
    n: usize,
    mode: GenMode,
    config: &GenConfig,
) -> Vec<Pair> {
    let cfg = config_for_mode(&mode, config);
    let var = cfg.vars.first().copied().unwrap_or(VarId::X);
    let var_str = var.as_str();

    (0..n)
        .into_par_iter()
        .map(|_i| {
            let mut rng = rand::thread_rng();
            let integral = generate_random_tree(&cfg, &mut rng);
            let integrand = differentiate(&integral, var_str);
            Pair { integrand, integral }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Equivalence variant generation (Sprint 1)
// ---------------------------------------------------------------------------

/// Generate K algebraically equivalent variants of an antiderivative.
/// Each variant is obtained by applying a rewrite rule that preserves
/// the integral identity (up to a constant).
pub fn generate_equivalents(integral: &ExprNode, k: usize) -> Vec<ExprNode> {
    let mut variants = Vec::with_capacity(k);
    let rules: Vec<fn(&ExprNode) -> Option<ExprNode>> = vec![
        rewrite_factor_constant,
        rewrite_distribute_add,
        rewrite_trig_identity,
        rewrite_log_normalize,
    ];

    for rule in &rules {
        if variants.len() >= k {
            break;
        }
        if let Some(v) = rule(integral) {
            if v != *integral && !variants.contains(&v) {
                variants.push(v);
            }
        }
    }

    // If we still need more, try applying rules to children
    if variants.len() < k {
        if let Some(v) = rewrite_negate_negate(integral) {
            if v != *integral && !variants.contains(&v) {
                variants.push(v);
            }
        }
    }

    variants.truncate(k);
    variants
}

/// Pull out a constant factor: c*f(x) → c * f(x) (explicit mul form)
fn rewrite_factor_constant(expr: &ExprNode) -> Option<ExprNode> {
    match expr {
        ExprNode::Binary(BinaryOp::Mul, l, r) => {
            // If left is a number, rewrite as (num * right)
            if let ExprNode::Num(n) = l.as_ref() {
                Some(ExprNode::Binary(
                    BinaryOp::Mul,
                    Box::new(ExprNode::Num(*n)),
                    Box::new(r.as_ref().clone()),
                ))
            } else if let ExprNode::Num(n) = r.as_ref() {
                // Swap: right*left → left*right
                Some(ExprNode::Binary(
                    BinaryOp::Mul,
                    Box::new(ExprNode::Num(*n)),
                    Box::new(l.as_ref().clone()),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Distribute addition: a*(b+c) → a*b + a*c
fn rewrite_distribute_add(expr: &ExprNode) -> Option<ExprNode> {
    match expr {
        ExprNode::Binary(BinaryOp::Mul, a, bc) => {
            if let ExprNode::Binary(BinaryOp::Add, b, c) = bc.as_ref() {
                Some(ExprNode::add(
                    ExprNode::mul(a.as_ref().clone(), b.as_ref().clone()),
                    ExprNode::mul(a.as_ref().clone(), c.as_ref().clone()),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// sin²(x) → 1 - cos²(x)
fn rewrite_trig_identity(expr: &ExprNode) -> Option<ExprNode> {
    match expr {
        ExprNode::Binary(BinaryOp::Pow, base, exp) => {
            if let ExprNode::Num(2) = exp.as_ref() {
                match base.as_ref() {
                    ExprNode::Unary(UnaryOp::Sin, inner) => {
                        Some(ExprNode::sub(
                            ExprNode::num(1),
                            ExprNode::pow(
                                ExprNode::cos(inner.as_ref().clone()),
                                ExprNode::num(2),
                            ),
                        ))
                    }
                    ExprNode::Unary(UnaryOp::Cos, inner) => {
                        Some(ExprNode::sub(
                            ExprNode::num(1),
                            ExprNode::pow(
                                ExprNode::sin(inner.as_ref().clone()),
                                ExprNode::num(2),
                            ),
                        ))
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// log(a*b) → log(a) + log(b)
fn rewrite_log_normalize(expr: &ExprNode) -> Option<ExprNode> {
    match expr {
        ExprNode::Unary(UnaryOp::Log, inner) => {
            if let ExprNode::Binary(BinaryOp::Mul, a, b) = inner.as_ref() {
                Some(ExprNode::add(
                    ExprNode::log(a.as_ref().clone()),
                    ExprNode::log(b.as_ref().clone()),
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// --f(x) → f(x)  (cancel double negation)
fn rewrite_negate_negate(expr: &ExprNode) -> Option<ExprNode> {
    match expr {
        // neg(neg(inner)) → inner
        ExprNode::Unary(UnaryOp::Neg, inner) => {
            if let ExprNode::Unary(UnaryOp::Neg, inner2) = inner.as_ref() {
                Some(inner2.as_ref().clone())
            } else {
                None
            }
        }
        // mul(-1, mul(-1, inner)) → inner
        ExprNode::Binary(BinaryOp::Mul, l, r) => {
            if let ExprNode::Num(-1) = l.as_ref() {
                if let ExprNode::Binary(BinaryOp::Mul, l2, r2) = r.as_ref() {
                    if let ExprNode::Num(-1) = l2.as_ref() {
                        return Some(r2.as_ref().clone());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// An (integrand, integral, equivalents) triple.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairWithEquivalents {
    pub integrand: ExprNode,
    pub integral: ExprNode,
    pub equivalents: Vec<ExprNode>,
}

/// Generate a batch with K=4 equivalence variants per pair.
pub fn generate_batch_with_equivalents(
    n: usize,
    mode: GenMode,
    config: &GenConfig,
    equiv_k: usize,
) -> Vec<PairWithEquivalents> {
    let cfg = config_for_mode(&mode, config);
    let var = cfg.vars.first().copied().unwrap_or(VarId::X);
    let var_str = var.as_str();

    (0..n)
        .into_par_iter()
        .map(|_i| {
            let mut rng = rand::thread_rng();
            let integral = generate_random_tree(&cfg, &mut rng);
            let integrand = differentiate(&integral, var_str);
            let equivalents = generate_equivalents(&integral, equiv_k);
            PairWithEquivalents {
                integrand,
                integral,
                equivalents,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::{PyGenConfig, py_generate_batch};

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use super::*;
    use crate::expr::PyExprTree;

    #[pyclass(name = "GenConfig")]
    #[derive(Clone, Debug)]
    pub struct PyGenConfig {
        pub inner: GenConfig,
    }

    #[pymethods]
    impl PyGenConfig {
        #[new]
        #[pyo3(signature = (
            max_depth = 6,
            max_nodes = 30,
            vars = None,
            params = None,
            allow_special_fns = false,
            num_range = None,
        ))]
        fn new(
            max_depth: usize,
            max_nodes: usize,
            vars: Option<Vec<String>>,
            params: Option<Vec<String>>,
            allow_special_fns: bool,
            num_range: Option<(i64, i64)>,
        ) -> Self {
            let vars = vars
                .unwrap_or_else(|| vec!["x".to_string()])
                .into_iter()
                .map(|s| VarId::from_str(&s).expect("unknown var"))
                .collect();
            let params = params
                .unwrap_or_default()
                .into_iter()
                .map(|s| ParamId::from_str(&s).expect("unknown param"))
                .collect();
            Self {
                inner: GenConfig {
                    max_depth,
                    max_nodes,
                    vars,
                    params,
                    allow_special_fns,
                    num_range: num_range.unwrap_or((-5, 5)),
                },
            }
        }

        fn __repr__(&self) -> String {
            format!("{:?}", self.inner)
        }
    }

    #[pyfunction]
    #[pyo3(name = "generate_batch")]
    pub fn py_generate_batch(
        n: usize,
        mode: &str,
        config: &PyGenConfig,
    ) -> PyResult<Vec<(PyExprTree, PyExprTree)>> {
        let mode = match mode {
            "univariate" => GenMode::Univariate,
            "multivariate" => GenMode::Multivariate,
            "definite" => GenMode::Definite,
            "parametric" => GenMode::Parametric,
            "special" => GenMode::SpecialFn,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("unknown mode: {other}"),
                ));
            }
        };
        if config.inner.max_depth == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_depth must be > 0",
            ));
        }
        if config.inner.max_nodes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_nodes must be > 0",
            ));
        }
        if config.inner.vars.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "vars must not be empty",
            ));
        }
        let pairs = generate_batch(n, mode, &config.inner);
        Ok(pairs
            .into_iter()
            .map(|p| {
                (
                    PyExprTree { inner: p.integrand },
                    PyExprTree { inner: p.integral },
                )
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_produces_valid_tree() {
        let cfg = GenConfig::default();
        let mut rng = rand::thread_rng();
        let tree = generate_random_tree(&cfg, &mut rng);
        assert!(tree.depth() >= 1);
        assert!(tree.node_count() >= 1);
    }

    #[test]
    fn gen_respects_max_depth() {
        let cfg = GenConfig {
            max_depth: 3,
            ..GenConfig::default()
        };
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let tree = generate_random_tree(&cfg, &mut rng);
            assert!(tree.depth() <= cfg.max_depth);
        }
    }

    #[test]
    fn batch_produces_pairs() {
        let cfg = GenConfig::default();
        let pairs = generate_batch(10, GenMode::Univariate, &cfg);
        assert_eq!(pairs.len(), 10);
        for p in &pairs {
            assert!(p.integrand.node_count() >= 1);
            assert!(p.integral.node_count() >= 1);
        }
    }

    #[test]
    fn mode_special_enables_special_fns() {
        let cfg = GenConfig::default();
        let pairs = generate_batch(50, GenMode::SpecialFn, &cfg);
        // At least some trees should be non-trivial
        let total_nodes: usize =
            pairs.iter().map(|p| p.integral.node_count()).sum();
        assert!(total_nodes > 50);
    }

    #[test]
    fn config_for_mode_parametric() {
        let base = GenConfig::default();
        let cfg = config_for_mode(&GenMode::Parametric, &base);
        assert!(!cfg.params.is_empty());
    }
}
