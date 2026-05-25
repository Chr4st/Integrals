use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

mod parse;
pub use parse::{from_prefix_string, to_prefix_string};

// ---------------------------------------------------------------------------
// Interned variable / parameter identifiers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VarId {
    X = 0,
    Y = 1,
    Z = 2,
    W = 3,
    T = 4,
}

impl VarId {
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "x" => Self::X,
            "y" => Self::Y,
            "z" => Self::Z,
            "w" => Self::W,
            "t" => Self::T,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::X => "x",
            Self::Y => "y",
            Self::Z => "z",
            Self::W => "w",
            Self::T => "t",
        }
    }

    pub const ALL: [VarId; 5] = [Self::X, Self::Y, Self::Z, Self::W, Self::T];
}

impl fmt::Display for VarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ParamId {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    N = 4,
    M = 5,
    K = 6,
}

impl ParamId {
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "a" => Self::A,
            "b" => Self::B,
            "c" => Self::C,
            "d" => Self::D,
            "n" => Self::N,
            "m" => Self::M,
            "k" => Self::K,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::C => "c",
            Self::D => "d",
            Self::N => "n",
            Self::M => "m",
            Self::K => "k",
        }
    }

    pub const ALL: [ParamId; 7] = [
        Self::A, Self::B, Self::C, Self::D, Self::N, Self::M, Self::K,
    ];
}

impl fmt::Display for ParamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Mathematical constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MathConst {
    Pi,
    EulerGamma,
    Catalan,
}

impl MathConst {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pi => "pi",
            Self::EulerGamma => "euler_gamma",
            Self::Catalan => "catalan",
        }
    }
}

impl fmt::Display for MathConst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Unary operator tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    // Special functions
    Erf,
    Ei,
    Si,
    Ci,
    Li,
    Gamma,
    Digamma,
    FresnelS,
    FresnelC,
    EllipticK,
    EllipticE,
}

/// Binary operator tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    // Special functions
    BesselJ,
    BesselY,
    Polylog,
}

impl UnaryOp {
    /// Static string name (no allocation).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sqrt => "sqrt",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Erf => "erf",
            Self::Ei => "ei",
            Self::Si => "si",
            Self::Ci => "ci",
            Self::Li => "li",
            Self::Gamma => "gamma",
            Self::Digamma => "digamma",
            Self::FresnelS => "fresnelS",
            Self::FresnelC => "fresnelC",
            Self::EllipticK => "ellipticK",
            Self::EllipticE => "ellipticE",
        }
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl BinaryOp {
    /// Static string name (no allocation).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::BesselJ => "besselJ",
            Self::BesselY => "besselY",
            Self::Polylog => "polylog",
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Expression tree node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExprNode {
    // Leaves
    Var(VarId),
    Param(ParamId),
    Num(i64),
    Rational(i64, i64),
    Const(MathConst),
    // Compound
    Unary(UnaryOp, Box<ExprNode>),
    Binary(BinaryOp, Box<ExprNode>, Box<ExprNode>),
    Hyp2F1(Box<ExprNode>, Box<ExprNode>, Box<ExprNode>, Box<ExprNode>),
}

// ---------------------------------------------------------------------------
// Helper constructors (keep expressions readable)
// ---------------------------------------------------------------------------
impl ExprNode {
    pub fn num(n: i64) -> Self {
        Self::Num(n)
    }
    pub fn var(s: &str) -> Self {
        Self::Var(VarId::from_str(s).unwrap_or_else(|| panic!("unknown var: {s}")))
    }
    pub fn param(s: &str) -> Self {
        Self::Param(ParamId::from_str(s).unwrap_or_else(|| panic!("unknown param: {s}")))
    }
    pub fn add(a: Self, b: Self) -> Self {
        Self::Binary(BinaryOp::Add, Box::new(a), Box::new(b))
    }
    pub fn sub(a: Self, b: Self) -> Self {
        Self::Binary(BinaryOp::Sub, Box::new(a), Box::new(b))
    }
    pub fn mul(a: Self, b: Self) -> Self {
        Self::Binary(BinaryOp::Mul, Box::new(a), Box::new(b))
    }
    pub fn div(a: Self, b: Self) -> Self {
        Self::Binary(BinaryOp::Div, Box::new(a), Box::new(b))
    }
    pub fn pow(a: Self, b: Self) -> Self {
        Self::Binary(BinaryOp::Pow, Box::new(a), Box::new(b))
    }
    pub fn neg(a: Self) -> Self {
        Self::Unary(UnaryOp::Neg, Box::new(a))
    }
    pub fn sin(a: Self) -> Self {
        Self::Unary(UnaryOp::Sin, Box::new(a))
    }
    pub fn cos(a: Self) -> Self {
        Self::Unary(UnaryOp::Cos, Box::new(a))
    }
    pub fn exp(a: Self) -> Self {
        Self::Unary(UnaryOp::Exp, Box::new(a))
    }
    pub fn log(a: Self) -> Self {
        Self::Unary(UnaryOp::Log, Box::new(a))
    }
    pub fn sqrt(a: Self) -> Self {
        Self::Unary(UnaryOp::Sqrt, Box::new(a))
    }
}

// ---------------------------------------------------------------------------
// Tree traversal helpers
// ---------------------------------------------------------------------------
impl ExprNode {
    #[inline]
    pub fn depth(&self) -> usize {
        match self {
            Self::Var(_) | Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => 1,
            Self::Unary(_, c) => 1 + c.depth(),
            Self::Binary(_, l, r) => 1 + l.depth().max(r.depth()),
            Self::Hyp2F1(a, b, c, d) => {
                1 + a.depth().max(b.depth()).max(c.depth()).max(d.depth())
            }
        }
    }

    #[inline]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Var(_) | Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => 1,
            Self::Unary(_, c) => 1 + c.node_count(),
            Self::Binary(_, l, r) => 1 + l.node_count() + r.node_count(),
            Self::Hyp2F1(a, b, c, d) => {
                1 + a.node_count() + b.node_count()
                  + c.node_count() + d.node_count()
            }
        }
    }

    #[inline]
    pub fn contains_var(&self, var: VarId) -> bool {
        match self {
            Self::Var(v) => *v == var,
            Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => false,
            Self::Unary(_, c) => c.contains_var(var),
            Self::Binary(_, l, r) => {
                l.contains_var(var) || r.contains_var(var)
            }
            Self::Hyp2F1(a, b, c, d) => {
                a.contains_var(var) || b.contains_var(var)
                || c.contains_var(var) || d.contains_var(var)
            }
        }
    }

    /// String-based contains_var for backward compatibility.
    #[inline]
    pub fn contains_var_str(&self, var: &str) -> bool {
        VarId::from_str(var).map_or(false, |v| self.contains_var(v))
    }

    pub fn free_vars(&self) -> HashSet<VarId> {
        let mut out = HashSet::new();
        self.collect_free_vars(&mut out);
        out
    }

    fn collect_free_vars(&self, out: &mut HashSet<VarId>) {
        match self {
            Self::Var(v) => { out.insert(*v); }
            Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => {}
            Self::Unary(_, c) => c.collect_free_vars(out),
            Self::Binary(_, l, r) => {
                l.collect_free_vars(out);
                r.collect_free_vars(out);
            }
            Self::Hyp2F1(a, b, c, d) => {
                a.collect_free_vars(out);
                b.collect_free_vars(out);
                c.collect_free_vars(out);
                d.collect_free_vars(out);
            }
        }
    }

    pub fn free_params(&self) -> HashSet<ParamId> {
        let mut out = HashSet::new();
        self.collect_free_params(&mut out);
        out
    }

    fn collect_free_params(&self, out: &mut HashSet<ParamId>) {
        match self {
            Self::Param(p) => { out.insert(*p); }
            Self::Var(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => {}
            Self::Unary(_, c) => c.collect_free_params(out),
            Self::Binary(_, l, r) => {
                l.collect_free_params(out);
                r.collect_free_params(out);
            }
            Self::Hyp2F1(a, b, c, d) => {
                a.collect_free_params(out);
                b.collect_free_params(out);
                c.collect_free_params(out);
                d.collect_free_params(out);
            }
        }
    }

    pub fn children(&self) -> Vec<&ExprNode> {
        match self {
            Self::Var(_) | Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => vec![],
            Self::Unary(_, c) => vec![c],
            Self::Binary(_, l, r) => vec![l, r],
            Self::Hyp2F1(a, b, c, d) => vec![a, b, c, d],
        }
    }

    #[inline]
    pub fn arity(&self) -> usize {
        match self {
            Self::Var(_) | Self::Param(_) | Self::Num(_)
            | Self::Rational(_, _) | Self::Const(_) => 0,
            Self::Unary(_, _) => 1,
            Self::Binary(_, _, _) => 2,
            Self::Hyp2F1(_, _, _, _) => 4,
        }
    }

    /// Check whether a node is a numeric zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Num(0) | Self::Rational(0, _))
    }

    /// Check whether a node is a numeric one.
    #[inline]
    pub fn is_one(&self) -> bool {
        match self {
            Self::Num(1) => true,
            Self::Rational(n, d) => n == d && *d != 0,
            _ => false,
        }
    }
}

impl fmt::Display for ExprNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", to_prefix_string(self))
    }
}

// ---------------------------------------------------------------------------
// PyO3 wrapper
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::PyExprTree;

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use super::*;

    #[pyclass(name = "ExprTree")]
    #[derive(Clone, Debug)]
    pub struct PyExprTree {
        pub inner: ExprNode,
    }

    #[pymethods]
    impl PyExprTree {
        #[new]
        fn new(prefix: &str) -> PyResult<Self> {
            let inner = from_prefix_string(prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(Self { inner })
        }

        fn depth(&self) -> usize {
            self.inner.depth()
        }
        fn node_count(&self) -> usize {
            self.inner.node_count()
        }
        fn contains_var(&self, var: &str) -> bool {
            self.inner.contains_var_str(var)
        }
        fn free_vars(&self) -> Vec<String> {
            self.inner.free_vars().into_iter().map(|v| v.as_str().to_owned()).collect()
        }
        fn free_params(&self) -> Vec<String> {
            self.inner.free_params().into_iter().map(|p| p.as_str().to_owned()).collect()
        }
        fn arity(&self) -> usize {
            self.inner.arity()
        }
        fn to_prefix(&self) -> String {
            to_prefix_string(&self.inner)
        }
        fn to_json(&self) -> PyResult<String> {
            serde_json::to_string(&self.inner)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }
        fn __repr__(&self) -> String {
            format!("ExprTree({})", to_prefix_string(&self.inner))
        }
        fn __str__(&self) -> String {
            to_prefix_string(&self.inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_expr() -> ExprNode {
        // sin(x) + 2*y
        ExprNode::add(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::mul(ExprNode::num(2), ExprNode::var("y")),
        )
    }

    #[test]
    fn test_depth() {
        assert_eq!(sample_expr().depth(), 3);
        assert_eq!(ExprNode::num(5).depth(), 1);
    }

    #[test]
    fn test_node_count() {
        assert_eq!(sample_expr().node_count(), 6);
    }

    #[test]
    fn test_contains_var() {
        let e = sample_expr();
        assert!(e.contains_var(VarId::X));
        assert!(e.contains_var(VarId::Y));
        assert!(!e.contains_var(VarId::Z));
    }

    #[test]
    fn test_free_vars() {
        let vars = sample_expr().free_vars();
        assert!(vars.contains(&VarId::X));
        assert!(vars.contains(&VarId::Y));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_free_params() {
        let e = ExprNode::add(ExprNode::param("a"), ExprNode::var("x"));
        assert!(e.free_params().contains(&ParamId::A));
        assert!(e.free_params().is_disjoint(&HashSet::from([ParamId::B])));
    }

    #[test]
    fn test_children_arity() {
        let e = sample_expr();
        assert_eq!(e.arity(), 2);
        assert_eq!(e.children().len(), 2);
        assert_eq!(ExprNode::num(1).arity(), 0);
    }

    #[test]
    fn test_is_zero_one() {
        assert!(ExprNode::num(0).is_zero());
        assert!(!ExprNode::num(0).is_one());
        assert!(ExprNode::num(1).is_one());
        assert!(ExprNode::Rational(3, 3).is_one());
    }
}
