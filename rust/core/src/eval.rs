use crate::expr::{BinaryOp, ExprNode, MathConst, UnaryOp, VarId, ParamId};

/// Error during numerical evaluation.
#[derive(Debug, Clone)]
pub enum EvalError {
    DivisionByZero,
    DomainError(&'static str),
    NotFinite,
    UnboundVariable(VarId),
    UnboundParam(ParamId),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::DomainError(msg) => write!(f, "domain error: {msg}"),
            Self::NotFinite => write!(f, "result not finite"),
            Self::UnboundVariable(v) => write!(f, "unbound variable: {v}"),
            Self::UnboundParam(p) => write!(f, "unbound parameter: {p}"),
        }
    }
}

/// Bindings for variable and parameter values.
pub struct Bindings {
    vars: [Option<f64>; 5],   // indexed by VarId
    params: [Option<f64>; 7], // indexed by ParamId
}

impl Bindings {
    pub fn new() -> Self {
        Self {
            vars: [None; 5],
            params: [None; 7],
        }
    }

    pub fn set_var(&mut self, v: VarId, val: f64) -> &mut Self {
        self.vars[v as usize] = Some(val);
        self
    }

    pub fn set_param(&mut self, p: ParamId, val: f64) -> &mut Self {
        self.params[p as usize] = Some(val);
        self
    }

    fn get_var(&self, v: VarId) -> Option<f64> {
        self.vars[v as usize]
    }

    fn get_param(&self, p: ParamId) -> Option<f64> {
        self.params[p as usize]
    }
}

impl Default for Bindings {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate an expression tree at given variable/parameter bindings.
pub fn evaluate(expr: &ExprNode, bindings: &Bindings) -> Result<f64, EvalError> {
    let val = eval_inner(expr, bindings)?;
    if val.is_finite() {
        Ok(val)
    } else {
        Err(EvalError::NotFinite)
    }
}

fn eval_inner(expr: &ExprNode, b: &Bindings) -> Result<f64, EvalError> {
    match expr {
        ExprNode::Var(v) => b.get_var(*v).ok_or(EvalError::UnboundVariable(*v)),
        ExprNode::Param(p) => b.get_param(*p).ok_or(EvalError::UnboundParam(*p)),
        ExprNode::Num(n) => Ok(*n as f64),
        ExprNode::Rational(n, d) => {
            if *d == 0 {
                return Err(EvalError::DivisionByZero);
            }
            Ok(*n as f64 / *d as f64)
        }
        ExprNode::Const(c) => Ok(match c {
            MathConst::Pi => std::f64::consts::PI,
            MathConst::EulerGamma => 0.5772156649015329,
            MathConst::Catalan => 0.9159655941772190,
        }),

        ExprNode::Unary(op, inner) => {
            let x = eval_inner(inner, b)?;
            eval_unary(*op, x)
        }
        ExprNode::Binary(op, l, r) => {
            let lv = eval_inner(l, b)?;
            let rv = eval_inner(r, b)?;
            eval_binary(*op, lv, rv)
        }
        ExprNode::Hyp2F1(..) => Err(EvalError::DomainError("hyp2f1 not supported")),
    }
}

#[inline]
fn eval_unary(op: UnaryOp, x: f64) -> Result<f64, EvalError> {
    let v = match op {
        UnaryOp::Neg => -x,
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tan => x.tan(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => {
            if x <= 0.0 { return Err(EvalError::DomainError("log of non-positive")); }
            x.ln()
        }
        UnaryOp::Sqrt => {
            if x < 0.0 { return Err(EvalError::DomainError("sqrt of negative")); }
            x.sqrt()
        }
        UnaryOp::Asin => {
            if x.abs() > 1.0 { return Err(EvalError::DomainError("asin domain")); }
            x.asin()
        }
        UnaryOp::Acos => {
            if x.abs() > 1.0 { return Err(EvalError::DomainError("acos domain")); }
            x.acos()
        }
        UnaryOp::Atan => x.atan(),
        UnaryOp::Sinh => x.sinh(),
        UnaryOp::Cosh => x.cosh(),
        UnaryOp::Tanh => x.tanh(),
        // Special functions — approximate implementations
        UnaryOp::Erf => erf_approx(x),
        UnaryOp::Gamma => {
            if x <= 0.0 && x.fract() == 0.0 {
                return Err(EvalError::DomainError("gamma at non-positive integer"));
            }
            gamma_approx(x)
        }
        // For functions we can't easily evaluate, return error
        _ => return Err(EvalError::DomainError("unsupported special function")),
    };
    if v.is_finite() { Ok(v) } else { Err(EvalError::NotFinite) }
}

#[inline]
fn eval_binary(op: BinaryOp, l: f64, r: f64) -> Result<f64, EvalError> {
    let v = match op {
        BinaryOp::Add => l + r,
        BinaryOp::Sub => l - r,
        BinaryOp::Mul => l * r,
        BinaryOp::Div => {
            if r == 0.0 { return Err(EvalError::DivisionByZero); }
            l / r
        }
        BinaryOp::Pow => l.powf(r),
        _ => return Err(EvalError::DomainError("unsupported binary special function")),
    };
    if v.is_finite() { Ok(v) } else { Err(EvalError::NotFinite) }
}

// ---------------------------------------------------------------------------
// Approximations for special functions
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun approximation for erf(x), max error ~1.5e-7.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
            + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Lanczos approximation for gamma(x).
fn gamma_approx(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        let sin_val = (std::f64::consts::PI * x).sin();
        if sin_val.abs() < 1e-15 {
            return f64::INFINITY;
        }
        std::f64::consts::PI / (sin_val * gamma_approx(1.0 - x))
    } else {
        let g = 7.0;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        let x = x - 1.0;
        let mut sum = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            sum += ci / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind_x(val: f64) -> Bindings {
        let mut b = Bindings::new();
        b.set_var(VarId::X, val);
        b
    }

    #[test]
    fn eval_num() {
        let e = ExprNode::num(42);
        assert_eq!(evaluate(&e, &Bindings::new()).unwrap(), 42.0);
    }

    #[test]
    fn eval_var() {
        let e = ExprNode::var("x");
        let b = bind_x(3.0);
        assert_eq!(evaluate(&e, &b).unwrap(), 3.0);
    }

    #[test]
    fn eval_add() {
        let e = ExprNode::add(ExprNode::var("x"), ExprNode::num(2));
        assert!((evaluate(&e, &bind_x(3.0)).unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eval_sin() {
        let e = ExprNode::sin(ExprNode::var("x"));
        let val = evaluate(&e, &bind_x(std::f64::consts::PI / 2.0)).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn eval_exp_log() {
        let e = ExprNode::log(ExprNode::exp(ExprNode::var("x")));
        let val = evaluate(&e, &bind_x(2.5)).unwrap();
        assert!((val - 2.5).abs() < 1e-10);
    }

    #[test]
    fn eval_pow() {
        let e = ExprNode::pow(ExprNode::var("x"), ExprNode::num(3));
        let val = evaluate(&e, &bind_x(2.0)).unwrap();
        assert!((val - 8.0).abs() < 1e-10);
    }

    #[test]
    fn eval_div_by_zero() {
        let e = ExprNode::div(ExprNode::num(1), ExprNode::num(0));
        assert!(evaluate(&e, &Bindings::new()).is_err());
    }

    #[test]
    fn eval_sqrt_negative() {
        let e = ExprNode::sqrt(ExprNode::num(-1));
        assert!(evaluate(&e, &Bindings::new()).is_err());
    }

    #[test]
    fn eval_rational() {
        let e = ExprNode::Rational(1, 3);
        let val = evaluate(&e, &Bindings::new()).unwrap();
        assert!((val - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn eval_const_pi() {
        let e = ExprNode::Const(MathConst::Pi);
        let val = evaluate(&e, &Bindings::new()).unwrap();
        assert!((val - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn eval_erf() {
        let e = ExprNode::Unary(UnaryOp::Erf, Box::new(ExprNode::num(1)));
        let val = evaluate(&e, &Bindings::new()).unwrap();
        // erf(1) ≈ 0.8427
        assert!((val - 0.8427007929).abs() < 1e-6);
    }

    #[test]
    fn eval_gamma() {
        let e = ExprNode::Unary(UnaryOp::Gamma, Box::new(ExprNode::num(5)));
        let val = evaluate(&e, &Bindings::new()).unwrap();
        // Gamma(5) = 4! = 24
        assert!((val - 24.0).abs() < 1e-6);
    }

    #[test]
    fn eval_complex_expr() {
        // sin(x)^2 + cos(x)^2 = 1
        let e = ExprNode::add(
            ExprNode::pow(ExprNode::sin(ExprNode::var("x")), ExprNode::num(2)),
            ExprNode::pow(ExprNode::cos(ExprNode::var("x")), ExprNode::num(2)),
        );
        let val = evaluate(&e, &bind_x(1.7)).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }
}
