//! Coverage-guaranteed expression generation.
//!
//! Enumerates all structural families of integration-relevant expressions
//! and ensures every family is represented in the training distribution.
//! Addresses the under-representation problem where random generation
//! exponentially under-samples rare but important structures.

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::expr::{ExprNode, UnaryOp, VarId};
use crate::gen::GenConfig;

/// A skeleton is a structural template with holes for concrete expressions.
/// Each skeleton represents a family of integrands.
#[derive(Clone, Debug)]
pub enum Skeleton {
    /// f(x) — single elementary function applied to variable
    Elementary(UnaryOp),
    /// f(g(x)) — composition of two elementary functions
    Composition(UnaryOp, UnaryOp),
    /// f(ax+b) — linear substitution pattern
    LinearSub(UnaryOp),
    /// f(g(x)) * g'(x) — u-substitution pattern (chain rule reverse)
    USub(UnaryOp, UnaryOp),
    /// u * dv pattern (integration by parts target)
    IBP(IBPTemplate),
    /// p(x)/q(x) — rational function (partial fractions target)
    Rational(usize, usize),
    /// Product of trig functions: sin^m(x) * cos^n(x)
    TrigProduct(u32, u32),
    /// Trig substitution pattern: f(x) / sqrt(a² ± x²)
    TrigSub(TrigSubType),
    /// Power rule: x^n
    PowerRule(i64),
    /// Exponential: e^(f(x)) * f'(x)
    ExpChain(UnaryOp),
    /// Log derivative: f'(x)/f(x)
    LogDerivative,
    /// Hyperbolic compositions
    HyperbolicComp(UnaryOp),
    /// Nested composition depth 3: f(g(h(x)))
    TripleComposition(UnaryOp, UnaryOp, UnaryOp),
    /// Product with polynomial: x^n * f(x)
    PolyProduct(u32, UnaryOp),
    /// Special function applied to polynomial
    SpecialFn(UnaryOp),
}

/// Templates for integration-by-parts targets.
#[derive(Clone, Debug)]
pub enum IBPTemplate {
    /// x^n * e^x
    PolyExp(u32),
    /// x^n * sin(x) or x^n * cos(x)
    PolyTrig(u32, UnaryOp),
    /// x^n * log(x)
    PolyLog(u32),
    /// e^x * sin(x) or e^x * cos(x)
    ExpTrig(UnaryOp),
    /// log(x)^n
    LogPower(u32),
    /// arctan(x), arcsin(x) etc — IBP with inverse trig
    InverseTrig(UnaryOp),
}

/// Types of trig substitution.
#[derive(Clone, Debug)]
pub enum TrigSubType {
    /// sqrt(a² - x²) → x = a*sin(θ)
    SinSub,
    /// sqrt(a² + x²) → x = a*tan(θ)
    TanSub,
    /// sqrt(x² - a²) → x = a*sec(θ)
    SecSub,
}

/// Enumerate all skeleton families for coverage tracking.
pub fn enumerate_skeletons() -> Vec<Skeleton> {
    let mut skeletons = Vec::new();

    let elementary_ops = [
        UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Tan,
        UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sqrt,
        UnaryOp::Asin, UnaryOp::Acos, UnaryOp::Atan,
        UnaryOp::Sinh, UnaryOp::Cosh, UnaryOp::Tanh,
    ];

    let _trig_ops = [UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Tan];
    let hyp_ops = [UnaryOp::Sinh, UnaryOp::Cosh, UnaryOp::Tanh];

    // 1. Elementary: f(x) for each f
    for &op in &elementary_ops {
        skeletons.push(Skeleton::Elementary(op));
    }

    // 2. Compositions: f(g(x)) for common pairs
    let inner_ops = [
        UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp,
        UnaryOp::Log, UnaryOp::Sqrt,
    ];
    for &outer in &elementary_ops {
        for &inner in &inner_ops {
            skeletons.push(Skeleton::Composition(outer, inner));
        }
    }

    // 3. Linear substitution: f(ax+b)
    for &op in &elementary_ops {
        skeletons.push(Skeleton::LinearSub(op));
    }

    // 4. U-substitution: f(g(x)) * g'(x)
    for &outer in &[UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sqrt] {
        for &inner in &[UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp, UnaryOp::Log] {
            skeletons.push(Skeleton::USub(outer, inner));
        }
    }

    // 5. IBP templates
    for n in 1..=4 {
        skeletons.push(Skeleton::IBP(IBPTemplate::PolyExp(n)));
        skeletons.push(Skeleton::IBP(IBPTemplate::PolyTrig(n, UnaryOp::Sin)));
        skeletons.push(Skeleton::IBP(IBPTemplate::PolyTrig(n, UnaryOp::Cos)));
        skeletons.push(Skeleton::IBP(IBPTemplate::PolyLog(n)));
    }
    skeletons.push(Skeleton::IBP(IBPTemplate::ExpTrig(UnaryOp::Sin)));
    skeletons.push(Skeleton::IBP(IBPTemplate::ExpTrig(UnaryOp::Cos)));
    for n in 1..=3 {
        skeletons.push(Skeleton::IBP(IBPTemplate::LogPower(n)));
    }
    for &op in &[UnaryOp::Asin, UnaryOp::Acos, UnaryOp::Atan] {
        skeletons.push(Skeleton::IBP(IBPTemplate::InverseTrig(op)));
    }

    // 6. Rational functions: deg(p)/deg(q) for small degrees
    for p_deg in 0..=4 {
        for q_deg in 1..=4 {
            skeletons.push(Skeleton::Rational(p_deg, q_deg));
        }
    }

    // 7. Trig products: sin^m * cos^n
    for m in 0..=4 {
        for n in 0..=4 {
            if m + n > 0 && m + n <= 6 {
                skeletons.push(Skeleton::TrigProduct(m, n));
            }
        }
    }

    // 8. Trig substitution patterns
    skeletons.push(Skeleton::TrigSub(TrigSubType::SinSub));
    skeletons.push(Skeleton::TrigSub(TrigSubType::TanSub));
    skeletons.push(Skeleton::TrigSub(TrigSubType::SecSub));

    // 9. Power rule: x^n for various n (including negative and fractional)
    for n in -5..=10 {
        if n != -1 {
            skeletons.push(Skeleton::PowerRule(n));
        }
    }
    skeletons.push(Skeleton::PowerRule(-1)); // special: yields log

    // 10. Exponential chain: e^(f(x)) * f'(x)
    for &op in &[UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Log, UnaryOp::Sqrt] {
        skeletons.push(Skeleton::ExpChain(op));
    }

    // 11. Log derivative: f'(x)/f(x)
    skeletons.push(Skeleton::LogDerivative);

    // 12. Hyperbolic compositions
    for &op in &hyp_ops {
        skeletons.push(Skeleton::HyperbolicComp(op));
    }

    // 13. Triple compositions: f(g(h(x)))
    let triple_fns = [UnaryOp::Sin, UnaryOp::Exp, UnaryOp::Log];
    for &f in &triple_fns {
        for &g in &triple_fns {
            for &h in &triple_fns {
                skeletons.push(Skeleton::TripleComposition(f, g, h));
            }
        }
    }

    // 14. Polynomial products: x^n * f(x)
    for n in 1..=3 {
        for &op in &elementary_ops {
            skeletons.push(Skeleton::PolyProduct(n, op));
        }
    }

    // 15. Special functions
    let special_ops = [
        UnaryOp::Erf, UnaryOp::Ei, UnaryOp::Si, UnaryOp::Ci,
        UnaryOp::Li, UnaryOp::Gamma, UnaryOp::Digamma,
        UnaryOp::FresnelS, UnaryOp::FresnelC,
        UnaryOp::EllipticK, UnaryOp::EllipticE,
    ];
    for &op in &special_ops {
        skeletons.push(Skeleton::SpecialFn(op));
    }

    skeletons
}

/// Instantiate a skeleton into a concrete ExprNode (the antiderivative).
/// The antiderivative is differentiated externally to get the integrand.
pub fn instantiate_skeleton(
    skeleton: &Skeleton,
    config: &GenConfig,
    rng: &mut impl Rng,
) -> ExprNode {
    let x = ExprNode::Var(config.vars.first().copied().unwrap_or(VarId::X));

    match skeleton {
        Skeleton::Elementary(op) => {
            ExprNode::Unary(*op, Box::new(x))
        }

        Skeleton::Composition(outer, inner) => {
            let inner_expr = ExprNode::Unary(*inner, Box::new(x));
            ExprNode::Unary(*outer, Box::new(inner_expr))
        }

        Skeleton::LinearSub(op) => {
            let a = random_nonzero_coeff(rng);
            let b = random_coeff(rng);
            let linear = ExprNode::add(
                ExprNode::mul(ExprNode::Num(a), x),
                ExprNode::Num(b),
            );
            ExprNode::Unary(*op, Box::new(linear))
        }

        Skeleton::USub(outer, inner) => {
            // Antiderivative of f(g(x)) * g'(x) is F(g(x))
            // So we generate F(g(x)) directly as the antiderivative
            let g = ExprNode::Unary(*inner, Box::new(x));
            ExprNode::Unary(*outer, Box::new(g))
        }

        Skeleton::IBP(template) => instantiate_ibp(template, &x, rng),

        Skeleton::Rational(p_deg, q_deg) => {
            let p = random_polynomial(&x, *p_deg, rng);
            let q = random_polynomial_nonzero(&x, *q_deg, rng);
            // Antiderivative: we generate log(q) * something or atan patterns
            // Simplest: just generate p/q and let differentiation handle it
            ExprNode::div(p, q)
        }

        Skeleton::TrigProduct(m, n) => {
            // Antiderivative involves sin^(m+1) * cos^(n-1) patterns
            let sin_part = if *m > 0 {
                ExprNode::pow(ExprNode::sin(x.clone()), ExprNode::Num(*m as i64))
            } else {
                ExprNode::Num(1)
            };
            let cos_part = if *n > 0 {
                ExprNode::pow(ExprNode::cos(x.clone()), ExprNode::Num(*n as i64))
            } else {
                ExprNode::Num(1)
            };
            ExprNode::mul(sin_part, cos_part)
        }

        Skeleton::TrigSub(sub_type) => {
            let a = random_small_positive(rng);
            match sub_type {
                TrigSubType::SinSub => {
                    // x * sqrt(a² - x²) pattern
                    let a2 = ExprNode::Num(a * a);
                    let x2 = ExprNode::pow(x.clone(), ExprNode::Num(2));
                    ExprNode::sqrt(ExprNode::sub(a2, x2))
                }
                TrigSubType::TanSub => {
                    let a2 = ExprNode::Num(a * a);
                    let x2 = ExprNode::pow(x.clone(), ExprNode::Num(2));
                    ExprNode::sqrt(ExprNode::add(a2, x2))
                }
                TrigSubType::SecSub => {
                    let a2 = ExprNode::Num(a * a);
                    let x2 = ExprNode::pow(x.clone(), ExprNode::Num(2));
                    ExprNode::sqrt(ExprNode::sub(x2, a2))
                }
            }
        }

        Skeleton::PowerRule(n) => {
            // Antiderivative of x^n is x^(n+1)/(n+1)
            ExprNode::pow(x, ExprNode::Num(*n + 1))
        }

        Skeleton::ExpChain(inner_op) => {
            // Antiderivative: e^(f(x))
            let f = ExprNode::Unary(*inner_op, Box::new(x));
            ExprNode::exp(f)
        }

        Skeleton::LogDerivative => {
            // Antiderivative of f'(x)/f(x) is log(f(x))
            // Generate log(random polynomial)
            let poly = random_polynomial(&x, rng.gen_range(1..=3), rng);
            ExprNode::log(poly)
        }

        Skeleton::HyperbolicComp(op) => {
            let a = random_nonzero_coeff(rng);
            let linear = ExprNode::mul(ExprNode::Num(a), x);
            ExprNode::Unary(*op, Box::new(linear))
        }

        Skeleton::TripleComposition(f, g, h) => {
            let h_x = ExprNode::Unary(*h, Box::new(x));
            let g_h = ExprNode::Unary(*g, Box::new(h_x));
            ExprNode::Unary(*f, Box::new(g_h))
        }

        Skeleton::PolyProduct(n, op) => {
            let x_n = ExprNode::pow(x.clone(), ExprNode::Num(*n as i64));
            let f_x = ExprNode::Unary(*op, Box::new(x.clone()));
            ExprNode::mul(x_n, f_x)
        }

        Skeleton::SpecialFn(op) => {
            let a = random_nonzero_coeff(rng);
            let inner = ExprNode::mul(ExprNode::Num(a), x);
            ExprNode::Unary(*op, Box::new(inner))
        }
    }
}

/// Generate a batch with guaranteed skeleton coverage.
/// Distributes samples uniformly across all skeleton families,
/// then fills remaining budget with random generation for diversity.
pub fn generate_covered_batch(
    total: usize,
    config: &GenConfig,
) -> Vec<ExprNode> {
    let skeletons = enumerate_skeletons();
    let n_families = skeletons.len();

    // Allocate: 70% to skeleton coverage, 30% to random exploration
    let covered_budget = (total * 70) / 100;
    let random_budget = total - covered_budget;
    let per_family = std::cmp::max(covered_budget / n_families, 1);

    let mut results = Vec::with_capacity(total);
    let mut rng = rand::thread_rng();

    // Phase 1: Generate per_family instances from each skeleton
    for skeleton in &skeletons {
        for _ in 0..per_family {
            let expr = instantiate_skeleton(skeleton, config, &mut rng);
            results.push(expr);
        }
    }

    // Phase 2: Random generation for diversity (uses existing gen_node)
    for _ in 0..random_budget {
        let expr = crate::gen::generate_random_tree(config, &mut rng);
        results.push(expr);
    }

    // Shuffle to avoid ordering effects during training
    results.shuffle(&mut rng);
    results.truncate(total);
    results
}

/// Coverage statistics for a generated batch.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CoverageStats {
    pub total_skeletons: usize,
    pub total_generated: usize,
    pub per_family_target: usize,
    pub skeleton_fraction: f64,
    pub random_fraction: f64,
}

pub fn coverage_stats(total: usize) -> CoverageStats {
    let skeletons = enumerate_skeletons();
    let n_families = skeletons.len();
    let covered_budget = (total * 70) / 100;
    let per_family = std::cmp::max(covered_budget / n_families, 1);

    CoverageStats {
        total_skeletons: n_families,
        total_generated: total,
        per_family_target: per_family,
        skeleton_fraction: 0.70,
        random_fraction: 0.30,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_nonzero_coeff(rng: &mut impl Rng) -> i64 {
    let n = rng.gen_range(1..=5_i64);
    if rng.gen_bool(0.5) { n } else { -n }
}

fn random_coeff(rng: &mut impl Rng) -> i64 {
    rng.gen_range(-5..=5)
}

fn random_small_positive(rng: &mut impl Rng) -> i64 {
    rng.gen_range(1..=5)
}

fn random_polynomial(x: &ExprNode, degree: usize, rng: &mut impl Rng) -> ExprNode {
    if degree == 0 {
        return ExprNode::Num(random_nonzero_coeff(rng));
    }

    let mut terms: Vec<ExprNode> = Vec::new();

    // Leading term (nonzero coefficient)
    let lead_coeff = random_nonzero_coeff(rng);
    let lead = ExprNode::mul(
        ExprNode::Num(lead_coeff),
        ExprNode::pow(x.clone(), ExprNode::Num(degree as i64)),
    );
    terms.push(lead);

    // Lower-degree terms (each with 70% probability)
    for d in (0..degree).rev() {
        if rng.gen_bool(0.7) {
            let c = random_coeff(rng);
            if c == 0 {
                continue;
            }
            if d == 0 {
                terms.push(ExprNode::Num(c));
            } else if d == 1 {
                terms.push(ExprNode::mul(ExprNode::Num(c), x.clone()));
            } else {
                terms.push(ExprNode::mul(
                    ExprNode::Num(c),
                    ExprNode::pow(x.clone(), ExprNode::Num(d as i64)),
                ));
            }
        }
    }

    // Sum all terms
    let mut result = terms.pop().unwrap();
    while let Some(term) = terms.pop() {
        result = ExprNode::add(term, result);
    }
    result
}

fn random_polynomial_nonzero(x: &ExprNode, degree: usize, rng: &mut impl Rng) -> ExprNode {
    // Ensure denominator isn't trivially zero by adding a positive constant
    let poly = random_polynomial(x, degree, rng);
    ExprNode::add(poly, ExprNode::Num(random_small_positive(rng)))
}

fn instantiate_ibp(template: &IBPTemplate, x: &ExprNode, _rng: &mut impl Rng) -> ExprNode {
    match template {
        IBPTemplate::PolyExp(n) => {
            // Antiderivative of x^n * e^x involves x^n * e^x - n*x^(n-1)*e^x + ...
            // Generate: x^n * e^x (differentiation produces the IBP integrand)
            ExprNode::mul(
                ExprNode::pow(x.clone(), ExprNode::Num(*n as i64)),
                ExprNode::exp(x.clone()),
            )
        }
        IBPTemplate::PolyTrig(n, trig_op) => {
            ExprNode::mul(
                ExprNode::pow(x.clone(), ExprNode::Num(*n as i64)),
                ExprNode::Unary(*trig_op, Box::new(x.clone())),
            )
        }
        IBPTemplate::PolyLog(n) => {
            ExprNode::mul(
                ExprNode::pow(x.clone(), ExprNode::Num(*n as i64)),
                ExprNode::log(x.clone()),
            )
        }
        IBPTemplate::ExpTrig(trig_op) => {
            ExprNode::mul(
                ExprNode::exp(x.clone()),
                ExprNode::Unary(*trig_op, Box::new(x.clone())),
            )
        }
        IBPTemplate::LogPower(n) => {
            ExprNode::pow(
                ExprNode::log(x.clone()),
                ExprNode::Num(*n as i64),
            )
        }
        IBPTemplate::InverseTrig(op) => {
            ExprNode::Unary(*op, Box::new(x.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_skeletons_nonempty() {
        let skeletons = enumerate_skeletons();
        assert!(skeletons.len() > 200);
    }

    #[test]
    fn instantiate_all_skeletons() {
        let skeletons = enumerate_skeletons();
        let config = GenConfig::default();
        let mut rng = rand::thread_rng();

        for skeleton in &skeletons {
            let expr = instantiate_skeleton(skeleton, &config, &mut rng);
            assert!(expr.node_count() >= 1);
        }
    }

    #[test]
    fn covered_batch_produces_expected_count() {
        let config = GenConfig::default();
        let batch = generate_covered_batch(1000, &config);
        assert_eq!(batch.len(), 1000);
    }

    #[test]
    fn coverage_stats_reports_families() {
        let stats = coverage_stats(10000);
        assert!(stats.total_skeletons > 200);
        assert!(stats.per_family_target >= 1);
    }
}
