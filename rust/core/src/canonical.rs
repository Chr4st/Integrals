//! Expression canonicalization via e-graph equality saturation.
//!
//! Uses the `egg` crate to define rewrite rules for algebraic identities,
//! then saturates to find a canonical form for deduplication.

use egg::{define_language, rewrite, Id, Language, RecExpr, Runner, Symbol};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

define_language! {
    /// E-graph language mirroring ExprNode operators.
    pub enum MathLang {
        // Leaves
        "var" = Var(Symbol),
        "param" = Param(Symbol),
        "num" = Num(i64),

        // Unary
        "neg" = Neg([Id; 1]),
        "sin" = Sin([Id; 1]),
        "cos" = Cos([Id; 1]),
        "tan" = Tan([Id; 1]),
        "exp" = Exp([Id; 1]),
        "log" = Log([Id; 1]),
        "sqrt" = Sqrt([Id; 1]),

        // Binary
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "pow" = Pow([Id; 2]),
    }
}

/// Core rewrite rules for algebraic canonicalization.
pub fn rewrite_rules() -> Vec<egg::Rewrite<MathLang, ()>> {
    vec![
        // Commutativity
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)"),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)"),

        // Associativity
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))"),
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))"),

        // Identity
        rewrite!("add-zero"; "(add ?a (num 0))" => "?a"),
        rewrite!("mul-one"; "(mul ?a (num 1))" => "?a"),
        rewrite!("pow-one"; "(pow ?a (num 1))" => "?a"),
        rewrite!("pow-zero"; "(pow ?a (num 0))" => "(num 1)"),

        // Inverse
        rewrite!("sub-self"; "(sub ?a ?a)" => "(num 0)"),
        rewrite!("div-self"; "(div ?a ?a)" => "(num 1)"),
        rewrite!("neg-neg"; "(neg (neg ?a))" => "?a"),

        // Distributivity
        rewrite!("distribute"; "(mul ?a (add ?b ?c))" => "(add (mul ?a ?b) (mul ?a ?c))"),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))"),

        // Log/Exp
        rewrite!("exp-log"; "(exp (log ?a))" => "?a"),
        rewrite!("log-exp"; "(log (exp ?a))" => "?a"),
        rewrite!("log-mul"; "(log (mul ?a ?b))" => "(add (log ?a) (log ?b))"),
        rewrite!("log-pow"; "(log (pow ?a ?b))" => "(mul ?b (log ?a))"),

        // Trig: Pythagorean identity (one direction for canonicalization)
        rewrite!("sin2-cos2"; "(add (pow (sin ?a) (num 2)) (pow (cos ?a) (num 2)))" => "(num 1)"),

        // Double neg as sub
        rewrite!("add-neg"; "(add ?a (neg ?b))" => "(sub ?a ?b)"),
    ]
}

/// Canonicalize a prefix-notation expression string.
///
/// Returns the smallest (by node count) equivalent expression found
/// within the e-graph node limit.
pub fn canonicalize(prefix: &str, node_limit: usize) -> Result<String, String> {
    let expr: RecExpr<MathLang> = prefix
        .parse()
        .map_err(|e| format!("parse error: {e}"))?;

    let runner = Runner::default()
        .with_expr(&expr)
        .with_node_limit(node_limit)
        .with_iter_limit(10)
        .run(&rewrite_rules());

    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);
    Ok(best.to_string())
}

/// Compute a canonical hash for deduplication.
pub fn canonical_hash(prefix: &str, node_limit: usize) -> Result<u64, String> {
    let canonical = canonicalize(prefix, node_limit)?;
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    Ok(hasher.finish())
}

/// Extract K diverse equivalent forms from the e-graph.
pub fn diverse_equivalents(prefix: &str, k: usize, node_limit: usize) -> Result<Vec<String>, String> {
    let expr: RecExpr<MathLang> = prefix
        .parse()
        .map_err(|e| format!("parse error: {e}"))?;

    let runner = Runner::default()
        .with_expr(&expr)
        .with_node_limit(node_limit)
        .with_iter_limit(10)
        .run(&rewrite_rules());

    // Extract multiple forms by varying the cost function
    let mut results = Vec::new();
    let extractor = egg::Extractor::new(&runner.egraph, egg::AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);
    results.push(best.to_string());

    // For diversity, we extract with different size preferences
    // This is a simplified approach; full implementation would use
    // a diversity-maximizing extractor
    if results.len() < k {
        let depth_extractor = egg::Extractor::new(&runner.egraph, egg::AstDepth);
        let (_, deep) = depth_extractor.find_best(runner.roots[0]);
        let deep_str = deep.to_string();
        if !results.contains(&deep_str) {
            results.push(deep_str);
        }
    }

    // Cap at k
    results.truncate(k);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commutativity() {
        let a = canonicalize("(add (var x) (var y))", 1000).unwrap();
        let b = canonicalize("(add (var y) (var x))", 1000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_identity() {
        let result = canonicalize("(add (var x) (num 0))", 1000).unwrap();
        assert_eq!(result, "(var x)");
    }

    #[test]
    fn test_exp_log() {
        let result = canonicalize("(exp (log (var x)))", 1000).unwrap();
        assert_eq!(result, "(var x)");
    }

    #[test]
    fn test_canonical_hash_comm() {
        let h1 = canonical_hash("(add (var x) (var y))", 1000).unwrap();
        let h2 = canonical_hash("(add (var y) (var x))", 1000).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_diverse_equivalents() {
        let equivs = diverse_equivalents("(add (var x) (var y))", 4, 1000).unwrap();
        assert!(!equivs.is_empty());
        assert!(equivs.len() <= 4);
    }
}
