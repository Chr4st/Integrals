use super::{BinaryOp, ExprNode, MathConst, ParamId, UnaryOp, VarId};

// Operator name tables are provided by Display impls on UnaryOp / BinaryOp
// in the parent module.  Parsing tables (string -> enum) remain here.

fn parse_unary_op(s: &str) -> Option<UnaryOp> {
    Some(match s {
        "neg" => UnaryOp::Neg,
        "sin" => UnaryOp::Sin,
        "cos" => UnaryOp::Cos,
        "tan" => UnaryOp::Tan,
        "exp" => UnaryOp::Exp,
        "log" => UnaryOp::Log,
        "sqrt" => UnaryOp::Sqrt,
        "asin" => UnaryOp::Asin,
        "acos" => UnaryOp::Acos,
        "atan" => UnaryOp::Atan,
        "sinh" => UnaryOp::Sinh,
        "cosh" => UnaryOp::Cosh,
        "tanh" => UnaryOp::Tanh,
        "erf" => UnaryOp::Erf,
        "ei" => UnaryOp::Ei,
        "si" => UnaryOp::Si,
        "ci" => UnaryOp::Ci,
        "li" => UnaryOp::Li,
        "gamma" => UnaryOp::Gamma,
        "digamma" => UnaryOp::Digamma,
        "fresnelS" => UnaryOp::FresnelS,
        "fresnelC" => UnaryOp::FresnelC,
        "ellipticK" => UnaryOp::EllipticK,
        "ellipticE" => UnaryOp::EllipticE,
        _ => return None,
    })
}

fn parse_binary_op(s: &str) -> Option<BinaryOp> {
    Some(match s {
        "add" => BinaryOp::Add,
        "sub" => BinaryOp::Sub,
        "mul" => BinaryOp::Mul,
        "div" => BinaryOp::Div,
        "pow" => BinaryOp::Pow,
        "besselJ" => BinaryOp::BesselJ,
        "besselY" => BinaryOp::BesselY,
        "polylog" => BinaryOp::Polylog,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Serialization: ExprNode -> prefix string (buffer-based)
// ---------------------------------------------------------------------------
pub fn to_prefix_string(node: &ExprNode) -> String {
    let mut buf = String::with_capacity(128);
    write_prefix(node, &mut buf);
    buf
}

fn write_prefix(node: &ExprNode, buf: &mut String) {
    use std::fmt::Write;
    match node {
        ExprNode::Var(v) => {
            buf.push_str("var:");
            buf.push_str(v.as_str());
        }
        ExprNode::Param(p) => {
            buf.push_str("param:");
            buf.push_str(p.as_str());
        }
        ExprNode::Num(n) => {
            let _ = write!(buf, "{n}");
        }
        ExprNode::Rational(n, d) => {
            let _ = write!(buf, "rat:{n}/{d}");
        }
        ExprNode::Const(c) => {
            buf.push_str("const:");
            let _ = write!(buf, "{c}");
        }
        ExprNode::Unary(op, c) => {
            let _ = write!(buf, "{op} ");
            write_prefix(c, buf);
        }
        ExprNode::Binary(op, l, r) => {
            let _ = write!(buf, "{op} ");
            write_prefix(l, buf);
            buf.push(' ');
            write_prefix(r, buf);
        }
        ExprNode::Hyp2F1(a, b, c, d) => {
            buf.push_str("hyp2f1 ");
            write_prefix(a, buf);
            buf.push(' ');
            write_prefix(b, buf);
            buf.push(' ');
            write_prefix(c, buf);
            buf.push(' ');
            write_prefix(d, buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing: prefix string -> ExprNode
// ---------------------------------------------------------------------------
pub fn from_prefix_string(s: &str) -> Result<ExprNode, String> {
    let tokens = tokenize(s);
    let mut pos = 0;
    let result = parse_node(&tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(format!(
            "trailing tokens after position {pos}: {:?}",
            &tokens[pos..]
        ));
    }
    Ok(result)
}

fn tokenize(s: &str) -> Vec<String> {
    s.split_whitespace().map(String::from).collect()
}

fn parse_node(tokens: &[String], pos: &mut usize) -> Result<ExprNode, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of input".to_string());
    }
    let tok = &tokens[*pos];

    // Leaf: var:name
    if let Some(name) = tok.strip_prefix("var:") {
        *pos += 1;
        let id = VarId::from_str(name)
            .ok_or_else(|| format!("unknown variable: {name}"))?;
        return Ok(ExprNode::Var(id));
    }
    // Leaf: param:name
    if let Some(name) = tok.strip_prefix("param:") {
        *pos += 1;
        let id = ParamId::from_str(name)
            .ok_or_else(|| format!("unknown parameter: {name}"))?;
        return Ok(ExprNode::Param(id));
    }
    // Leaf: rat:n/d
    if let Some(rest) = tok.strip_prefix("rat:") {
        *pos += 1;
        let mut parts = rest.splitn(2, '/');
        let n: i64 = parts
            .next()
            .ok_or_else(|| format!("bad rational: {tok}"))?
            .parse()
            .map_err(|e| format!("{e}"))?;
        let d: i64 = parts
            .next()
            .ok_or_else(|| format!("bad rational: {tok}"))?
            .parse()
            .map_err(|e| format!("{e}"))?;
        if d == 0 {
            return Err("rational with zero denominator".to_string());
        }
        return Ok(ExprNode::Rational(n, d));
    }
    // Leaf: const:name
    if let Some(name) = tok.strip_prefix("const:") {
        *pos += 1;
        let c = match name {
            "pi" => MathConst::Pi,
            "euler_gamma" => MathConst::EulerGamma,
            "catalan" => MathConst::Catalan,
            _ => return Err(format!("unknown constant: {name}")),
        };
        return Ok(ExprNode::Const(c));
    }
    // Leaf: integer literal
    if let Ok(n) = tok.parse::<i64>() {
        *pos += 1;
        return Ok(ExprNode::Num(n));
    }
    // hyp2f1 (4 children)
    if tok == "hyp2f1" {
        *pos += 1;
        let a = parse_node(tokens, pos)?;
        let b = parse_node(tokens, pos)?;
        let c = parse_node(tokens, pos)?;
        let d = parse_node(tokens, pos)?;
        return Ok(ExprNode::Hyp2F1(
            Box::new(a), Box::new(b), Box::new(c), Box::new(d),
        ));
    }
    // Unary op
    if let Some(op) = parse_unary_op(tok) {
        *pos += 1;
        let child = parse_node(tokens, pos)?;
        return Ok(ExprNode::Unary(op, Box::new(child)));
    }
    // Binary op
    if let Some(op) = parse_binary_op(tok) {
        *pos += 1;
        let left = parse_node(tokens, pos)?;
        let right = parse_node(tokens, pos)?;
        return Ok(ExprNode::Binary(op, Box::new(left), Box::new(right)));
    }

    Err(format!("unknown token: {tok}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let e = ExprNode::add(ExprNode::var("x"), ExprNode::num(3));
        let s = to_prefix_string(&e);
        assert_eq!(s, "add var:x 3");
        let parsed = from_prefix_string(&s).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn roundtrip_nested() {
        let e = ExprNode::mul(
            ExprNode::sin(ExprNode::var("x")),
            ExprNode::exp(ExprNode::param("a")),
        );
        let s = to_prefix_string(&e);
        let parsed = from_prefix_string(&s).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn roundtrip_rational_const() {
        let e = ExprNode::add(
            ExprNode::Rational(1, 2),
            ExprNode::Const(MathConst::Pi),
        );
        let s = to_prefix_string(&e);
        assert!(s.contains("rat:1/2"));
        assert!(s.contains("const:pi"));
        let parsed = from_prefix_string(&s).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn roundtrip_hyp2f1() {
        let e = ExprNode::Hyp2F1(
            Box::new(ExprNode::num(1)),
            Box::new(ExprNode::num(2)),
            Box::new(ExprNode::num(3)),
            Box::new(ExprNode::var("x")),
        );
        let s = to_prefix_string(&e);
        let parsed = from_prefix_string(&s).unwrap();
        assert_eq!(parsed, e);
    }

    #[test]
    fn error_trailing_tokens() {
        let r = from_prefix_string("add var:x 3 5");
        assert!(r.is_err());
    }

    #[test]
    fn error_empty() {
        let r = from_prefix_string("");
        assert!(r.is_err());
    }
}
