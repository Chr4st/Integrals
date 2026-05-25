use std::fmt::Write;

use crate::expr::ExprNode;

/// Replace all numeric leaves (Num, Rational) with a canonical constant
/// placeholder `C`, keeping structure and variable/parameter names.
/// Returns a canonical prefix-notation string.
pub fn skeleton(expr: &ExprNode) -> String {
    let mut buf = String::with_capacity(64);
    write_skeleton(expr, &mut buf);
    buf
}

fn write_skeleton(expr: &ExprNode, buf: &mut String) {
    match expr {
        ExprNode::Num(_) | ExprNode::Rational(_, _) => buf.push('C'),
        ExprNode::Var(v) => {
            buf.push_str("var:");
            buf.push_str(v.as_str());
        }
        ExprNode::Param(p) => {
            buf.push_str("param:");
            buf.push_str(p.as_str());
        }
        ExprNode::Const(c) => {
            let _ = write!(buf, "const:{c}");
        }
        ExprNode::Unary(op, child) => {
            let _ = write!(buf, "{op} ");
            write_skeleton(child, buf);
        }
        ExprNode::Binary(op, l, r) => {
            let _ = write!(buf, "{op} ");
            write_skeleton(l, buf);
            buf.push(' ');
            write_skeleton(r, buf);
        }
        ExprNode::Hyp2F1(a, b, c, d) => {
            buf.push_str("hyp2f1 ");
            write_skeleton(a, buf);
            buf.push(' ');
            write_skeleton(b, buf);
            buf.push(' ');
            write_skeleton(c, buf);
            buf.push(' ');
            write_skeleton(d, buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_replaces_numbers() {
        let e = ExprNode::add(
            ExprNode::mul(ExprNode::num(3), ExprNode::var("x")),
            ExprNode::num(5),
        );
        assert_eq!(skeleton(&e), "add mul C var:x C");
    }

    #[test]
    fn skeleton_replaces_rationals() {
        let e = ExprNode::Rational(1, 2);
        assert_eq!(skeleton(&e), "C");
    }

    #[test]
    fn skeleton_preserves_structure() {
        let e = ExprNode::sin(ExprNode::add(
            ExprNode::var("x"),
            ExprNode::num(1),
        ));
        assert_eq!(skeleton(&e), "sin add var:x C");
    }

    #[test]
    fn skeleton_preserves_constants() {
        use crate::expr::MathConst;
        let e = ExprNode::mul(
            ExprNode::Const(MathConst::Pi),
            ExprNode::var("x"),
        );
        assert_eq!(skeleton(&e), "mul const:pi var:x");
    }
}
