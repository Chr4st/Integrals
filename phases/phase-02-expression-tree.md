# Phase 2: Expression Tree (Rust Core)

## Goal
Define the core expression tree data structure in Rust with Python bindings.

## What an Expression Tree Is
Every math expression is a tree. `sin(x*y) + 3` becomes:
```
      add
     /   \
   sin    3
    |
   mul
  / \
 x   y
```
Each node is either an operator (add, mul, sin...) with children,
or a leaf (variable x, number 3, constant pi).

## File: `rust/core/src/expr.rs`

Define an enum `ExprNode`:
```rust
pub enum ExprNode {
    // Leaves (no children)
    Var(VarName),           // x, y, z, w, t
    Param(ParamName),       // a, b, c, n, k, alpha, beta
    Num(i64),               // integers: 0, 1, 2, -3, 100
    Rational(i64, i64),     // fractions: 1/2, 3/7
    Const(MathConst),       // pi, euler_gamma, catalan

    // Unary operators (1 child)
    Neg(Box<ExprNode>),
    Sin(Box<ExprNode>),
    Cos(Box<ExprNode>),
    Tan(Box<ExprNode>),
    Exp(Box<ExprNode>),
    Log(Box<ExprNode>),
    Sqrt(Box<ExprNode>),
    Asin(Box<ExprNode>),
    Acos(Box<ExprNode>),
    Atan(Box<ExprNode>),
    Sinh(Box<ExprNode>),
    Cosh(Box<ExprNode>),
    Tanh(Box<ExprNode>),
    // Special functions (unary)
    Erf(Box<ExprNode>),
    Ei(Box<ExprNode>),
    Si(Box<ExprNode>),
    Ci(Box<ExprNode>),
    Li(Box<ExprNode>),
    Gamma(Box<ExprNode>),
    Digamma(Box<ExprNode>),
    FresnelS(Box<ExprNode>),
    FresnelC(Box<ExprNode>),
    EllipticK(Box<ExprNode>),
    EllipticE(Box<ExprNode>),

    // Binary operators (2 children)
    Add(Box<ExprNode>, Box<ExprNode>),
    Sub(Box<ExprNode>, Box<ExprNode>),
    Mul(Box<ExprNode>, Box<ExprNode>),
    Div(Box<ExprNode>, Box<ExprNode>),
    Pow(Box<ExprNode>, Box<ExprNode>),
    // Special functions (binary)
    BesselJ(Box<ExprNode>, Box<ExprNode>),  // order, argument
    BesselY(Box<ExprNode>, Box<ExprNode>),
    Polylog(Box<ExprNode>, Box<ExprNode>),  // order, argument

    // Higher arity
    Hyp2F1(Box<ExprNode>, Box<ExprNode>, Box<ExprNode>, Box<ExprNode>),
}
```

## Helper Methods on ExprNode
- `depth(&self) -> usize`: max depth of the tree
- `node_count(&self) -> usize`: total number of nodes
- `contains_var(&self, var: VarName) -> bool`: does this subtree contain variable x?
- `free_vars(&self) -> HashSet<VarName>`: all variables in the expression
- `free_params(&self) -> HashSet<ParamName>`: all parameters in the expression
- `children(&self) -> Vec<&ExprNode>`: direct children of this node
- `arity(&self) -> usize`: 0 for leaves, 1 for unary, 2 for binary, 4 for Hyp2F1
- `to_prefix_string(&self) -> String`: serialize to prefix notation
- `from_prefix_string(s: &str) -> Result<Self>`: parse from prefix notation
- `to_json(&self) -> String`: serialize to JSON for Python interop

## PyO3 Bindings: `rust/core/src/lib.rs`
Expose `ExprNode` as `PyExprTree` class:
- `PyExprTree.depth() -> int`
- `PyExprTree.node_count() -> int`
- `PyExprTree.contains_var(name: str) -> bool`
- `PyExprTree.free_vars() -> set[str]`
- `PyExprTree.to_prefix() -> str`
- `PyExprTree.from_prefix(s: str) -> PyExprTree`
- `PyExprTree.to_json() -> str`

## Verification
- `cargo test` passes for all helper methods
- Round-trip: `from_prefix(to_prefix(tree)) == tree` for 1000 random trees
- Python: `from neurips_core import PyExprTree` works
