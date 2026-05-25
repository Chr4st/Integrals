# Phase 10a: Tree GNN — Node Embedding

## Goal
Build the initial embedding layer that turns each node in the expression
tree into a 256-number vector. This is the starting point for the GNN.

## What This Does (Simply)
Before the GNN can process the tree, each node needs to be described
as a list of numbers. Think of it as a fingerprint:
- "sin" → [0.3, -0.1, 0.8, ...] (256 numbers)
- "x" → [0.5, 0.2, -0.4, ...] (256 numbers)
- "add" → [-0.2, 0.6, 0.1, ...] (256 numbers)

Each node's fingerprint is built from 3 parts:
1. What symbol it is (64 numbers)
2. What role it plays in this integral (64 numbers)
3. What its subtree looks like (128 numbers)

## File: `python/neurips/models/tree_gnn.py`

### Part 1: Symbol Embedding (64-dim)
```python
self.symbol_emb = nn.Embedding(256, 64)
# 256 possible symbols (from tokenizer), each gets 64 numbers.
# At first these are RANDOM. During training, the model learns that:
#   sin and cos should have similar embeddings (both oscillatory, arity 1)
#   sin and add should be different (different arity, different math)
#   x and y should be similar (both variables)
#   x and 3 should be different (variable vs constant)
```

### Part 2: Role Embedding (64-dim)
```python
self.role_mlp = nn.Sequential(
    nn.Linear(12, 32),  # 12 input features → 32 hidden
    nn.ReLU(),          # nonlinearity
    nn.Linear(32, 64),  # 32 hidden → 64 output
)
```

The 12 input features (one number each, all 0 or 1 except where noted):
1. `is_int_var`: is this node the integration variable x?
2. `is_free_var`: is this a different variable (y, z)?
3. `is_param`: is this a symbolic parameter (a, b)?
4. `is_operator`: is this an operator (add, sin, pow...)?
5. `is_leaf`: is this a leaf node (no children)?
6. `depth_normalized`: depth ÷ max_depth (float 0.0 to 1.0)
7. `subtree_has_int_var`: does anything below contain x?
8. `subtree_has_free_var`: does anything below contain y/z?
9. `left_child_has_var`: does left subtree contain x?
10. `right_child_has_var`: does right subtree contain x?
11. `num_descendants_normalized`: nodes below ÷ total nodes
12. `composition_depth`: how many unary functions stacked above

**Why ReLU?** A linear layer computes `output = weight × input + bias`.
This can only learn straight lines. ReLU (max(0, x)) bends the line,
letting the network learn curved relationships. Two linear layers
with ReLU between them can approximate any smooth function.

### Part 3: Structural Embedding (128-dim)
```python
self.struct_mlp = nn.Sequential(
    nn.Linear(40, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
)
```

The 40 input features (per-node subtree statistics):
- 8 operator-type counts at this node's subtree (add, mul, sin, cos, exp, log, pow, other)
- 8 depth-distribution bins (how deep is the subtree below?)
- 5 function-composition features (sin∘cos, exp∘poly, etc.)
- 5 variable-pattern features (polynomial, rational, transcendental, mixed, constant)
- 5 complexity measures (node count, max depth, branching factor, leaf ratio, op ratio)
- 9 reserved (zeros for future extension)

### Combining the Parts
```python
class NodeEmbedding(nn.Module):
    def forward(self, symbol_ids, role_features, struct_features):
        s = self.symbol_emb(symbol_ids)       # [num_nodes, 64]
        r = self.role_mlp(role_features)       # [num_nodes, 64]
        t = self.struct_mlp(struct_features)   # [num_nodes, 128]
        return torch.cat([s, r, t], dim=-1)    # [num_nodes, 256]
```

The output is [num_nodes, 256]: each node in the tree gets a 256-number
vector combining what it is, what role it plays, and what's below it.

## Verification
- Single node: symbol_id=20 (sin) → embedding shape [256]
- 10-node tree → 10 embeddings, each shape [256]
- Embedding values change during training (not frozen)
- Similar symbols get similar embeddings after training (cos similarity > 0.8 for sin/cos)
