# Phase 11b: Variable-Aware Attention

## Goal
Build the attention layer that helps the model understand which parts
of the input tree depend on the integration variable x. This is critical
for multivariate integrals where the model must separate "parts involving x"
from "parts that are constant w.r.t. x."

## The Dependency Mask (Free — No Learning)
```python
def compute_dependency_mask(tree: ExprNode, int_var: str) -> list[bool]:
    """For each node: does its subtree contain the integration variable?
    
    Computed bottom-up in one pass. No neural network needed.
    
    Example: ∫ x·sin(y) dx
         mul          → True (subtree contains x)
        /   \
       x     sin      → x: True, sin: False (subtree is sin(y), no x)
              |
              y       → False (y is not x)
    """
    def _recurse(node):
        if node.is_leaf():
            return str(node) == int_var  # True only for x
        child_deps = [_recurse(c) for c in node.children()]
        return any(child_deps)  # True if ANY child depends on x
    # ... collect into list indexed by node ID
```

## Variable-Aware Attention Layer
```python
class VariableAwareAttention(nn.Module):
    def __init__(self, dim=256, n_heads=8):
        self.attn = nn.MultiheadAttention(dim, n_heads)
        # Learned bias: how much to boost same-dependency attention
        self.dep_bias = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, node_embs, dep_mask):
        """
        node_embs: [num_nodes, 256]
        dep_mask: [num_nodes] boolean — True if subtree contains int_var
        """
        n = len(dep_mask)
        dep = dep_mask.float()  # [n] — 1.0 for x-dependent, 0.0 for independent
        
        # Same-group bonus matrix:
        # both depend on x → bonus
        # both independent of x → bonus
        # one depends, one doesn't → no bonus (or penalty)
        same_dep = dep.unsqueeze(0) * dep.unsqueeze(1)           # [n, n]
        same_indep = (1-dep).unsqueeze(0) * (1-dep).unsqueeze(1) # [n, n]
        bias = (same_dep + same_indep) * self.dep_bias           # [n, n]
        
        # bias[i,j] > 0 when nodes i,j have same x-dependency
        # bias[i,j] = 0 when they differ
        # This gets ADDED to attention scores before softmax
        
        out, attn_weights = self.attn(
            node_embs, node_embs, node_embs,
            attn_mask=bias
        )
        return out
```

## Full Tree Model Assembly
```python
class TreeIntegrator(nn.Module):
    def __init__(self, config):
        self.node_embed = NodeEmbedding()          # Phase 10a
        self.gnn_encoder = TreeMessagePassing()     # Phase 10b
        self.var_attn = VariableAwareAttention()    # this phase
        self.decoder = TreeDecoder()                # Phase 11a
    
    def forward(self, input_tree, int_var):
        # 1. Embed each node: [num_nodes, 256]
        node_embs = self.node_embed(input_tree)
        
        # 2. Message passing: nodes exchange info (8 rounds)
        encoded = self.gnn_encoder(node_embs, input_tree.edge_index)
        
        # 3. Variable-aware attention: group by x-dependency
        dep_mask = compute_dependency_mask(input_tree, int_var)
        encoded = self.var_attn(encoded, dep_mask)
        
        # 4. Decode: build output tree top-down
        output_tree = self.decoder.decode_tree(encoded)
        return output_tree
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
        # Expected: ~9M total
```

## Verification
- Dependency mask: x·sin(y) → [True, True, False, False] for [mul, x, sin, y]
- Attention weights: x-dependent nodes attend more to each other after training
- Full forward pass: input tree → output tree, no errors
- Gradient flows from output to all input nodes
- Total parameters: ~9M (vs 95M for sequence transformer)
