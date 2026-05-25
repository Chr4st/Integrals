# Phase 11a: Top-Down Tree Decoder

## Goal
Build the decoder that generates the output antiderivative tree
level by level, from root to leaves.

## How It Works (Simply)
Instead of writing the answer left-to-right like a sentence,
build it top-down like an org chart:

1. Pick the root: "this answer is a product" → mul
2. Pick root's children: "left is a fraction, right is sin" → div, sin
3. Pick their children: div → (pow, 2), sin → (y)
4. Pick pow's children: (x, 2)
5. Done: mul(div(pow(x,2), 2), sin(y)) = x²/2 · sin(y)

The model makes BIG decisions first (overall structure) and
SMALL decisions last (specific constants). This is how humans
think about integrals too.

## File: `python/neurips/models/tree_decoder.py`

### Decoder Architecture
```python
class TreeDecoder(nn.Module):
    def __init__(self, node_dim=256, n_levels=8, n_heads=8):
        # Cross-attention: output nodes look at input nodes
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=node_dim,  # 256: size of each node vector
                num_heads=n_heads,   # 8: parallel attention perspectives
                # each head looks at 256/8 = 32 numbers independently
            )
            for _ in range(n_levels)
        ])
        
        # Predict what symbol each node is
        self.symbol_head = nn.Linear(node_dim, 256)  # 256 → 256 vocab
        
        # Generate child embeddings from parent
        # Parent's 256 numbers → two children's 256 numbers each
        self.child_init = nn.Linear(node_dim, node_dim * 2)  # 256 → 512
        
        # Initial root embedding (learned starting point)
        self.root_emb = nn.Parameter(torch.randn(node_dim))
```

### Level-by-Level Generation
```python
def decode_tree(self, encoder_output, max_levels=8):
    # Start with just a root
    current_level = [self.root_emb.unsqueeze(0)]  # [1, 256]
    tree_nodes = []
    
    for level in range(max_levels):
        if not current_level:
            break  # all nodes are leaves, tree is complete
        
        level_embs = torch.stack(current_level)  # [n_nodes, 256]
        
        # Cross-attention: output nodes look at encoded input
        attended = self.cross_attn[level](
            query=level_embs,
            key=encoder_output,
            value=encoder_output,
        )  # [n_nodes, 256]
        
        # Predict symbol for each node
        logits = self.symbol_head(attended)  # [n_nodes, 256]
        symbols = sample_or_argmax(logits)   # [n_nodes]
        
        # Determine arity of each predicted symbol
        arities = [ARITY_TABLE[s.item()] for s in symbols]
        
        # Generate children for non-leaf nodes
        next_level = []
        for node_emb, arity in zip(attended, arities):
            if arity == 0:
                continue  # leaf node, no children needed
            child_pair = self.child_init(node_emb)  # [512]
            left = child_pair[:256]   # first 256 → left child
            right = child_pair[256:]  # second 256 → right child
            next_level.append(left)
            if arity >= 2:
                next_level.append(right)
        
        tree_nodes.extend(list(zip(symbols, attended)))
        current_level = next_level
    
    return reconstruct_tree(tree_nodes)
```

## Verification
- Root-only tree (e.g., just "x"): correctly predicts single leaf
- 3-level tree: generates valid tree structure
- All arities respected: sin always gets exactly 1 child
- Cross-attention weights are non-uniform (model looks at specific input nodes)
- Parameter count: ~4.2M
