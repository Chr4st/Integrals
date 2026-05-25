# Phase 10b: Tree GNN — Message Passing

## Goal
Build the message passing layers where nodes exchange information
along tree edges. After 8 rounds, every node "knows about" the
full tree, not just itself.

## What Message Passing Does (Simply)
Round 1: Each node knows only about itself.
Round 2: Each node now also knows about its immediate neighbors.
Round 3: Each node knows about neighbors-of-neighbors.
...
Round 8: Each node knows about the entire tree.

The information travels ALONG THE TREE EDGES. This is the key
difference from a regular transformer (where every token looks
at every other token regardless of structure).

## File: `python/neurips/models/tree_gnn.py` (continued)

### One Round of Message Passing
```python
class MessageRound(nn.Module):
    def __init__(self, dim=256):
        # Transform for messages going DOWN (parent → child)
        self.parent_msg = nn.Linear(dim, dim)
        # Transform for messages going UP (child → parent)
        self.child_msg = nn.Linear(dim, dim)
        # Combine: [my_state | parent_msg | child_avg] → new state
        self.update = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),  # 768 → 512
            nn.ReLU(),
            nn.Linear(dim * 2, dim),       # 512 → 256
        )
        self.norm = nn.LayerNorm(dim)
```

### Message Aggregation
```python
def aggregate_messages(node_embs, edge_index, direction):
    """Collect messages from neighbors.
    
    edge_index: [2, num_edges] — pairs of (source, target)
    direction="down": parent→child messages (for each child, get parent's msg)
    direction="up": child→parent messages (for each parent, average children's msgs)
    """
    if direction == "up":
        # For each parent, average its children's embeddings
        # scatter_mean: groups by parent ID, averages child vectors
        return scatter_mean(node_embs[edge_index[0]], edge_index[1])
    else:
        # For each child, copy its parent's embedding
        return node_embs[edge_index[1]]
```

### Full Message Passing Stack
```python
class TreeMessagePassing(nn.Module):
    def __init__(self, node_dim=256, n_rounds=8):
        self.rounds = nn.ModuleList([
            MessageRound(node_dim) for _ in range(n_rounds)
        ])
    
    def forward(self, node_embs, edge_index):
        """
        node_embs: [total_nodes, 256]  — all nodes from all trees in batch
        edge_index: [2, total_edges]   — all parent-child edges
        """
        for round_layer in self.rounds:
            parent_msgs = aggregate(
                round_layer.parent_msg(node_embs), edge_index, "down"
            )
            child_msgs = aggregate(
                round_layer.child_msg(node_embs), edge_index, "up"
            )
            combined = torch.cat([node_embs, parent_msgs, child_msgs], dim=-1)
            updated = round_layer.update(combined)
            node_embs = round_layer.norm(node_embs + updated)  # residual + norm
        return node_embs
```

## Parameters: ~4.2M
8 rounds × (3 linear layers + LayerNorm) per round
= 8 × (256×256 + 256×256 + 768×512 + 512×256 + 256)
≈ 525K per round × 8 = 4.2M

## Verification
- After 8 rounds, root node embedding changes when ANY leaf changes
  (information propagated from leaf to root)
- Gradient from root loss reaches all leaf nodes
- Batch of 32 trees: correct output shapes, no cross-tree contamination
- Parameter count: ~4.2M
