# Phase 9a: Grammar-Constrained Decoding
## Goal
Build the grammar mask that prevents the decoder from generating
invalid prefix-notation expressions. This ensures every output
can be parsed back into a valid expression tree.

## Why This Is Needed
Not every prefix token sequence is valid (`add x` needs 2 args but got 1).
The mask forces every output to be a well-formed expression tree.

## File: `python/neurips/models/grammar.py`

### The Arity Stack (the core algorithm)
Track how many arguments each operator still needs:

```
Step 1: Generate "add"    → Stack: [2]       add needs 2 args
Step 2: Generate "sin"    → Stack: [1, 1]    add needs 1 more, sin needs 1
Step 3: Generate "x"      → Stack: [1]       sin got x, popped. add needs 1
Step 4: Generate "mul"    → Stack: [0, 2]    add will be done after mul. mul needs 2
Step 5: Generate "x"      → Stack: [0, 1]    mul needs 1 more
Step 6: Generate "x"      → Stack: []        complete! add(sin(x), mul(x, x))
```

```python
class ArityStack:
    def __init__(self):
        self.stack: list[int] = []
    
    def push_op(self, token_id: int):
        """Operator generated: push its arity requirement."""
        arity = ARITY_TABLE[token_id]  # sin→1, add→2, x→0, etc.
        if arity > 0:
            self.stack.append(arity)
        else:
            # Leaf: decrement the top of stack
            self._consume()
    
    def _consume(self):
        """A leaf/completed subtree: decrement parent's remaining count."""
        if self.stack:
            self.stack[-1] -= 1
        # Cascade: if parent is now satisfied, pop and consume its parent
        while self.stack and self.stack[-1] == 0:
            self.stack.pop()
            if self.stack:
                self.stack[-1] -= 1
    
    def is_complete(self) -> bool:
        """Is the expression fully formed?"""
        return len(self.stack) == 0
    
    def remaining_slots(self) -> int:
        """How many more tokens needed to complete the expression?"""
        return sum(self.stack)
```

### Grammar Mask Generation
```python
class PrefixGrammarMask:
    def get_mask(self, generated_ids: list[int]) -> torch.BoolTensor:
        """Returns [vocab_size] mask. True = allowed, False = blocked."""
        stack = ArityStack()
        for token_id in generated_ids:
            stack.push_op(token_id)
        
        mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool)
        
        if stack.is_complete():
            mask[EOS_ID] = True  # only EOS allowed
            return mask
        
        # Prevent trees that are too deep
        if len(stack.stack) > MAX_TREE_DEPTH:
            # Only allow leaves (arity-0 tokens)
            for tid in LEAF_TOKENS:
                mask[tid] = True
        else:
            # Allow any operator or leaf
            for tid in ALL_EXPRESSION_TOKENS:
                mask[tid] = True
        
        return mask
```

### How the Mask Is Applied
```python
# In the decoder's forward pass:
logits = decoder_layer(...)           # [vocab_size] raw scores
mask = grammar.get_mask(generated)    # [vocab_size] True/False
logits[~mask] = float('-inf')         # blocked tokens → -infinity
probs = torch.softmax(logits, dim=-1) # only valid tokens get probability
next_token = sample(probs)            # sample from valid tokens only
```

## Verification
- 100K random samples: ALL parse back to valid expression trees (zero invalid)
- Mask never blocks all tokens (always at least one valid option)
- EOS is only allowed when expression is complete
- Max depth constraint is enforced (no trees deeper than 20)
