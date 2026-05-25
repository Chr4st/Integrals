# Phase 8: Sequence Transformer (Baseline Architecture)

## Goal
Implement the standard encoder-decoder transformer that takes prefix-notation
sequences as input and generates prefix-notation antiderivatives as output.
This is the Lample & Charton (2020) style baseline we compare against.

## What This Architecture Does
1. Takes the integrand as a flat token sequence: `[INDEF] [VAR] x mul x sin x`
2. Encodes it into a list of hidden vectors (one per token)
3. Decodes token by token, producing the antiderivative sequence
4. Each output token is constrained by grammar rules (valid prefix notation)

## File: `python/neurips/models/seq_transformer.py`

### Hyperparameters
```
d_model = 640     # size of each token's hidden vector
                  # (640 numbers describe each token's "state")
n_heads = 10      # number of attention heads
                  # (10 different "perspectives" looking at the sequence)
n_layers = 10     # number of transformer layers (depth of processing)
d_ff = 2560       # feedforward hidden size (4x d_model, standard ratio)
dropout = 0.1     # randomly zero out 10% of values during training
                  # (prevents memorization, forces generalization)
vocab_size = 256  # from tokenizer (Phase 7)
max_seq_len = 512 # longest sequence the model handles
```

### Why These Numbers
- **d_model=640**: Each token is described by 640 numbers. Bigger = more
  expressive but slower. 640 is enough to represent complex math patterns.
  For comparison: GPT-2 uses 768, BERT uses 768. We're slightly smaller
  because our vocab is much smaller (256 vs 50K).
- **n_heads=10**: Attention heads let the model look at different aspects
  simultaneously. Head 1 might track "where is x?", head 2 might track
  "what operators are nearby?", etc. 10 heads, each looking at 640/10=64
  numbers.
- **n_layers=10**: Each layer is one round of "every token looks at every
  other token and updates itself." 10 rounds is enough for the information
  to propagate fully. More layers = more capacity but diminishing returns.

### Encoder
```python
class SeqEncoder(nn.Module):
    # Input: token IDs [batch, seq_len] → integers like [5, 7, 70, 12, ...]
    # Step 1: Token embedding: each ID → 640-dim vector
    #         Shape: [batch, seq_len, 640]
    # Step 2: Add positional encoding (sinusoidal)
    #         So the model knows token order (1st, 2nd, 3rd...)
    # Step 3: Inject [FEAT] token embedding (replace with 688→640 projected features)
    # Step 4: Pass through 10 transformer layers
    #         Each layer: self-attention → feedforward → normalize
    # Output: [batch, seq_len, 640] — enriched representations
```

### Decoder
```python
class SeqDecoder(nn.Module):
    # Input: previously generated tokens + encoder output
    # Step 1: Token embedding + positional encoding (same as encoder)
    # Step 2: Pass through 10 transformer layers
    #         Each layer:
    #           - Masked self-attention (can only look at past tokens, not future)
    #           - Cross-attention (looks at encoder output to "read" the integrand)
    #           - Feedforward → normalize
    # Step 3: Project to vocab logits: [batch, seq_len, 640] → [batch, seq_len, 256]
    #         Each position gets 256 scores, one per possible next token
    # Step 4: Apply grammar mask (Phase 9) to zero out invalid tokens
    # Output: probability distribution over next token
```

### Full Model
```python
class SeqTransformer(nn.Module):
    def __init__(self, config):
        self.encoder = SeqEncoder(config)
        self.decoder = SeqDecoder(config)
        self.feat_proj = nn.Linear(688, 640)  # project features to model dim
    
    def forward(self, src_ids, tgt_ids, features):
        feat_emb = self.feat_proj(features)       # [batch, 688] → [batch, 640]
        enc_out = self.encoder(src_ids, feat_emb)  # [batch, src_len, 640]
        logits = self.decoder(tgt_ids, enc_out)    # [batch, tgt_len, 256]
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
        # Expected: ~95M parameters
```

## Verification
- `model.count_parameters()` is ~95M
- Forward pass: random input → output shape [batch, tgt_len, 256]
- Overfit test: train on 10 pairs for 100 steps → loss → 0
