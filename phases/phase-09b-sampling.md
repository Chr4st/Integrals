# Phase 9b: Temperature Sampling + Verification Pipeline

## Goal
Build the sampling strategy that generates N=25 diverse candidate answers
per integral, and the pipeline that verifies each until one passes.

## Why Sample Multiple Times
A model that's right 5% of the time per guess sounds bad.
But if you guess 25 times: 1 - (1-0.05)^25 = 72.3% overall.
Temperature sampling + verification turns a weak predictor into a strong solver.

## File: `python/neurips/models/sampler.py`

### Temperature Sampling
```python
def sample_candidates(
    model, encoder_output, grammar,
    n_samples: int = 25,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_length: int = 256,
) -> list[list[int]]:
    """Generate n_samples different candidate sequences."""
    candidates = []
    for _ in range(n_samples):
        tokens = [SOS_ID]
        for step in range(max_length):
            logits = model.decode_step(encoder_output, tokens)
            
            # Apply grammar mask (Phase 9a)
            mask = grammar.get_mask(tokens[1:])  # skip SOS
            logits[~mask] = float('-inf')
            
            # Temperature scaling
            # T < 1: sharpen (more deterministic)
            # T = 1: original distribution
            # T > 1: flatten (more random)
            logits = logits / temperature
            
            # Top-p (nucleus) sampling
            # Sort tokens by probability, keep cumulative < 0.95
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)
            cutoff = (cumsum < top_p).sum() + 1
            probs[sorted_idx[cutoff:]] = 0  # zero out tail
            probs = probs / probs.sum()  # renormalize
            
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            
            if next_token == EOS_ID:
                break
        
        candidates.append(tokens)
    return candidates
```

### Sample-and-Verify Pipeline
```python
def sample_and_verify(model, example, oracle, tokenizer,
                      n_samples=25, temperature=0.7, top_p=0.95):
    """The full inference pipeline for one integral.
    
    Returns the first verified-correct answer, or None if all fail.
    """
    # Encode the integrand once
    src_ids = tokenizer.encode(example)
    features = extract_features(example["tree"], example["var"])
    enc_out = model.encode(src_ids, features)
    
    # Generate N candidates
    grammar = PrefixGrammarMask()
    candidates = sample_candidates(
        model, enc_out, grammar, n_samples, temperature, top_p
    )
    
    # Verify each candidate
    for candidate_ids in candidates:
        try:
            candidate_tree = tokenizer.decode(candidate_ids)
            if oracle.verify(candidate_tree, example["tree"], example["task_info"]):
                return candidate_tree  # found a correct answer
        except (ParseError, TimeoutError):
            continue  # skip invalid or slow candidates
    
    return None  # all 25 failed
```

## Verification
- Temperature=0: output matches greedy decoding exactly
- N=25: produces 25 DIFFERENT candidates (not all identical)
- Grammar mask: no candidate is an invalid expression
- sample_and_verify returns None only when all 25 truly fail verification
- Timing: 25 candidates generated in < 2 seconds per integral
