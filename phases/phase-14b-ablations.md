# Phase 14b: Ablation Studies + Error Analysis

## Goal
Run controlled experiments to justify each architectural decision.
Ablations answer: "what happens if we remove this component?"

## File: `python/neurips/evaluation/ablations.py`

### Ablation 1: Message Passing Rounds
Train tree GNN with 2, 4, 6, 8, 10 rounds. Keep everything else identical.
```python
def ablate_message_rounds():
    for n_rounds in [2, 4, 6, 8, 10]:
        model = TreeIntegrator(config, message_rounds=n_rounds)
        train(model, ...)
        results[n_rounds] = evaluate(model, test_data, ...)
    # Expected: accuracy increases up to 8, diminishing after
    # This justifies our choice of 8 rounds
```

### Ablation 2: Variable-Aware Attention
Train tree GNN with vs without the variable-aware attention layer.
```python
def ablate_variable_attention():
    model_with = TreeIntegrator(config, use_var_attn=True)
    model_without = TreeIntegrator(config, use_var_attn=False)
    # Expected: big gap on multivariate (5-15%), small gap on univariate
    # This justifies the variable-aware attention design
```

### Ablation 4: Curriculum Learning
Train both models with vs without curriculum.
```python
def ablate_curriculum():
    model_curriculum = train(model, curriculum=True)
    model_no_curriculum = train(model, curriculum=False)
    # Expected: curriculum improves hard integrals by 5-10%
    # Final accuracy also higher with curriculum
```

### Ablation 5: Number of Samples
Evaluate with N = 1, 5, 10, 25, 50 samples.
```python
def ablate_n_samples():
    for n in [1, 5, 10, 25, 50]:
        results[n] = evaluate(model, test_data, n_samples=n)
    # Expected: steep improvement 1→25, diminishing 25→50
    # Plot: solve rate vs compute cost (linear x-axis = N, y-axis = %)
```

## File: `python/neurips/evaluation/analysis.py`

### Error Analysis
```python
def analyze_failures(model, test_data, oracle, tokenizer):
    """For every integral the model fails on, classify WHY it failed."""
    failure_types = {
        "wrong_structure": [],   # output tree shape is wrong
        "wrong_constants": [],   # right shape, wrong numbers
        "wrong_function": [],    # e.g., sin instead of cos
        "timeout": [],           # verification timed out
        "unparseable": [],       # output isn't a valid expression
        "all_25_wrong": [],      # all candidates verified as wrong
    }
    
    for example in test_data:
        candidates = sample_candidates(model, example, n=25)
        
        if not candidates:
            failure_types["unparseable"].append(example)
            continue
        
        # Check first candidate in detail
        best = candidates[0]
        target = example["antiderivative_tree"]
        
        if same_structure(best, target) and not same_constants(best, target):
            failure_types["wrong_constants"].append(example)
        elif not same_structure(best, target):
            failure_types["wrong_structure"].append(example)
        # ... etc.
```

## File: `scripts/evaluate.py` (updated)
```bash
python scripts/evaluate.py --ablation rounds
python scripts/evaluate.py --ablation var_attn
python scripts/evaluate.py --ablation decoder
python scripts/evaluate.py --ablation curriculum
python scripts/evaluate.py --ablation n_samples
python scripts/evaluate.py --ablation split
python scripts/evaluate.py --error-analysis
```

## Verification
- Each ablation produces a clear trend (not random noise)
- All ablations use same test set for fair comparison
- Error categories are mutually exclusive and sum to total failures
- Plots render correctly and are publication-quality
