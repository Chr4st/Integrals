# Phase 14a: Evaluation + Benchmarking

## Goal
Evaluate both architectures on the held-out 20% test set.
Measure solve rate across all task types and difficulty tiers.
Compare tree GNN (9M params) vs sequence transformer (95M params).

## File: `python/neurips/evaluation/benchmark.py`

### Main Evaluation Function
```python
def evaluate(model, test_data, oracle, tokenizer, config) -> dict:
    """Run model on full test set. Return solve rates."""
    model.eval()  # disable dropout (use full model capacity)
    
    results = {
        "total": 0, "solved": 0,
        "by_task": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_tier": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_task_tier": defaultdict(lambda: {"total": 0, "solved": 0}),
    }
    
    for example in tqdm(test_data, desc="Evaluating"):
        results["total"] += 1
        task = example["task"]
        tier = example["difficulty_tier"]
        key = (task, tier)
        
        # Generate N=25 candidates, verify each
        answer = sample_and_verify(
            model, example, oracle, tokenizer,
            n_samples=config.n_samples,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        
        solved = answer is not None
        results["solved"] += int(solved)
        results["by_task"][task]["total"] += 1
        results["by_task"][task]["solved"] += int(solved)
        results["by_tier"][tier]["total"] += 1
        results["by_tier"][tier]["solved"] += int(solved)
        results["by_task_tier"][key]["total"] += 1
        results["by_task_tier"][key]["solved"] += int(solved)
    
    return results
```

### Decoding Strategy Comparison
Run evaluation three times per model, with different decoding methods:
```python
def compare_decoding(model, test_data, oracle, tokenizer):
    """Compare greedy vs beam vs sampling."""
    strategies = {
        "greedy": {"n_samples": 1, "temperature": 0.0},
        "beam_10": {"beam_width": 10},
        "sample_25": {"n_samples": 25, "temperature": 0.7, "top_p": 0.95},
    }
    results = {}
    for name, params in strategies.items():
        results[name] = evaluate(model, test_data, oracle, tokenizer, params)
    return results
```

### Results Table Printer
```python
def print_comparison(seq_results, tree_results):
    """Print the main comparison table."""
    # Format:
    #                     Seq Transformer (95M)    Tree GNN (9M)
    #                     Greedy  Beam  Sample     Greedy  Beam  Sample
    # Overall:            ?.?%    ?.?%  ?.?%       ?.?%    ?.?%  ?.?%
    # Univariate:         ...
    #   easy:             ...
    #   medium:           ...
    # Multivariate:       ...
    # ...
```

### LaTeX Table Generator
```python
def to_latex_table(results) -> str:
    """Generate LaTeX table for the paper."""
```

## File: `scripts/evaluate.py`
```bash
python scripts/evaluate.py --model seq --checkpoint best --split test
python scripts/evaluate.py --model tree --checkpoint best --split test
python scripts/evaluate.py --model both --all-strategies
```

## Verification
- Evaluation completes on full test set without crashes
- Per-task totals sum to overall total
- Confidence intervals are reasonable (±1-3%, not ±50%)
- LaTeX table compiles correctly in the paper
