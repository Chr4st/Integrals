# Phase 12b: Curriculum Learning
## Goal
Build the scheduler that controls WHAT data the model sees at each epoch.
Start easy, add complexity gradually. This helps the model learn faster
and reach higher final accuracy than training on everything at once.
## Why Curriculum Learning Works (Simply)
Imagine learning piano. Day 1: you don't start with Chopin.
You start with scales, then simple songs, then harder pieces.
Each step builds on skills from the previous one.
Same for the model:
- Epoch 1: learn ∫ x² dx = x³/3 (univariate, easy)
- Epoch 20: learn ∫ x·sin(y) dx (add second variable)
- Epoch 40: learn ∫ e^{-x²} dx = √π/2 · erf(x) (add special functions)
If you show the hardest integrals on epoch 1, the loss is maximum
(random guessing) and the model learns nothing useful from that example.
## File: `python/neurips/training/curriculum.py`
### Task Scheduler
```python
class CurriculumScheduler:
    def get_active_tasks(self, epoch: int) -> dict[str, float]:
        """Which task types to train on, and their sampling weights.
        Weights are probabilities: they sum to 1.0."""
        if epoch <= 15:
            # Phase 1: Master the basics first
            return {"univariate": 1.0}
        elif epoch <= 30:
            # Phase 2: Add multivariate + parametric
            return {
                "univariate": 0.50,    # still half the data
                "multivariate": 0.30,  # the novel contribution
                "parametric": 0.20,
            }
        elif epoch <= 45:
            # Phase 3: Add definite + special functions
            return {
                "univariate": 0.30,
                "multivariate": 0.25,
                "parametric": 0.15,
                "definite": 0.15,
                "special_fn": 0.15,
            }
        else:
            # Phase 4: All tasks, roughly uniform
            return {
                "univariate": 0.25,
                "multivariate": 0.20,
                "parametric": 0.15,
                "definite": 0.20,
                "special_fn": 0.20,
            }
    def get_max_difficulty(self, epoch: int) -> str:
        """Cap how hard the integrals can be."""
        if epoch <= 10:
            return "medium"      # depth ≤ 6, nodes ≤ 15
        elif epoch <= 25:
            return "hard"        # depth ≤ 10, nodes ≤ 30
        else:
            return "very_hard"   # no cap
```
### Data Filtering + Sampling
```python
def filter_data(train_data, active_tasks, max_difficulty):
    """Select examples matching current curriculum stage."""
    difficulty_order = ["easy", "medium", "hard", "very_hard"]
    max_idx = difficulty_order.index(max_difficulty)
    allowed_tiers = set(difficulty_order[:max_idx + 1])
    filtered = [
        ex for ex in train_data
        if ex["task"] in active_tasks
        and ex["difficulty_tier"] in allowed_tiers
    ]
    return filtered
def weighted_sample(filtered_data, task_weights, n_samples):
    """Sample n examples according to task weights."""
    tasks = list(task_weights.keys())
    weights = list(task_weights.values())
    by_task = {t: [ex for ex in filtered_data if ex["task"] == t] for t in tasks}
    sampled = []
    for _ in range(n_samples):
        task = random.choices(tasks, weights=weights, k=1)[0]
        if by_task[task]:
            sampled.append(random.choice(by_task[task]))
    return sampled
```
## Verification
- Epoch 1: only univariate data in training batch
- Epoch 20: batch contains univariate + multivariate + parametric
- Epoch 50: batch contains all 5 task types
- Task ratios in batch match weights (within ±5%)
