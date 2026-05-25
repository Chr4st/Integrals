# P4-F005: MCTS for Action Policy

**Status**: draft
**Priority**: P1 (was Sprint 4b, evidence supports pulling forward)

## Problem

Action policy uses greedy search with optional beam search (width=4) for step-by-step integration. 4-action discrete space (substitute, integrate_by_parts, partial_fractions, close) is textbook MCTS territory. MCTS with neural value estimation finds higher-quality action sequences than greedy/beam by exploring the full decision tree.

Current self-play generates trajectories by sampling actions — no lookahead or backpropagation of long-term value.

## Solution

### Phase A: Value Network

Add a value head to the existing tree GNN encoder:
- Input: pooled tree embedding (256-dim) of current integrand state
- Output: scalar value estimate (predicted probability of reaching correct antiderivative from this state)
- Train with Monte Carlo returns from completed episodes

### Phase B: MCTS Engine

Implement PUCT-style MCTS (AlphaZero variant):
- **Selection**: UCB1 + neural prior (action policy logits as prior probabilities)
- **Expansion**: apply action via Rust env, get new state
- **Simulation**: neural value network evaluation (no random rollout)
- **Backpropagation**: update visit counts and value estimates up the tree
- N_simulations = 100 per root state (configurable)

### Phase C: Cached Verification

Per-step CAS verification is CPU-bound. Mitigate:
- Hash-based cache: `(expression_prefix, action_id) → result` stored in LRU cache
- Rust-side numerical spot-check: evaluate at 5 random points, compare derivative vs original (fast filter, SymPy only for uncertain cases)
- Batch SymPy calls for leaf evaluations

### Phase D: Training Loop Integration

- Use MCTS to generate training data: run MCTS at root, collect (state, policy_target, value_target) tuples
- Policy target = normalized visit counts (not raw action logits)
- Value target = episode outcome (+1 correct, -1 timeout/wrong, intermediate from n-step returns)
- Interleave MCTS data generation with gradient updates (AlphaZero-style)

## Acceptance Criteria

- [ ] Value network head added to tree GNN (256→1 MLP)
- [ ] PUCT-MCTS implementation with configurable N_simulations
- [ ] Hash-based verification cache with hit rate ≥50% during MCTS
- [ ] Rust-side numerical spot-check filter (5-point evaluation)
- [ ] MCTS solves ≥20% more test integrands than greedy search
- [ ] Training loop alternates between MCTS data gen and policy/value updates
- [ ] Wall-clock training time ≤3x greedy (with caching)
- [ ] Property test: MCTS never selects invalid actions (grammar constraints preserved)

## Affected Files

- `python/neurips/models/tree_gnn.py` — add value head
- NEW: `python/neurips/inference/mcts.py` — MCTS engine
- `python/neurips/inference/action_search.py` — integrate MCTS as search strategy
- `python/neurips/training/self_play.py` — generate MCTS-guided trajectories
- `rust/core/src/verify.rs` — add numerical spot-check function
- NEW: `python/neurips/inference/verify_cache.py` — LRU verification cache
- `configs/default.toml` — add `[inference.mcts]` section

## Risks

- SymPy verification at each MCTS node could be 10-50x slower without caching
- Value network quality bootstraps from random → need careful warm-start
- 100 simulations * max_depth=10 * 4 actions = up to 4000 SymPy calls per root → cache is critical
