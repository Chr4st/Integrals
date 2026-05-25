"""Self-play curriculum for action-space training (Sprint 3).

Generates training trajectories by sampling action chains of length
L ∈ {1, 2, 3}, applying them through the Rust env to build
(integrand, action_chain → F) pairs. Difficulty auto-escalates
when the policy saturates short chains.

Supports both REINFORCE (default) and MCTS-guided self-play.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from neurips.inference.mcts import MCTSConfig, MCTSResult, mcts_search


@dataclass(frozen=True)
class Trajectory:
    """A single self-play trajectory."""

    integrand_prefix: str
    antiderivative_prefix: str
    action_ids: list[int]
    rewards: list[float]
    total_reward: float
    chain_length: int
    step_features: list[list[float]] = field(default_factory=list)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play data generation."""

    min_chain_length: int = 1
    max_chain_length: int = 3
    episodes_per_batch: int = 256
    complexity_threshold: float = 0.8
    escalation_step: int = 1000
    max_steps_per_episode: int = 10


@dataclass
class CurriculumState:
    """Tracks curriculum difficulty and escalation."""

    current_min_length: int = 1
    current_max_length: int = 1
    episodes_completed: int = 0
    recent_solve_rate: float = 0.0
    solve_history: list[bool] = field(default_factory=list)

    def update(self, solved: bool, config: SelfPlayConfig) -> None:
        self.episodes_completed += 1
        self.solve_history.append(solved)

        # Keep rolling window of last 100 episodes
        if len(self.solve_history) > 100:
            self.solve_history = self.solve_history[-100:]

        self.recent_solve_rate = (
            sum(self.solve_history) / len(self.solve_history)
        )

        # Escalate difficulty when solve rate is high
        if (
            self.recent_solve_rate > config.complexity_threshold
            and self.episodes_completed % config.escalation_step == 0
            and self.current_max_length < config.max_chain_length
        ):
            self.current_max_length = min(
                self.current_max_length + 1,
                config.max_chain_length,
            )


def generate_trajectories(
    policy: Any,
    env_factory: Any,
    integrands: list[str],
    config: SelfPlayConfig,
    curriculum: CurriculumState,
    device: str = "cpu",
) -> list[Trajectory]:
    """Generate a batch of self-play trajectories.

    Args:
        policy: ActionPolicy model (used for action selection).
        env_factory: Callable(prefix, var, max_steps) -> IntegrationEnv.
        integrands: List of integrand prefix strings to sample from.
        config: Self-play configuration.
        curriculum: Mutable curriculum state (updated in-place).
        device: Torch device.

    Returns:
        List of completed trajectories.
    """
    trajectories: list[Trajectory] = []

    for _ in range(config.episodes_per_batch):
        integrand = random.choice(integrands)
        chain_length = random.randint(
            curriculum.current_min_length,
            curriculum.current_max_length,
        )

        traj = _run_episode(
            policy=policy,
            env_factory=env_factory,
            integrand_prefix=integrand,
            target_length=chain_length,
            max_steps=config.max_steps_per_episode,
            device=device,
        )

        if traj is not None:
            trajectories.append(traj)
            curriculum.update(traj.total_reward > 0, config)
        else:
            curriculum.update(False, config)

    return trajectories


def _run_episode(
    policy: Any,
    env_factory: Any,
    integrand_prefix: str,
    target_length: int,
    max_steps: int,
    device: str,
) -> Trajectory | None:
    """Run a single self-play episode."""
    env = env_factory(integrand_prefix, "x", max_steps)
    action_ids: list[int] = []
    rewards: list[float] = []
    step_features: list[list[float]] = []

    for step in range(max_steps):
        if env.is_done():
            break

        raw_features = env.state_features()
        step_features.append(raw_features)
        features = torch.tensor(
            raw_features, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            logits = policy(features)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action_id = dist.sample().item()

        cur_state = (
            env.current_state_prefix()
            if hasattr(env, "current_state_prefix")
            else integrand_prefix
        )

        try:
            if action_id == 3:  # close
                reward, done, verify_ok = env.step_close(cur_state)
            elif action_id == 0:  # substitute
                reward, done, verify_ok = env.step_substitute(cur_state)
            elif action_id == 1:  # IBP
                reward, done, verify_ok = env.step_ibp(
                    cur_state, cur_state
                )
            else:
                reward, done, verify_ok = 0.0, False, True
        except Exception:
            return None

        action_ids.append(action_id)
        rewards.append(reward)

        if done:
            break

    if not action_ids:
        return None

    final_state = (
        env.current_state_prefix()
        if hasattr(env, "current_state_prefix")
        else integrand_prefix
    )

    return Trajectory(
        integrand_prefix=integrand_prefix,
        antiderivative_prefix=final_state,
        action_ids=action_ids,
        rewards=rewards,
        total_reward=sum(rewards),
        chain_length=len(action_ids),
        step_features=step_features,
    )


@dataclass
class MCTSSelfPlayConfig:
    """Configuration for MCTS-guided self-play data generation."""

    episodes_per_batch: int = 64
    mcts: MCTSConfig = field(default_factory=MCTSConfig)


def generate_mcts_trajectories(
    policy_fn: Callable[[str], torch.Tensor],
    value_fn: Callable[[str], float],
    env_step_fn: Callable[[str, int], tuple[str, bool, bool]],
    integrands: list[str],
    config: MCTSSelfPlayConfig = MCTSSelfPlayConfig(),
) -> list[tuple[str, list[float], float]]:
    """Generate MCTS policy targets for training.

    Returns list of (integrand, policy_target, root_value) tuples.
    """
    results: list[tuple[str, list[float], float]] = []

    for _ in range(config.episodes_per_batch):
        integrand = random.choice(integrands)
        mcts_result = mcts_search(
            root_state=integrand,
            policy_fn=policy_fn,
            value_fn=value_fn,
            env_step_fn=env_step_fn,
            config=config.mcts,
        )
        results.append((integrand, mcts_result.policy_target, mcts_result.root_value))

    return results


def compute_mcts_policy_loss(
    policy: Any,
    mcts_targets: list[tuple[str, list[float], float]],
    feature_fn: Callable[[str], torch.Tensor],
    device: str = "cpu",
    value_head: Any = None,
) -> torch.Tensor:
    """Cross-entropy loss between policy output and MCTS visit distribution.

    Also trains value head if provided.
    """
    if not mcts_targets:
        return torch.tensor(0.0, device=device, requires_grad=True)

    all_features = []
    all_targets = []
    all_values = []

    for integrand, policy_target, root_value in mcts_targets:
        feat = feature_fn(integrand)
        all_features.append(feat)
        all_targets.append(torch.tensor(policy_target, device=device))
        all_values.append(root_value)

    features = torch.stack(all_features).to(device)
    targets = torch.stack(all_targets)
    logits = policy(features)
    log_probs = torch.log_softmax(logits, dim=-1)
    policy_loss = -(targets * log_probs).sum(dim=-1).mean()

    if value_head is not None:
        values_pred = value_head(features).squeeze(-1)
        values_target = torch.tensor(all_values, device=device, dtype=torch.float32)
        value_loss = torch.nn.functional.mse_loss(values_pred, values_target)
        return policy_loss + value_loss

    return policy_loss


def compute_policy_loss(
    policy: Any,
    trajectories: list[Trajectory],
    device: str = "cpu",
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute REINFORCE policy gradient loss from trajectories.

    Recomputes log-probabilities from the policy to ensure gradients
    flow back to the model parameters. Uses discounted returns with
    a mean-return baseline for variance reduction.
    """
    all_log_probs: list[torch.Tensor] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []

    # First pass: compute all discounted returns
    for traj in trajectories:
        if not traj.action_ids:
            continue
        returns: list[float] = []
        g = 0.0
        for r in reversed(traj.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        all_returns.extend(returns)

    if not all_returns:
        return torch.tensor(0.0, device=device, requires_grad=True)

    baseline = sum(all_returns) / len(all_returns)

    # Second pass: recompute log-probs from the policy
    return_idx = 0
    for traj in trajectories:
        if not traj.action_ids:
            continue
        n_steps = len(traj.action_ids)
        if traj.step_features and len(traj.step_features) == n_steps:
            features = torch.tensor(
                traj.step_features, dtype=torch.float32, device=device,
            )
        else:
            embed_dim = policy.mlp[0].in_features
            features = torch.zeros(n_steps, embed_dim, device=device)
        logits = policy(features)
        log_probs = torch.log_softmax(logits, dim=-1)

        for step_i in range(n_steps):
            action_id = traj.action_ids[step_i]
            lp = log_probs[step_i, action_id]
            all_log_probs.append(lp)
            all_advantages.append(all_returns[return_idx] - baseline)
            return_idx += 1

    if not all_log_probs:
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_prob_stack = torch.stack(all_log_probs)
    adv_tensor = torch.tensor(
        all_advantages, device=device, dtype=torch.float32,
    )
    loss = -(log_prob_stack * adv_tensor.detach()).mean()
    return loss
