"""Action-space search strategies for step-by-step integration (Sprint 3).

Headline: greedy policy (1 rollout per integrand).
Appendix ablation: beam search (width=4).
MCTS deferred to Sprint 4b.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from neurips.models.action_policy import NUM_ACTIONS


@dataclass(frozen=True)
class ActionTrace:
    """A single action in the proof trace."""

    action_id: int
    action_name: str
    state_before: str
    state_after: str
    verify_ok: bool


@dataclass
class SearchResult:
    """Result of an action-space search."""

    solved: bool
    antiderivative: str | None = None
    trace: list[ActionTrace] = field(default_factory=list)
    total_steps: int = 0
    reward: float = 0.0


ACTION_NAMES = ["substitute", "integrate_by_parts", "partial_fractions", "close"]


def greedy_search(
    policy: Any,
    encoder: Any,
    env_factory: Any,
    integrand_prefix: str,
    var: str = "x",
    max_steps: int = 10,
    device: str = "cpu",
) -> SearchResult:
    """Greedy action-space search (headline strategy).

    At each step, select the highest-probability action and execute it.
    The policy network provides action logits; the Rust env validates.

    Args:
        policy: ActionPolicy model.
        encoder: Tree GNN encoder that produces pooled embeddings.
        env_factory: Callable that creates IntegrationEnv instances.
        integrand_prefix: Prefix string of the integrand.
        var: Integration variable.
        max_steps: Maximum number of steps before giving up.
        device: Torch device.

    Returns:
        SearchResult with trace and solution (if found).
    """
    env = env_factory(integrand_prefix, var, max_steps)
    trace: list[ActionTrace] = []

    for step in range(max_steps):
        if env.is_done():
            break

        # Get state features and encode
        features = torch.tensor(
            env.state_features(), dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            # Use features directly as embedding (feature dim → embed dim
            # projection handled by the policy's first linear layer)
            logits = policy(features)
            action_id = logits.argmax(dim=-1).item()

        # Get current state from the environment
        state_before = (
            env.current_state_prefix()
            if hasattr(env, "current_state_prefix")
            else integrand_prefix
        )

        # Execute the selected action
        try:
            if action_id == 3:  # close
                reward, done, verify_ok = env.step_close(state_before)
            elif action_id == 0:  # substitute
                reward, done, verify_ok = env.step_substitute(state_before)
            elif action_id == 1:  # IBP
                reward, done, verify_ok = env.step_ibp(
                    state_before, state_before
                )
            else:  # partial_fractions — no-op in simplified env
                reward, done, verify_ok = 0.0, False, True
        except Exception:
            break

        state_after = (
            env.current_state_prefix()
            if hasattr(env, "current_state_prefix")
            else integrand_prefix
        )

        trace.append(ActionTrace(
            action_id=action_id,
            action_name=ACTION_NAMES[action_id],
            state_before=state_before,
            state_after=state_after,
            verify_ok=verify_ok,
        ))

        if done and verify_ok:
            return SearchResult(
                solved=True,
                antiderivative=state_after,
                trace=trace,
                total_steps=step + 1,
                reward=reward,
            )

    return SearchResult(
        solved=False,
        trace=trace,
        total_steps=len(trace),
    )


def beam_search(
    policy: Any,
    encoder: Any,
    env_factory: Any,
    integrand_prefix: str,
    var: str = "x",
    beam_width: int = 4,
    max_steps: int = 10,
    device: str = "cpu",
) -> SearchResult:
    """Beam search over action sequences (appendix ablation).

    Maintains `beam_width` parallel rollouts, expanding the top-k
    actions at each step. Returns the first successful trace.
    """
    # Each beam entry: (env, trace, cumulative_log_prob)
    beams: list[tuple[Any, list[ActionTrace], float]] = [
        (env_factory(integrand_prefix, var, max_steps), [], 0.0)
    ]

    for step in range(max_steps):
        candidates: list[tuple[Any, list[ActionTrace], float]] = []

        for env, trace, cum_logp in beams:
            if env.is_done():
                continue

            features = torch.tensor(
                env.state_features(), dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits = policy(features)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            # Expand top-k actions
            topk = min(beam_width, NUM_ACTIONS)
            top_logps, top_ids = log_probs.topk(topk)

            for i in range(topk):
                action_id = top_ids[i].item()
                action_logp = top_logps[i].item()

                # Clone env state for this branch
                import copy
                env_copy = copy.deepcopy(env)

                cur_state = (
                    env_copy.current_state_prefix()
                    if hasattr(env_copy, "current_state_prefix")
                    else integrand_prefix
                )

                try:
                    if action_id == 3:
                        reward, done, verify_ok = env_copy.step_close(
                            cur_state
                        )
                    elif action_id == 0:
                        reward, done, verify_ok = env_copy.step_substitute(
                            cur_state
                        )
                    elif action_id == 1:
                        reward, done, verify_ok = env_copy.step_ibp(
                            cur_state, cur_state
                        )
                    else:
                        reward, done, verify_ok = 0.0, False, True
                except Exception:
                    continue

                state_after = (
                    env_copy.current_state_prefix()
                    if hasattr(env_copy, "current_state_prefix")
                    else integrand_prefix
                )

                new_trace = trace + [ActionTrace(
                    action_id=action_id,
                    action_name=ACTION_NAMES[action_id],
                    state_before=cur_state,
                    state_after=state_after,
                    verify_ok=verify_ok,
                )]

                if done and verify_ok:
                    return SearchResult(
                        solved=True,
                        antiderivative=state_after,
                        trace=new_trace,
                        total_steps=step + 1,
                        reward=reward,
                    )

                candidates.append((
                    env_copy,
                    new_trace,
                    cum_logp + action_logp,
                ))

        if not candidates:
            break

        # Keep top beam_width candidates by cumulative log probability
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]

    return SearchResult(solved=False, total_steps=max_steps)
