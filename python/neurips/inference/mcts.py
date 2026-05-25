"""PUCT-style Monte Carlo Tree Search for the 4-action integration policy.

Uses neural value estimation (no random rollout). The action policy provides
prior probabilities; the value network estimates Q from any state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from neurips.models.action_policy import NUM_ACTIONS


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: str
    parent: MCTSNode | None = None
    action_taken: int | None = None
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    is_solved: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass(frozen=True)
class MCTSConfig:
    """MCTS hyperparameters."""

    n_simulations: int = 100
    puct_c: float = 1.4
    max_depth: int = 10
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25


@dataclass(frozen=True)
class MCTSResult:
    """Result from MCTS search at a root state."""

    solved: bool
    best_action_sequence: list[int]
    visit_counts: list[int]
    policy_target: list[float]
    root_value: float
    antiderivative: str | None = None


def _puct_score(child: MCTSNode, parent_visits: int, c: float) -> float:
    """PUCT selection score: Q + c * prior * sqrt(parent_N) / (1 + child_N)."""
    exploration = c * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return child.q_value + exploration


def _select(node: MCTSNode, config: MCTSConfig) -> MCTSNode:
    """Select leaf by following highest PUCT score."""
    current = node
    while current.children and not current.is_terminal:
        best_score = -float("inf")
        best_child = None
        for child in current.children.values():
            score = _puct_score(child, current.visit_count, config.puct_c)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            break
        current = best_child
    return current


def _expand(
    node: MCTSNode,
    policy_fn: Callable[[str], torch.Tensor],
    env_step_fn: Callable[[str, int], tuple[str, bool, bool]],
) -> None:
    """Expand a leaf node by creating children for all valid actions."""
    priors = policy_fn(node.state)
    for action_id in range(NUM_ACTIONS):
        if priors[action_id] < 1e-8:
            continue
        next_state, terminal, solved = env_step_fn(node.state, action_id)
        child = MCTSNode(
            state=next_state,
            parent=node,
            action_taken=action_id,
            prior=priors[action_id].item(),
            is_terminal=terminal,
            is_solved=solved,
        )
        node.children[action_id] = child


def _backpropagate(node: MCTSNode, value: float) -> None:
    """Propagate value estimate back to root."""
    current: MCTSNode | None = node
    while current is not None:
        current.visit_count += 1
        current.total_value += value
        current = current.parent


def mcts_search(
    root_state: str,
    policy_fn: Callable[[str], torch.Tensor],
    value_fn: Callable[[str], float],
    env_step_fn: Callable[[str, int], tuple[str, bool, bool]],
    config: MCTSConfig = MCTSConfig(),
) -> MCTSResult:
    """Run MCTS from a root state.

    Args:
        root_state: prefix-notation integrand string.
        policy_fn: state → [NUM_ACTIONS] action probabilities.
        value_fn: state → scalar value estimate.
        env_step_fn: (state, action) → (next_state, is_terminal, is_solved).
        config: MCTS hyperparameters.

    Returns:
        MCTSResult with policy targets and best action sequence.
    """
    root = MCTSNode(state=root_state)

    for _ in range(config.n_simulations):
        leaf = _select(root, config)

        if leaf.is_terminal:
            value = 1.0 if leaf.is_solved else -1.0
        else:
            depth = 0
            current: MCTSNode | None = leaf
            while current is not None and current.parent is not None:
                depth += 1
                current = current.parent

            if depth >= config.max_depth:
                value = value_fn(leaf.state)
            else:
                _expand(leaf, policy_fn, env_step_fn)
                value = value_fn(leaf.state)

        _backpropagate(leaf, value)

    visit_counts = [
        root.children[a].visit_count if a in root.children else 0
        for a in range(NUM_ACTIONS)
    ]
    total_visits = sum(visit_counts) or 1

    if config.temperature < 1e-6:
        policy_target = [0.0] * NUM_ACTIONS
        best_action = max(range(NUM_ACTIONS), key=lambda a: visit_counts[a])
        policy_target[best_action] = 1.0
    else:
        counts_temp = [
            c ** (1.0 / config.temperature) for c in visit_counts
        ]
        total_temp = sum(counts_temp) or 1.0
        policy_target = [c / total_temp for c in counts_temp]

    best_sequence = _extract_best_sequence(root)
    solved = any(
        child.is_solved
        for child in root.children.values()
    )

    antiderivative = None
    if solved:
        for child in root.children.values():
            if child.is_solved:
                antiderivative = child.state
                break

    return MCTSResult(
        solved=solved,
        best_action_sequence=best_sequence,
        visit_counts=visit_counts,
        policy_target=policy_target,
        root_value=root.q_value,
        antiderivative=antiderivative,
    )


def _extract_best_sequence(root: MCTSNode) -> list[int]:
    """Follow most-visited children to extract best action sequence."""
    sequence = []
    current = root
    while current.children:
        best_action = max(
            current.children.keys(),
            key=lambda a: current.children[a].visit_count,
        )
        sequence.append(best_action)
        current = current.children[best_action]
        if current.is_terminal:
            break
    return sequence
