"""Monte Carlo Tree Search decoder for symbolic integration.

Replaces beam search / temperature sampling with MCTS where terminal
reward is SymPy verification: diff(F, x) == f.

Select -> Expand -> Simulate -> Evaluate -> Backpropagate.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import sympy as sp
import torch
import torch.nn.functional as F

from integral_engine.grammar_mask import PrefixGrammarMask
from integral_engine.tokenizer import from_prefix
from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    SEQ_TOKENS,
    SEQ_VOCAB_SIZE,
)


class MCTSNode:
    """Single node in the MCTS token tree.

    Each node represents a token choice at a particular position
    in the generated sequence.
    """

    __slots__ = (
        "token_id",
        "parent",
        "children",
        "visit_count",
        "total_value",
        "prior",
    )

    def __init__(
        self,
        token_id: int,
        parent: Optional[MCTSNode] = None,
        prior: float = 0.0,
    ) -> None:
        self.token_id = token_id
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0
        self.prior = prior

    def ucb1(self, exploration_c: float = 1.414) -> float:
        """Upper Confidence Bound for Trees.

        Balances exploitation (mean value) with exploration
        (inverse sqrt of visit ratio). Unvisited nodes return inf
        to guarantee exploration.
        """
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count if self.parent else 1
        exploitation = self.total_value / self.visit_count
        exploration = exploration_c * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """A leaf has no expanded children."""
        return len(self.children) == 0

    def best_child(self, exploration_c: float = 1.414) -> MCTSNode:
        """Return child with highest UCB1 score."""
        if not self.children:
            raise ValueError("No children to select from")
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_c),
        )


def _extract_sequence(node: MCTSNode) -> List[int]:
    """Walk from node up to root, return token sequence (root-first)."""
    tokens: List[int] = []
    current: Optional[MCTSNode] = node
    while current is not None:
        tokens.append(current.token_id)
        current = current.parent
    tokens.reverse()
    return tokens


def _verify_sequence(
    token_ids: List[int],
    integrand_expr: sp.Basic,
    var: sp.Symbol,
) -> float:
    """Verify a candidate antiderivative via SymPy differentiation.

    Returns 1.0 if diff(F, x) simplifies to f, 0.0 otherwise.
    """
    # Strip special tokens and convert to string tokens
    str_tokens = [
        SEQ_TOKENS[idx]
        for idx in token_ids
        if idx not in (BOS_INDEX, EOS_INDEX, PAD_INDEX)
        and 0 <= idx < len(SEQ_TOKENS)
    ]
    if not str_tokens:
        return 0.0
    try:
        candidate = from_prefix(str_tokens)
        derivative = sp.diff(candidate, var)
        diff = sp.simplify(derivative - integrand_expr)
        if diff == 0:
            return 1.0
    except Exception:
        pass
    return 0.0


class MCTSDecoder:
    """MCTS-based autoregressive decoder for the IntegralTransformer.

    Searches the token tree using UCB1 selection, model-guided
    expansion, greedy rollout simulation, and SymPy verification
    as terminal reward.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        grammar_mask: Optional[PrefixGrammarMask] = None,
        exploration_c: float = 1.414,
        max_simulations: int = 200,
        max_len: int = 256,
        top_k: int = 20,
    ) -> None:
        self.model = model
        self.grammar_mask = grammar_mask
        self.exploration_c = exploration_c
        self.max_simulations = max_simulations
        self.max_len = max_len
        self.top_k = top_k

    @torch.no_grad()
    def search(
        self,
        src: torch.Tensor,
        depth_features: torch.Tensor,
        integrand_expr: sp.Basic,
        var: sp.Symbol,
    ) -> List[Tuple[List[int], float]]:
        """Run MCTS and return completed sequences sorted by visit count.

        Args:
            src: Encoded source tokens, shape (1, seq_len).
            depth_features: Depth feature vector, shape (1, feature_dim).
            integrand_expr: The integrand as a SymPy expression.
            var: Integration variable as a SymPy Symbol.

        Returns:
            List of (token_id_sequence, reward) tuples, sorted by
            descending visit count. Reward is 1.0 for verified
            antiderivatives, 0.0 otherwise.
        """
        self.model.eval()
        device = src.device

        # Encode source once (shared across all simulations)
        memory = self.model.encode(src, depth_features)

        root = MCTSNode(token_id=BOS_INDEX)
        completed: List[Tuple[List[int], float, int]] = []

        for _ in range(self.max_simulations):
            # --- SELECT ---
            node = root
            while not node.is_leaf() and node.token_id != EOS_INDEX:
                node = node.best_child(self.exploration_c)

            # Skip already-terminal nodes
            if node.token_id == EOS_INDEX:
                continue

            seq_so_far = _extract_sequence(node)
            if len(seq_so_far) >= self.max_len:
                continue

            # --- EXPAND ---
            self._expand(node, seq_so_far, memory, device)

            if not node.children:
                continue

            # Pick the child with highest prior for simulation
            sim_child = max(
                node.children.values(), key=lambda c: c.prior
            )

            # --- SIMULATE ---
            rollout_seq = self._simulate(
                sim_child, seq_so_far, memory, device
            )

            # --- EVALUATE ---
            reward = _verify_sequence(
                rollout_seq, integrand_expr, var
            )

            if (
                len(rollout_seq) > 0
                and rollout_seq[-1] == EOS_INDEX
            ):
                completed.append((
                    rollout_seq, reward, sim_child.visit_count + 1
                ))

            # --- BACKPROPAGATE ---
            self._backpropagate(sim_child, reward)

        # Sort completed sequences by visit count (descending)
        completed.sort(key=lambda x: x[2], reverse=True)
        return [(seq, reward) for seq, reward, _ in completed]

    def _expand(
        self,
        node: MCTSNode,
        seq_so_far: List[int],
        memory: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Expand a leaf node by creating children for top-K tokens."""
        tgt = torch.tensor([seq_so_far], device=device)
        logits = self.model.decode(tgt, memory)
        # logits shape: (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]

        # Apply grammar mask if provided
        if self.grammar_mask is not None:
            grammar_state = PrefixGrammarMask(device)
            # Replay sequence to build grammar state (skip BOS)
            for tid in seq_so_far[1:]:
                grammar_state.step(tid)
            valid_mask = grammar_state.valid_token_mask()
            next_logits = next_logits.masked_fill(~valid_mask, float("-inf"))

        # Softmax to get priors
        priors = F.softmax(next_logits, dim=-1)

        # Select top-K tokens
        k = min(self.top_k, (priors > 0).sum().item())
        if k == 0:
            return
        topk_priors, topk_indices = priors.topk(int(k))

        for prior_val, token_idx in zip(
            topk_priors.tolist(), topk_indices.tolist()
        ):
            child = MCTSNode(
                token_id=token_idx,
                parent=node,
                prior=prior_val,
            )
            node.children[token_idx] = child

    def _simulate(
        self,
        start_node: MCTSNode,
        parent_seq: List[int],
        memory: torch.Tensor,
        device: torch.device,
    ) -> List[int]:
        """Greedy rollout from start_node to EOS or max_len."""
        seq = parent_seq + [start_node.token_id]

        for _ in range(self.max_len - len(seq)):
            if seq[-1] == EOS_INDEX:
                break

            tgt = torch.tensor([seq], device=device)
            logits = self.model.decode(tgt, memory)
            next_logits = logits[0, -1, :]

            # Greedy: pick argmax
            next_token = next_logits.argmax().item()
            seq.append(next_token)

        return seq

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        """Update visit counts and total values up to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent
