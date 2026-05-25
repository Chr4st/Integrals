"""Diverse beam search with grammar-constrained decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from neurips.data.vocab import EOS, SOS
from neurips.models.grammar import PrefixGrammarMask


@dataclass(frozen=True)
class BeamCandidate:
    """Single beam search candidate."""

    tokens: list[int]
    log_prob: float
    finished: bool


def diverse_beam_search(
    decoder: torch.nn.Module,
    memory: torch.Tensor,
    tokenizer: object,
    beam_width: int = 10,
    n_groups: int = 2,
    diversity_penalty: float = 0.5,
    max_len: int = 512,
) -> list[str]:
    """Run diverse beam search with grammar constraints.

    Args:
        decoder: SeqDecoder module with output_proj.
        memory: [1, src_len, d_model] encoder output.
        tokenizer: Tokenizer with decode method.
        beam_width: Total beams (split across groups).
        n_groups: Number of diversity groups.
        diversity_penalty: Penalty for tokens already chosen by prior groups.
        max_len: Maximum output length.

    Returns:
        List of decoded candidate strings, ranked by log-probability.
    """
    device = memory.device
    group_size = beam_width // n_groups

    # Initialize: one beam per group, all starting with SOS.
    groups: list[list[BeamCandidate]] = [
        [BeamCandidate(tokens=[SOS], log_prob=0.0, finished=False)]
        for _ in range(n_groups)
    ]

    for _step in range(max_len):
        all_finished = all(
            all(c.finished for c in group) for group in groups
        )
        if all_finished:
            break

        # Track tokens selected by earlier groups at this step.
        prior_tokens: list[torch.Tensor] = []

        for g_idx, group in enumerate(groups):
            active = [c for c in group if not c.finished]
            finished = [c for c in group if c.finished]

            if not active:
                groups[g_idx] = finished
                continue

            # Batch decode all active beams in this group.
            tgt_batch = torch.tensor(
                [c.tokens for c in active], device=device
            )
            logits = decoder(tgt_batch, memory.expand(len(active), -1, -1))
            step_logits = logits[:, -1, :]  # [n_active, vocab]

            # Apply grammar constraints per beam.
            for i, cand in enumerate(active):
                grammar = PrefixGrammarMask()
                for tok in cand.tokens:
                    mask = grammar.step(tok)
                step_logits[i, ~mask] = float("-inf")

            log_probs = F.log_softmax(step_logits, dim=-1)

            # Apply diversity penalty from prior groups.
            if prior_tokens and diversity_penalty > 0:
                penalty = torch.zeros_like(log_probs[0])
                for pt in prior_tokens:
                    penalty.scatter_add_(0, pt, torch.ones_like(pt, dtype=penalty.dtype))
                log_probs = log_probs - diversity_penalty * penalty.unsqueeze(0)

            # Expand each active beam.
            all_candidates: list[BeamCandidate] = list(finished)
            for i, cand in enumerate(active):
                top_k = min(group_size * 2, log_probs.size(-1))
                values, indices = log_probs[i].topk(top_k)
                for v, idx in zip(values.tolist(), indices.tolist()):
                    new_lp = cand.log_prob + v
                    new_tokens = cand.tokens + [idx]
                    is_done = idx == EOS
                    all_candidates.append(
                        BeamCandidate(new_tokens, new_lp, is_done)
                    )

            # Keep top group_size by log_prob.
            all_candidates.sort(key=lambda c: c.log_prob, reverse=True)
            groups[g_idx] = all_candidates[:group_size]

            # Record tokens chosen by this group for diversity.
            chosen = torch.tensor(
                [c.tokens[-1] for c in groups[g_idx] if not c.finished],
                device=device,
            )
            if chosen.numel() > 0:
                prior_tokens.append(chosen)

    # Collect all candidates across groups, deduplicate, rank.
    all_results: list[BeamCandidate] = []
    for group in groups:
        all_results.extend(group)

    # Length-normalize log probs.
    all_results.sort(
        key=lambda c: c.log_prob / max(len(c.tokens), 1), reverse=True
    )

    seen: set[str] = set()
    decoded: list[str] = []
    for cand in all_results:
        prefix_str, _ = tokenizer.decode(cand.tokens)
        if prefix_str and prefix_str not in seen:
            seen.add(prefix_str)
            decoded.append(prefix_str)

    return decoded
