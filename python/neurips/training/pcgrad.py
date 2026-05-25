"""PCGrad — Projecting Conflicting Gradients (Yu et al., NeurIPS 2020).

Eliminates gradient interference between tasks by projecting each
task's gradient away from conflicting directions.

Uses torch.autograd.grad() to avoid mutating .grad buffers, which
preserves compatibility with gradient accumulation.
"""

from __future__ import annotations

import random

import torch


def pcgrad_backward(
    losses: list[torch.Tensor],
    model: torch.nn.Module,
) -> None:
    """Compute PCGrad-corrected gradients and ADD to existing .grad buffers.

    Uses torch.autograd.grad() for per-task gradient computation to avoid
    clearing accumulated gradients from prior microbatches.

    Args:
        losses: Per-task scalar losses (already scaled by accumulation).
        model: Module whose .parameters() receive the corrected gradient.
    """
    if len(losses) <= 1:
        losses[0].backward()
        return

    params = [p for p in model.parameters() if p.requires_grad]

    # Compute per-task gradients without mutating .grad.
    task_grads: list[list[torch.Tensor]] = []
    for loss in losses:
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True,
        )
        task_grads.append([
            g.clone() if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, params)
        ])

    # Project gradients: for each task, project away conflicts with others.
    n_tasks = len(task_grads)
    corrected: list[list[torch.Tensor]] = [
        [g.clone() for g in grads] for grads in task_grads
    ]

    order = list(range(n_tasks))
    random.shuffle(order)

    for i in order:
        for j in order:
            if i == j:
                continue
            gi_flat = torch.cat([g.reshape(-1) for g in corrected[i]])
            gj_flat = torch.cat([g.reshape(-1) for g in task_grads[j]])
            dot = torch.dot(gi_flat, gj_flat)

            if dot < 0:
                gj_norm_sq = torch.dot(gj_flat, gj_flat)
                if gj_norm_sq > 0:
                    coeff = dot / gj_norm_sq
                    for k in range(len(corrected[i])):
                        corrected[i][k] = corrected[i][k] - coeff * task_grads[j][k]

    # Average corrected gradients and ADD to existing .grad (not replace).
    for p_idx, p in enumerate(params):
        avg_grad = sum(corrected[t][p_idx] for t in range(n_tasks)) / n_tasks
        if p.grad is not None:
            p.grad.add_(avg_grad)
        else:
            p.grad = avg_grad
