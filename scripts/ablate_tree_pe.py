"""Tree GNN PE ablation — tests 4 tree positional encoding variants.

Usage:
    python scripts/ablate_tree_pe.py --output-dir results/tree_pe_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TREE_PE_VARIANTS = ("none", "depth_index", "rwse", "laplacian")
ABLATION_EPOCHS = 10
SUBSET_SIZE = 50_000


def build_model(pe_type: str) -> torch.nn.Module:
    from neurips.models.tree_gnn import TreeIntegrator

    return TreeIntegrator(
        node_vocab=256,
        node_dim=256,
        message_rounds=8,
        decoder_levels=8,
        n_heads=8,
        pe_type=pe_type,
    )


def run_ablation(
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for pe_type in TREE_PE_VARIANTS:
        logger.info("=== Tree PE variant: %s ===", pe_type)
        model = build_model(pe_type).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Parameters: %d", n_params)

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        losses: list[float] = []
        t0 = time.time()

        for epoch in range(1, ABLATION_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            max_nodes = 32

            for _ in range(SUBSET_SIZE // batch_size):
                node_ids = torch.randint(0, 256, (batch_size, max_nodes), device=device)
                adj = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
                for b in range(batch_size):
                    for i in range(1, max_nodes):
                        parent = torch.randint(0, i, (1,)).item()
                        adj[b, parent, i] = 1.0
                        adj[b, i, parent] = 1.0

                depths = torch.randint(0, 8, (batch_size, max_nodes), device=device)
                child_indices = torch.randint(0, 4, (batch_size, max_nodes), device=device)
                targets = torch.randint(0, 256, (batch_size, 64), device=device)

                logits = model(
                    node_ids, adj,
                    depths=depths,
                    child_indices=child_indices,
                )
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            logger.info("  Epoch %d/%d  loss=%.4f", epoch, ABLATION_EPOCHS, avg_loss)

        elapsed = time.time() - t0
        results[pe_type] = {
            "n_params": n_params,
            "final_loss": losses[-1],
            "losses": losses,
            "wall_time_s": elapsed,
            "throughput_samples_per_s": (SUBSET_SIZE * ABLATION_EPOCHS) / elapsed,
        }

    out_path = output_dir / "tree_pe_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    logger.info("\n%-14s %10s %10s %12s", "Tree PE", "Params", "Loss", "Samples/s")
    for pe_type, r in results.items():
        logger.info(
            "%-14s %10d %10.4f %12.0f",
            pe_type, r["n_params"], r["final_loss"], r["throughput_samples_per_s"],
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree PE ablation study")
    parser.add_argument("--output-dir", type=Path, default=Path("results/tree_pe_ablation"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    run_ablation(args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
