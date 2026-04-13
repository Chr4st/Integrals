"""Training loop for the integral prediction Transformer.

Usage:
    python -m integral_engine.train --data data/processed/ --checkpoint checkpoints/
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from integral_engine.model import IntegralTransformer
from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    PAD_INDEX,
    SEQ_TOKEN_TO_INDEX,
    SEQ_VOCAB_SIZE,
    UNK_INDEX,
    feature_schema_hash,
)


class IntegralDataset(Dataset):
    """Dataset of tokenized (integrand, antiderivative) pairs."""

    def __init__(
        self,
        data_dir: str,
        max_input_len: int = MAX_INPUT_LEN,
        max_output_len: int = MAX_OUTPUT_LEN,
    ):
        self.items: List[Dict] = []
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        data_path = Path(data_dir)
        for shard_path in sorted(data_path.glob("shard_*.jsonl.gz")):
            with gzip.open(shard_path, "rt") as f:
                for line in f:
                    item = json.loads(line)
                    # Filter out sequences too long after BOS/EOS
                    if (len(item["input_tokens"]) <= max_input_len - 2
                            and len(item["output_tokens"]) <= max_output_len - 2):
                        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]

        src_indices = [BOS_INDEX]
        for t in item["input_tokens"][: self.max_input_len - 2]:
            src_indices.append(SEQ_TOKEN_TO_INDEX.get(t, UNK_INDEX))
        src_indices.append(EOS_INDEX)

        tgt_indices = [BOS_INDEX]
        for t in item["output_tokens"][: self.max_output_len - 2]:
            tgt_indices.append(SEQ_TOKEN_TO_INDEX.get(t, UNK_INDEX))
        tgt_indices.append(EOS_INDEX)

        depth_feature = torch.tensor(item["depth_feature"], dtype=torch.float32)

        return {
            "src": torch.tensor(src_indices, dtype=torch.long),
            "tgt": torch.tensor(tgt_indices, dtype=torch.long),
            "depth_feature": depth_feature,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad sequences in a batch to uniform length."""
    src_seqs = [item["src"] for item in batch]
    tgt_seqs = [item["tgt"] for item in batch]
    features = torch.stack([item["depth_feature"] for item in batch])

    src_max = max(s.size(0) for s in src_seqs)
    tgt_max = max(t.size(0) for t in tgt_seqs)

    src_padded = torch.full((len(batch), src_max), PAD_INDEX, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max), PAD_INDEX, dtype=torch.long)
    src_mask = torch.ones(len(batch), src_max, dtype=torch.bool)
    tgt_mask = torch.ones(len(batch), tgt_max, dtype=torch.bool)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : s.size(0)] = s
        src_mask[i, : s.size(0)] = False
        tgt_padded[i, : t.size(0)] = t
        tgt_mask[i, : t.size(0)] = False

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "depth_feature": features,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def compute_class_weights(dataset: IntegralDataset, cap: float = 10.0) -> torch.Tensor:
    """Compute inverse-frequency class weights for output tokens, capped."""
    counts: Counter = Counter()
    for item in dataset.items:
        for t in item["output_tokens"]:
            idx = SEQ_TOKEN_TO_INDEX.get(t, UNK_INDEX)
            counts[idx] += 1

    total = sum(counts.values())
    weights = torch.ones(SEQ_VOCAB_SIZE)
    for idx, count in counts.items():
        w = total / (SEQ_VOCAB_SIZE * count)
        weights[idx] = min(w, cap)

    return weights


class WarmupInvSqrtScheduler:
    """Linear warmup then inverse square root decay."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            lr = self.base_lr * math.sqrt(self.warmup_steps / self.step_num)

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


def train_epoch(
    model: IntegralTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInvSqrtScheduler,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        feat = batch["depth_feature"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_input_mask = tgt_mask[:, :-1]

        logits = model(src, tgt_input, feat, src_mask, tgt_input_mask)

        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat) / accumulation_steps
        loss.backward()

        non_pad = target_flat != PAD_INDEX
        total_loss += loss.item() * accumulation_steps * non_pad.sum().item()
        total_tokens += non_pad.sum().item()
        preds = logits_flat.argmax(dim=-1)
        correct_tokens += ((preds == target_flat) & non_pad).sum().item()

        is_accumulation_boundary = (
            (step + 1) % accumulation_steps == 0
            or (step + 1) == len(dataloader)
        )
        if is_accumulation_boundary:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
    }


@torch.no_grad()
def evaluate(
    model: IntegralTransformer,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_matches = 0
    total_seqs = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        feat = batch["depth_feature"].to(device)
        src_mask = batch["src_mask"].to(device)
        tgt_mask = batch["tgt_mask"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]
        tgt_input_mask = tgt_mask[:, :-1]

        logits = model(src, tgt_input, feat, src_mask, tgt_input_mask)

        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        non_pad = target_flat != PAD_INDEX
        total_loss += loss.item() * non_pad.sum().item()
        total_tokens += non_pad.sum().item()
        preds = logits_flat.argmax(dim=-1)
        correct_tokens += ((preds == target_flat) & non_pad).sum().item()

        # Exact match per sequence
        preds_2d = preds.reshape(tgt_target.shape)
        non_pad_2d = tgt_target != PAD_INDEX
        for i in range(tgt_target.size(0)):
            mask = non_pad_2d[i]
            if torch.all(preds_2d[i][mask] == tgt_target[i][mask]):
                exact_matches += 1
            total_seqs += 1

    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_matches / max(total_seqs, 1),
    }


def train(
    data_dir: str,
    checkpoint_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 4e-5,
    patience: int = 5,
    device_str: str = "auto",
    accumulation_steps: int = 1,
    max_seq_len: int = MAX_INPUT_LEN,
) -> None:
    """Full training pipeline."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    effective_batch = batch_size * accumulation_steps
    print(f"Using device: {device}")
    print(f"Micro-batch: {batch_size}, Accumulation: {accumulation_steps}, "
          f"Effective batch: {effective_batch}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    use_pin_memory = device.type == "cuda"
    num_workers = 2 if device.type == "cuda" else 0

    print("Loading datasets...")
    train_dataset = IntegralDataset(
        os.path.join(data_dir, "train"),
        max_input_len=max_seq_len,
        max_output_len=max_seq_len,
    )
    val_dataset = IntegralDataset(
        os.path.join(data_dir, "val"),
        max_input_len=max_seq_len,
        max_output_len=max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=use_pin_memory,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = IntegralTransformer().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=PAD_INDEX)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9,
    )
    scheduler = WarmupInvSqrtScheduler(optimizer, warmup_steps=4000)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            accumulation_steps=accumulation_steps,
        )
        print(
            f"Train: loss={train_metrics['loss']:.4f}, "
            f"token_acc={train_metrics['token_accuracy']:.4f}"
        )

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"Val:   loss={val_metrics['loss']:.4f}, "
            f"token_acc={val_metrics['token_accuracy']:.4f}, "
            f"exact_match={val_metrics['exact_match']:.4f}"
        )

        # Phase 7: Save rolling last-5 epoch checkpoints (for weight averaging)
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt")
        arch_hash = hashlib.sha256(
            json.dumps(sorted(model.state_dict().keys())).encode(),
        ).hexdigest()[:16]
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "architecture_hash": arch_hash,
            "feature_schema_hash": feature_schema_hash(),
        }
        torch.save(ckpt_data, epoch_ckpt_path)

        # Remove old epoch checkpoints beyond the last 5
        epoch_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
        for old_ckpt in epoch_files[:-5]:
            old_ckpt.unlink()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(ckpt_data, best_path)
            print(f"  Saved best checkpoint (val_loss={best_val_loss:.4f}, "
                  f"exact_match={val_metrics['exact_match']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train integral Transformer")
    parser.add_argument("--data", required=True, help="Processed data directory")
    parser.add_argument(
        "--checkpoint", default="checkpoints/", help="Checkpoint directory",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seq-len", type=int, default=MAX_INPUT_LEN,
                        help="Max sequence length (filters longer examples)")
    args = parser.parse_args()

    train(
        args.data, args.checkpoint, args.epochs, args.batch_size,
        args.lr, args.patience, args.device, args.accumulation_steps,
        args.max_seq_len,
    )
