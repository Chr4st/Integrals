# Phase 12a: Training Loop

## Goal
Build the training pipeline that trains BOTH architectures on the same
data. Fair comparison requires identical data, schedule, and evaluation.

## What Training Does (Simply)
1. Show the model an integral (input)
2. The model guesses the answer (prediction)
3. Compare guess to real answer → a number called "loss"
   (high loss = bad guess, low loss = good guess)
4. Adjust model's numbers to reduce the loss (gradient update)
5. Repeat 920K × 60 = 55 million times

## File: `python/neurips/training/trainer.py`

### Training Step (one batch)
```python
def train_step(model, batch, optimizer, model_type):
    optimizer.zero_grad()  # clear old gradients
    
    if model_type == "seq":
        logits = model(
            batch["src_ids"],      # integrand tokens [batch, src_len]
            batch["tgt_ids"][:, :-1],  # answer tokens (shifted by 1)
            batch["features"],     # 688-dim structural features
        )
        # Cross-entropy: "how surprised is the model by the correct token?"
        loss = F.cross_entropy(
            logits.reshape(-1, 256),       # predicted: [batch×len, 256]
            batch["tgt_ids"][:, 1:].reshape(-1),  # actual: [batch×len]
            ignore_index=PAD_ID,  # don't penalize padding tokens
        )
    
    elif model_type == "tree":
        predicted_symbols = model(batch["input_trees"], batch["int_var"])
        # Same cross-entropy, but per-node instead of per-position
        loss = node_cross_entropy(predicted_symbols, batch["target_symbols"])
    
    loss.backward()  # compute gradients
    
    # Gradient clipping: prevent any single update from being too large
    # Without this, one bad batch can destroy the model
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()  # update model parameters
    return loss.item()
```

### Optimizer + Learning Rate Schedule
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # learning rate: how big each step is
    weight_decay=0.01,  # slightly shrink all params each step (prevents overfitting)
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=60,      # total epochs
    eta_min=1e-6,  # minimum learning rate at the end
)
