# P4-F002: FlashAttention-2 Integration

**Status**: draft
**Priority**: P0 (zero-risk, pure speedup)

## Problem

Seq transformer (95M, 10 layers, 10 heads, d_model=640) uses standard PyTorch attention. FlashAttention-2 provides 2-3x kernel-level speedup with identical mathematical output. No architectural changes required.

## Solution

1. Add `flash-attn` package to `pyproject.toml`
2. Replace `F.scaled_dot_product_attention` calls in `seq_transformer.py` with FlashAttention-2 kernel
3. Use `torch.nn.functional.scaled_dot_product_attention` with `enable_flash=True` backend (PyTorch 2.x native path) as fallback for non-CUDA hardware
4. Verify numerical equivalence via property test (max absolute diff < 1e-5)

## Acceptance Criteria

- [ ] FlashAttention-2 active during training on CUDA hardware
- [ ] Graceful fallback to standard SDPA on CPU/MPS
- [ ] Training throughput improvement ≥1.5x measured on batch_size=256, seq_len=512
- [ ] Numerical equivalence test passes (max diff < 1e-5 vs standard attention)
- [ ] No change to model outputs or validation loss
- [ ] Memory reduction measured and logged

## Affected Files

- `pyproject.toml` — add `flash-attn` optional dependency
- `python/neurips/models/seq_transformer.py` — attention replacement
- `tests/python/test_models.py` — add numerical equivalence test

## Risks

- Minimal. FlashAttention-2 is standard practice. Only risk: CUDA version compatibility.
