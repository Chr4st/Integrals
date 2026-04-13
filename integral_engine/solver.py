"""Public API for the integral prediction engine.

Two-stage pipeline:
  Stage 1: SymPy direct integration (4s timeout via ProcessPoolExecutor)
  Stage 2: ML model inference + constant solving (15s timeout via ThreadPoolExecutor)

Usage:
    from integral_engine.solver import solve_integral
    result = solve_integral("sin(x)")
"""
from __future__ import annotations

import atexit
import os
import threading
import uuid
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from typing import Any, Dict, List, Optional, Tuple

import structlog
import sympy as sp

from integral_engine.vocabulary import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    SEQ_TOKENS,
    feature_schema_hash,
)

logger = structlog.get_logger(__name__)

# Lazy-loaded model singleton
_model = None
_model_device = None

SYMPY_TIMEOUT = 4.0
ML_TIMEOUT = 15.0

# Phase 10: Persistent process pool singleton for SymPy stage
_sympy_pool: Optional[ProcessPoolExecutor] = None
_sympy_pool_lock = threading.Lock()


def _get_sympy_pool() -> ProcessPoolExecutor:
    """Return a module-level singleton ProcessPoolExecutor for SymPy."""
    global _sympy_pool
    if _sympy_pool is not None:
        return _sympy_pool
    with _sympy_pool_lock:
        if _sympy_pool is None:
            _sympy_pool = ProcessPoolExecutor(max_workers=1)
            atexit.register(_shutdown_sympy_pool)
    return _sympy_pool


def _shutdown_sympy_pool() -> None:
    global _sympy_pool
    if _sympy_pool is not None:
        _sympy_pool.shutdown(wait=False)
        _sympy_pool = None


def _sympy_integrate(expr_str: str, var_str: str) -> Optional[str]:
    """Integrate with SymPy in a subprocess (for timeout safety)."""
    x = sp.Symbol(var_str)
    expr = sp.sympify(expr_str)
    result = sp.integrate(expr, x)

    if result.has(sp.Integral):
        return None
    if result.has(sp.Piecewise):
        return None

    return str(result)


def _to_latex(result_str: str) -> Optional[str]:
    """Convert SymPy result string to LaTeX with + C."""
    result = sp.sympify(result_str)

    if result.has(sp.Integral):
        return None
    if result.has(sp.Piecewise):
        return None

    simplified = sp.simplify(result)
    return sp.latex(simplified) + " + C"


class CheckpointError(Exception):
    """Raised when a model checkpoint is missing, corrupted, or incompatible."""


def _validate_checkpoint(
    checkpoint_path: str, model: Any,
) -> dict:
    """Validate and load a checkpoint file.

    Checks: file exists, non-empty, deserializable, contains expected keys,
    state_dict keys match model, tensor shapes match.
    """
    import torch

    if not os.path.exists(checkpoint_path):
        raise CheckpointError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Set INTEGRAL_MODEL_PATH env var or run training first."
        )

    file_size = os.path.getsize(checkpoint_path)
    if file_size == 0:
        raise CheckpointError(f"Checkpoint is empty (0 bytes): {checkpoint_path}")

    logger.info(
        "loading_checkpoint", path=checkpoint_path, size_mb=round(file_size / 1e6, 2),
    )

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True,
        )
    except Exception as e:
        raise CheckpointError(f"Failed to deserialize checkpoint: {e}") from e

    if "model_state_dict" not in checkpoint:
        available = list(checkpoint.keys())
        raise CheckpointError(
            f"Checkpoint missing 'model_state_dict' key. Available keys: {available}"
        )

    ckpt_keys = set(checkpoint["model_state_dict"].keys())
    model_keys = set(model.state_dict().keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        raise CheckpointError(
            f"state_dict mismatch.\n"
            f"  Missing keys ({len(missing)}): {sorted(missing)[:5]}...\n"
            f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}..."
        )

    model_sd = model.state_dict()
    for key in model_keys:
        expected_shape = model_sd[key].shape
        actual_shape = checkpoint["model_state_dict"][key].shape
        if expected_shape != actual_shape:
            raise CheckpointError(
                f"Shape mismatch for '{key}': "
                f"expected {expected_shape}, got {actual_shape}"
            )

    # Feature schema validation (non-fatal warning if key absent for old checkpoints)
    ckpt_schema = checkpoint.get("feature_schema_hash")
    current_schema = feature_schema_hash()
    if ckpt_schema is not None and ckpt_schema != current_schema:
        raise CheckpointError(
            f"Feature schema mismatch: checkpoint={ckpt_schema}, "
            f"current={current_schema}. Feature heads or depth config changed "
            f"since this checkpoint was trained."
        )

    if "val_metrics" in checkpoint:
        logger.info(
            "checkpoint_metadata",
            epoch=checkpoint.get("epoch"),
            val_metrics=checkpoint.get("val_metrics"),
        )

    return checkpoint


def _load_model() -> Tuple[Any, Any]:
    """Lazily load the ML model. Returns (model, device)."""
    global _model, _model_device

    if _model is not None:
        return _model, _model_device

    import torch
    from integral_engine.model import IntegralTransformer

    # Phase 7: Prefer averaged checkpoint if available
    default_path = "checkpoints/best.pt"
    averaged_path = "checkpoints/averaged.pt"
    if os.path.exists(averaged_path):
        default_path = averaged_path
    checkpoint_path = os.environ.get("INTEGRAL_MODEL_PATH", default_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = IntegralTransformer()

    if os.path.exists(checkpoint_path):
        checkpoint = _validate_checkpoint(checkpoint_path, model)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    _model = model
    _model_device = device
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("model_loaded", device=str(device), parameters=param_count)
    return model, device


def _try_candidate(
    seq: List[int],
    log_prob: float,
    expr: sp.Basic,
    var: sp.Symbol,
) -> Optional[Tuple[str, float]]:
    """Try to verify a single candidate sequence. Returns (latex, confidence) or None."""
    import numpy as np

    from integral_engine.constant_solver import solve_constants
    from integral_engine.tokenizer import from_prefix

    tokens = []
    for idx in seq:
        if idx in (BOS_INDEX, EOS_INDEX, PAD_INDEX):
            continue
        if 0 <= idx < len(SEQ_TOKENS):
            tokens.append(SEQ_TOKENS[idx])

    if not tokens:
        return None

    template_expr = from_prefix(tokens)

    constants = sorted(
        [
            s for s in template_expr.free_symbols
            if s.name.startswith("C") and s.name[1:].isdigit()
        ],
        key=lambda s: s.name,
    )

    result = solve_constants(template_expr, expr, var, constants)

    if result is not None:
        latex = _to_latex(str(result))
        if latex is not None:
            confidence = float(np.exp(log_prob))
            return latex, confidence

    return None


def _ml_solve(expr_str: str, var_str: str) -> Optional[Tuple[str, float]]:
    """ML inference + constant solving. Returns (latex, confidence) or None.

    Phase 5: Primary strategy is temperature sampling with verification oracle.
    Generates N=25 diverse samples, verifies each with SymPy differentiation.
    Falls back to beam search (width 10) if no sample verifies.
    """
    import torch

    from integral_engine.feature_extractor import extract_features
    from integral_engine.tokenizer import encode

    var = sp.Symbol(var_str)
    expr = sp.sympify(expr_str)

    model, device = _load_model()

    src_indices = encode(expr)
    src = torch.tensor([src_indices], device=device)
    depth_feat = torch.tensor(
        [extract_features(expr, var).tolist()], dtype=torch.float32, device=device,
    )

    # Phase 5: Sampling + verification oracle (25 samples, batched in groups of 5)
    n_samples = 25
    batch_size = 5
    for batch_start in range(0, n_samples, batch_size):
        for _ in range(batch_size):
            try:
                seq, log_prob = model.sample(
                    src, depth_feat, max_len=256, temperature=0.7, top_p=0.95,
                )
                result = _try_candidate(seq, log_prob, expr, var)
                if result is not None:
                    logger.info(
                        "sample_verified",
                        sample_index=batch_start,
                    )
                    return result
            except Exception:
                logger.debug("sample_failed", exc_info=True)
                continue

    # Fallback: beam search (width 10)
    logger.info("sampling_exhausted_falling_back_to_beam_search")
    beams = model.generate(src, depth_feat, max_len=256, beam_width=10)

    for beam_idx, (seq, log_prob) in enumerate(beams):
        try:
            result = _try_candidate(seq, log_prob, expr, var)
            if result is not None:
                return result
        except Exception:
            logger.debug(
                "beam_candidate_failed",
                beam_index=beam_idx,
                token_count=len(seq),
                exc_info=True,
            )
            continue

    return None


def solve_integral(expr_str: str, var: str = "x") -> Dict[str, Any]:
    """Solve an integral using SymPy-first with ML fallback.

    Args:
        expr_str: Integrand as a SymPy-parseable infix string.
        var: Variable of integration (default "x").

    Returns:
        {
            "status": "solved_sympy" | "solved_ml" | "failed",
            "latex": str | None,
            "method": str,
            "confidence": float | None
        }

    Raises:
        ValueError: If expr_str cannot be parsed by SymPy.
    """
    solve_id = uuid.uuid4().hex[:8]
    log = logger.bind(solve_id=solve_id, expr=expr_str, var=var)

    try:
        sp.sympify(expr_str)
    except (sp.SympifyError, SyntaxError, TypeError) as e:
        raise ValueError(f"Cannot parse expression: {expr_str!r}") from e

    log.info("solve_start")

    # Stage 1: SymPy direct
    try:
        pool = _get_sympy_pool()
        future = pool.submit(_sympy_integrate, expr_str, var)
        result_str = future.result(timeout=SYMPY_TIMEOUT)

        if result_str is not None:
            latex = _to_latex(result_str)
            if latex is not None:
                result = {
                    "status": "solved_sympy",
                    "latex": latex,
                    "method": "SymPy direct integration",
                    "confidence": None,
                }
                log.info("solve_complete", status="solved_sympy")
                return result
    except FuturesTimeoutError:
        log.info("sympy_stage_timeout", timeout_seconds=SYMPY_TIMEOUT)
    except Exception:
        log.warning("sympy_stage_error", exc_info=True)

    # Stage 2: ML (ThreadPoolExecutor to keep model in-process)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_ml_solve, expr_str, var)
            ml_result = future.result(timeout=ML_TIMEOUT)

            if ml_result is not None:
                latex, confidence = ml_result
                result = {
                    "status": "solved_ml",
                    "latex": latex,
                    "method": "Transformer prediction with constant solving",
                    "confidence": confidence,
                }
                log.info("solve_complete", status="solved_ml")
                return result
    except FuturesTimeoutError:
        log.info("ml_stage_timeout", timeout_seconds=ML_TIMEOUT)
    except Exception:
        log.warning("ml_stage_error", exc_info=True)

    log.info("solve_complete", status="failed")
    return {
        "status": "failed",
        "latex": None,
        "method": "All methods exhausted",
        "confidence": None,
    }
