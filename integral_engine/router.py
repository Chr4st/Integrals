"""Method-routing classifier for integration techniques (Spec 9).

Classifies integrands by the most appropriate integration technique,
enabling the solver to route expressions to specialized sub-solvers.

Architecture:
  - IntegrationRouter: lightweight MLP classifier over depth-encoded features
  - Heuristic classify_integrand for generating training labels
  - Four technique classes: polynomial/rational, trigonometric,
    exponential/logarithmic, special function
"""
from __future__ import annotations

from typing import Dict

import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from integral_engine.vocabulary import FEATURE_DIM

# Technique class mapping: index -> human-readable name
TECHNIQUE_CLASSES: Dict[int, str] = {
    0: "polynomial_rational",
    1: "trigonometric",
    2: "exponential_logarithmic",
    3: "special_function",
}

# SymPy function types for heuristic classification
_TRIG_FUNCS = (
    sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
    sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc,
    sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch,
    sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch,
)

_EXP_LOG_FUNCS = (sp.exp, sp.log)

_SPECIAL_FUNCS = (
    sp.erf, sp.erfc, sp.erfi,
    sp.gamma, sp.loggamma, sp.polygamma,
    sp.besselj, sp.bessely, sp.besseli, sp.besselk,
    sp.Ei,
)


class IntegrationRouter(nn.Module):
    """Lightweight MLP classifier for integration technique routing.

    Maps depth-encoded feature vectors to technique class probabilities.
    Default feature_dim=608 matches the FEATURE_DIM from vocabulary.
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = 128,
        n_classes: int = 4,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute technique class probabilities.

        Args:
            features: (batch, feature_dim) depth-encoded feature vectors.

        Returns:
            (batch, n_classes) softmax probability distribution.
        """
        logits = self.net(features)
        return F.softmax(logits, dim=-1)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices.

        Args:
            features: (batch, feature_dim) depth-encoded feature vectors.

        Returns:
            (batch,) int tensor of class indices.
        """
        probs = self.forward(features)
        return probs.argmax(dim=-1)


def _contains_type(expr: sp.Basic, func_types: tuple) -> bool:
    """Check if expression tree contains any function of the given types."""
    if isinstance(expr, func_types):
        return True
    return any(_contains_type(arg, func_types) for arg in expr.args)


def classify_integrand(expr: sp.Basic, var: sp.Symbol) -> int:
    """Heuristic classifier for generating integration technique labels.

    Checks expression structure to assign a technique class:
      1 (trigonometric) if any trig/hyperbolic function is present
      2 (exponential_logarithmic) if exp or log is present
      3 (special_function) if special functions (erf, Bessel, etc.) are present
      0 (polynomial_rational) otherwise

    Priority: trig > exp/log > special > polynomial/rational.
    This ordering reflects that trig integrands often need specialized
    substitutions even when other function types are also present.

    Args:
        expr: SymPy expression (the integrand).
        var: integration variable (unused in current heuristic but
             available for future refinements).

    Returns:
        Integer class index (0-3).
    """
    if _contains_type(expr, _TRIG_FUNCS):
        return 1
    if _contains_type(expr, _EXP_LOG_FUNCS):
        return 2
    if _contains_type(expr, _SPECIAL_FUNCS):
        return 3
    return 0
