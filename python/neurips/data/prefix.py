"""Shared prefix-notation parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from neurips.data.vocab import ARITY


@dataclass(frozen=True)
class PrefixNode:
    """A node in a parsed prefix expression tree."""

    token: str
    depth: int
    children: tuple[int, ...] = ()  # indices into flat node list


def parse_prefix(tokens: Sequence[str]) -> list[PrefixNode]:
    """Parse a prefix token sequence into a flat list of PrefixNode.

    Uses a stack to track remaining argument slots at each depth.
    Returns nodes in the order they appear in the token sequence.
    """
    if not tokens:
        return []

    nodes: list[PrefixNode] = []
    # Stack of (remaining_args, depth, parent_index)
    stack: list[list[int]] = []  # [remaining_args, depth, node_index]
    child_collector: dict[int, list[int]] = {}

    for tok in tokens:
        depth = len(stack)
        node_idx = len(nodes)
        arity = ARITY.get(tok, 0)

        nodes.append(PrefixNode(token=tok, depth=depth))

        # Register this node as a child of the parent
        if stack:
            parent_idx = stack[-1][2]
            child_collector.setdefault(parent_idx, []).append(node_idx)
            stack[-1][0] -= 1

        # If this node expects children, push onto stack
        if arity > 0:
            stack.append([arity, depth, node_idx])

        # Pop completed frames
        while stack and stack[-1][0] == 0:
            finished = stack.pop()
            fidx = finished[2]
            kids = tuple(child_collector.get(fidx, ()))
            nodes[fidx] = PrefixNode(
                token=nodes[fidx].token,
                depth=nodes[fidx].depth,
                children=kids,
            )

    return nodes


def tokenize_prefix(prefix_str: str) -> list[str]:
    """Split a prefix string into tokens."""
    return prefix_str.strip().split()


def tree_depth(nodes: list[PrefixNode]) -> int:
    """Return maximum depth in the parsed tree."""
    if not nodes:
        return 0
    return max(n.depth for n in nodes)


# ---------------------------------------------------------------------------
# Python ↔ Rust prefix format translation
# ---------------------------------------------------------------------------

_VARS: frozenset[str] = frozenset(["x", "y", "z", "w", "t"])
_PARAMS: frozenset[str] = frozenset(["a", "b", "c", "d", "n", "m", "k"])
_CONSTS: frozenset[str] = frozenset(["pi", "euler_gamma", "catalan"])

# Python vocab name → Rust parser name (only where they differ)
_FUNC_PY_TO_RUST: dict[str, str] = {
    "Ei": "ei",
    "Si": "si",
    "Ci": "ci",
    "Li": "li",
    "Gamma": "gamma",
    "BesselJ": "besselJ",
    "BesselY": "besselY",
    "FresnelS": "fresnelS",
    "FresnelC": "fresnelC",
    "EllipticK": "ellipticK",
    "EllipticE": "ellipticE",
    "Hyp2F1": "hyp2f1",
}

# Reverse: Rust name → Python vocab name
_FUNC_RUST_TO_PY: dict[str, str] = {v: k for k, v in _FUNC_PY_TO_RUST.items()}


def python_prefix_to_rust(prefix_str: str) -> str:
    """Translate a Python-format prefix string to Rust format.

    Python uses bare tokens:  ``add x 3``
    Rust expects prefixed:    ``add var:x 3``
    """
    tokens = prefix_str.strip().split()
    out: list[str] = []
    for tok in tokens:
        if tok in _VARS:
            out.append(f"var:{tok}")
        elif tok in _PARAMS:
            out.append(f"param:{tok}")
        elif tok in _CONSTS:
            out.append(f"const:{tok}")
        elif tok in _FUNC_PY_TO_RUST:
            out.append(_FUNC_PY_TO_RUST[tok])
        else:
            out.append(tok)
    return " ".join(out)


def rust_prefix_to_python(prefix_str: str) -> str:
    """Translate a Rust-format prefix string to Python format.

    Rust uses prefixed:     ``add var:x 3``
    Python uses bare:       ``add x 3``
    """
    tokens = prefix_str.strip().split()
    out: list[str] = []
    for tok in tokens:
        if tok.startswith("var:"):
            out.append(tok[4:])
        elif tok.startswith("param:"):
            out.append(tok[6:])
        elif tok.startswith("const:"):
            out.append(tok[6:])
        elif tok.startswith("rat:"):
            # rat:1/2 → keep as-is (numbers in Python are just integers)
            out.append(tok)
        elif tok in _FUNC_RUST_TO_PY:
            out.append(_FUNC_RUST_TO_PY[tok])
        else:
            out.append(tok)
    return " ".join(out)
