"""LRU verification cache for MCTS per-step CAS checks.

Caches (state, action) → (next_state, is_terminal, is_solved) results
to avoid redundant SymPy calls during tree search.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable


class VerificationCache:
    """LRU cache for environment step results."""

    def __init__(self, max_size: int = 100_000) -> None:
        self._cache: OrderedDict[tuple[str, int], tuple[str, bool, bool]] = OrderedDict()
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self,
        state: str,
        action: int,
        compute_fn: Callable[[str, int], tuple[str, bool, bool]],
    ) -> tuple[str, bool, bool]:
        """Look up cached result or compute and cache."""
        key = (state, action)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        result = compute_fn(state, action)
        self._cache[key] = result

        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0
