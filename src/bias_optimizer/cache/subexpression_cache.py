"""Bounded in-memory reuse of expensive typed-DLS subtrees across candidates."""

from __future__ import annotations

import sys
from collections import OrderedDict
from typing import Any

import networkx as nx
import numpy as np

_CACHEABLE_OPERATIONS = frozenset(
    {
        "skeletonize",
        "graph",
        "paths",
        "edge_lengths",
        "angles",
        "delta",
        "delta_angle",
    }
)


def _estimate_bytes(value: Any) -> int:
    """Conservatively estimate memory retained by one immutable DSL value."""
    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if isinstance(item, nx.Graph):
            return (
                sys.getsizeof(item)
                + 192 * item.number_of_nodes()
                + 112 * item.number_of_edges()
            )
        groups = getattr(item, "groups", None)
        if isinstance(groups, tuple):
            return sys.getsizeof(item) + sum(visit(group) for group in groups)
        if isinstance(item, (tuple, list)):
            return sys.getsizeof(item) + sum(visit(child) for child in item)
        return sys.getsizeof(item)

    return max(1, visit(value))


class SubexpressionCache:
    """LRU cache scoped by exact dataset row and canonical subtree JSON."""

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self._values: OrderedDict[tuple[str, str], tuple[Any, int]] = OrderedDict()
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def metrics(self) -> dict[str, int | float]:
        """Return a snapshot suitable for experiment reports."""
        requests = self._hits + self._misses
        return {
            "entries": len(self._values),
            "bytes": self._current_bytes,
            "max_bytes": self.max_bytes,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / requests if requests else 0.0,
        }

    def get(
        self, sample_key: str, expression_key: str, *, operation: str
    ) -> Any | None:
        if operation not in _CACHEABLE_OPERATIONS:
            return None
        key = (sample_key, expression_key)
        result = self._values.get(key)
        if result is None:
            self._misses += 1
            return None
        self._values.move_to_end(key)
        self._hits += 1
        return result[0]

    def set(
        self,
        sample_key: str,
        expression_key: str,
        value: Any,
        *,
        operation: str,
    ) -> None:
        if operation not in _CACHEABLE_OPERATIONS:
            return
        size = _estimate_bytes(value)
        if size > self.max_bytes:
            return
        key = (sample_key, expression_key)
        previous = self._values.pop(key, None)
        if previous is not None:
            self._current_bytes -= previous[1]
        self._values[key] = (value, size)
        self._current_bytes += size
        while self._current_bytes > self.max_bytes:
            _, (_, removed_size) = self._values.popitem(last=False)
            self._current_bytes -= removed_size
            self._evictions += 1
