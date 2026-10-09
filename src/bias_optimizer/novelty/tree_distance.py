"""Structural distance and novelty metrics for expression programs."""

from __future__ import annotations

from functools import lru_cache

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.novelty.descriptors import expression_nodes


@lru_cache(maxsize=32_768)
def _distance(left: Expr, right: Expr) -> int:
    root_cost = int(left.op != right.op) + int(left.params != right.params)
    rows = len(left.args) + 1
    cols = len(right.args) + 1
    costs = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = costs[i - 1][0] + len(expression_nodes(left.args[i - 1]))
    for j in range(1, cols):
        costs[0][j] = costs[0][j - 1] + len(expression_nodes(right.args[j - 1]))
    for i in range(1, rows):
        for j in range(1, cols):
            delete = costs[i - 1][j] + len(expression_nodes(left.args[i - 1]))
            insert = costs[i][j - 1] + len(expression_nodes(right.args[j - 1]))
            replace = costs[i - 1][j - 1] + _distance(
                left.args[i - 1], right.args[j - 1]
            )
            costs[i][j] = min(delete, insert, replace)
    return root_cost + costs[-1][-1]


def tree_distance(left: Expr, right: Expr) -> int:
    """Return ordered tree edit cost with subtree insert/delete costs."""
    if not isinstance(left, Expr) or not isinstance(right, Expr):
        raise TypeError("tree_distance requires Expr values")
    return _distance(left, right)


def structural_novelty(program: Expr, archive: tuple[Expr, ...]) -> float:
    """Return the normalized distance to the closest structurally distinct AST."""
    if not isinstance(program, Expr) or not all(
        isinstance(item, Expr) for item in archive
    ):
        raise TypeError("structural_novelty requires Expr values")
    distinct = tuple(item for item in archive if item != program)
    if not distinct:
        return 1.0
    node_count = len(expression_nodes(program))
    distance = min(tree_distance(program, other) for other in distinct)
    denominator = max(
        node_count,
        max(len(expression_nodes(other)) for other in distinct),
        1,
    )
    return min(1.0, distance / denominator)
