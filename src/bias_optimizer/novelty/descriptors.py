"""Deterministic structural descriptors for quality-diversity archiving."""

from __future__ import annotations

from dataclasses import dataclass

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.primitives import PRIMITIVES

IMPLEMENTED_NICHES = (
    "topology",
    "graph_structure",
    "geometry",
    "stroke_dynamics",
    "spatial_relations",
    "frequency_scale",
    "raw_pixels",
    "compositional",
    "hybrid",
)


@dataclass(frozen=True, slots=True)
class ProgramDescriptor:
    primary_niche: str
    complexity_bin: str
    feature_families: tuple[str, ...]
    primitives: tuple[str, ...]
    depth: int
    node_count: int


def expression_nodes(expr: Expr) -> tuple[Expr, ...]:
    """Return a stable preorder traversal of an expression tree."""
    return (expr, *(node for child in expr.args for node in expression_nodes(child)))


def describe_program(expr: Expr) -> ProgramDescriptor:
    nodes = expression_nodes(expr)
    families = sorted(
        {
            PRIMITIVES[node.op].family
            for node in nodes
            if PRIMITIVES[node.op].family is not None
            and PRIMITIVES[node.op].family != "compositional"
        }
    )
    if len(families) > 1:
        niche = "hybrid"
    elif families:
        niche = families[0]
    else:
        niche = "compositional"
    node_count = len(nodes)
    if node_count <= 5:
        complexity = "compact"
    elif node_count <= 10:
        complexity = "moderate"
    else:
        complexity = "deep"

    def operation_depth(node: Expr) -> int:
        if node.op == "image":
            return 0
        return 1 + max((operation_depth(child) for child in node.args), default=0)

    depth = operation_depth(expr)
    return ProgramDescriptor(
        primary_niche=niche,
        complexity_bin=complexity,
        feature_families=tuple(families),
        primitives=tuple(sorted({node.op for node in nodes})),
        depth=depth,
        node_count=node_count,
    )
