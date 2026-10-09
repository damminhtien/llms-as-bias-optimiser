"""Behavioral descriptors for quality-diversity program search."""

from __future__ import annotations

from dataclasses import dataclass

from bias_optimizer.dsl.ast import Expr

DescriptorCell = tuple[str, str, str, str, str]

SOURCES = ("topology", "graph", "path_geometry", "pixel", "mixed")
ORDERS = ("orderless", "first_order", "higher_order")
SPATIAL_MODES = ("global", "localized")
COMPOSITIONS = ("single", "composite")
COMPLEXITIES = ("compact", "moderate", "deep")
MECHANISM_FAMILIES = (
    "order_sensitive",
    "spatial_relational",
    "graph_relational",
    "free_exploration",
)
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

# Behavioral targets shown to the proposer. The full archive remains open to
# every valid five-axis cell; this list only gives the search a stable compass.
DESCRIPTOR_TARGET_CELLS: tuple[DescriptorCell, ...] = (
    ("topology", "orderless", "global", "single", "compact"),
    ("graph", "orderless", "global", "single", "moderate"),
    ("graph", "higher_order", "global", "single", "deep"),
    ("path_geometry", "orderless", "global", "single", "moderate"),
    ("path_geometry", "first_order", "global", "single", "moderate"),
    ("path_geometry", "higher_order", "global", "single", "moderate"),
    ("path_geometry", "higher_order", "global", "single", "deep"),
    ("path_geometry", "higher_order", "localized", "single", "deep"),
    ("path_geometry", "orderless", "localized", "composite", "deep"),
    ("pixel", "orderless", "localized", "single", "compact"),
    ("mixed", "orderless", "global", "composite", "moderate"),
    ("mixed", "higher_order", "localized", "composite", "deep"),
)


@dataclass(frozen=True, slots=True)
class ProgramDescriptor:
    """Describe the behavior represented by a program, not its plumbing."""

    source: str
    order: str
    spatial: str
    composition: str
    complexity: str
    primitives: tuple[str, ...]
    depth: int
    node_count: int

    def __post_init__(self) -> None:
        for name, allowed in (
            ("source", SOURCES),
            ("order", ORDERS),
            ("spatial", SPATIAL_MODES),
            ("composition", COMPOSITIONS),
            ("complexity", COMPLEXITIES),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"invalid {name} descriptor: {getattr(self, name)!r}")
        object.__setattr__(self, "primitives", tuple(sorted(set(self.primitives))))
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if type(self.node_count) is not int or self.node_count <= 0:
            raise ValueError("node_count must be a positive integer")

    @property
    def cell(self) -> DescriptorCell:
        return (
            self.source,
            self.order,
            self.spatial,
            self.composition,
            self.complexity,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "order": self.order,
            "spatial": self.spatial,
            "composition": self.composition,
            "complexity": self.complexity,
        }


def expression_nodes(expr: Expr) -> tuple[Expr, ...]:
    """Return a stable preorder traversal of an expression tree."""
    return (expr, *(node for child in expr.args for node in expression_nodes(child)))


def _source(nodes: tuple[Expr, ...]) -> str:
    operations = {node.op for node in nodes}
    sources: set[str] = set()
    if operations.intersection({"flatten_pixels", "spatial_split"}):
        sources.add("pixel")
    if operations.intersection({"connected_components", "cycle_rank"}):
        sources.add("topology")
    if "degree_sequence" in operations:
        sources.add("graph")
    if operations.intersection({"angles", "edge_lengths", "delta_angle"}):
        sources.add("path_geometry")
    if "path_summary" in operations:
        summary_measures = {
            node.params.get("measure", "length")
            for node in nodes
            if node.op == "path_summary"
        }
        if summary_measures.intersection(
            {"length", "centroid_x", "centroid_y", "aspect_ratio"}
        ):
            sources.add("path_geometry")
        if summary_measures.intersection({"is_loop", "branch_endpoints"}):
            sources.add("graph")
    if "count" in operations and not sources:
        sources.add("graph")
    if len(sources) > 1:
        return "mixed"
    if sources:
        return next(iter(sources))
    return "graph"


def _operation_depth(expr: Expr) -> int:
    if expr.op == "image":
        return 0
    return 1 + max((_operation_depth(child) for child in expr.args), default=0)


def describe_program(expr: Expr) -> ProgramDescriptor:
    """Describe source, order, spatial scope, composition, and cost of an AST."""
    nodes = expression_nodes(expr)
    operations = {node.op for node in nodes}
    node_count = len(nodes)
    if node_count <= 5:
        complexity = "compact"
    elif node_count <= 10:
        complexity = "moderate"
    else:
        complexity = "deep"

    if operations.intersection({"autocorrelation", "run_length_encode"}):
        order = "higher_order"
    elif operations.intersection({"delta", "delta_angle"}):
        order = "first_order"
    else:
        order = "orderless"

    localized = bool(operations.intersection({"spatial_split", "spatial_condition"}))
    localized |= any(
        node.op == "path_summary"
        and node.params.get("measure", "length") in {"centroid_x", "centroid_y"}
        for node in nodes
    )

    composite = bool(
        operations.intersection(
            {"concat", "ratio", "pairwise_difference", "cross_histogram"}
        )
    )
    return ProgramDescriptor(
        source=_source(nodes),
        order=order,
        spatial="localized" if localized else "global",
        composition="composite" if composite else "single",
        complexity=complexity,
        primitives=tuple(sorted(operations)),
        depth=_operation_depth(expr),
        node_count=node_count,
    )


def mechanism_family(expr: Expr) -> str:
    """Classify the dominant structural mechanism for balanced pilot reports."""
    nodes = expression_nodes(expr)
    operations = {node.op for node in nodes}
    summaries = {
        node.params.get("measure", "length")
        for node in nodes
        if node.op == "path_summary"
    }
    if "cross_histogram" in operations and summaries.intersection(
        {"is_loop", "branch_endpoints"}
    ):
        return "graph_relational"
    if "spatial_condition" in operations or summaries.intersection(
        {"centroid_x", "centroid_y"}
    ):
        return "spatial_relational"
    if operations.intersection(
        {"delta", "delta_angle", "run_length_encode", "autocorrelation"}
    ):
        return "order_sensitive"
    return "free_exploration"
