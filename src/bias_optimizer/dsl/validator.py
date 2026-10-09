"""Static type checking and search-track constraints for DSL programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.primitives import PRIMITIVES
from bias_optimizer.dsl.types import TypeInfo, ValueType


class SearchTrack(StrEnum):
    AUGMENTATION = "augmentation"
    DISCOVERY = "discovery"


@dataclass(frozen=True, slots=True)
class ProgramConstraints:
    """Hard limits applied before any candidate reaches the evaluator."""

    track: SearchTrack = SearchTrack.DISCOVERY
    max_feature_dim: int | None = None
    max_depth: int = 6
    max_nodes: int = 127
    forbid_raw_pixels: bool = False
    require_vector_root: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.track, SearchTrack):
            object.__setattr__(self, "track", SearchTrack(self.track))
        if self.max_feature_dim is None:
            default_dimension = 128 if self.track is SearchTrack.DISCOVERY else 1_024
            object.__setattr__(self, "max_feature_dim", default_dimension)
        for name in ("max_feature_dim", "max_depth", "max_nodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("forbid_raw_pixels", "require_vector_root"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")


def infer_type(expr: Expr) -> TypeInfo:
    """Infer a node type and static feature dimension, rejecting bad ASTs."""
    if not isinstance(expr, Expr):
        raise TypeError("infer_type requires an Expr")
    definition = PRIMITIVES.get(expr.op)
    if definition is None:
        raise ValueError(f"unknown DSL primitive {expr.op!r}")

    children = tuple(infer_type(arg) for arg in expr.args)
    definition.validate_inputs(tuple(child.value_type for child in children))
    params: Mapping[str, object] = expr.parameter_values
    definition.validate_params(params)
    dimension = definition.output_dimension(children, params)
    return TypeInfo(
        value_type=definition.output_type,
        dimension=dimension,
        depth=(
            0
            if expr.op == "image"
            else 1 + max((child.depth for child in children), default=0)
        ),
        node_count=1 + sum(child.node_count for child in children),
        operations=frozenset({expr.op}).union(
            *(child.operations for child in children)
        ),
    )


def validate_program(
    expr: Expr,
    constraints: ProgramConstraints | None = None,
) -> TypeInfo:
    """Type-check a vector/scalar root and enforce the selected search track."""
    policy = constraints if constraints is not None else ProgramConstraints()
    info = infer_type(expr)
    if info.value_type not in {ValueType.VECTOR, ValueType.SCALAR}:
        raise TypeError(
            f"program root must produce Vector or Scalar, got {info.value_type.value}"
        )
    if policy.require_vector_root and info.value_type is not ValueType.VECTOR:
        raise TypeError("program root must produce Vector on this search track")
    dimension = 1 if info.value_type is ValueType.SCALAR else info.dimension
    if dimension is None:
        raise ValueError("program feature dimension must be statically known")
    if dimension <= 0:
        raise ValueError("program feature dimension must be positive")
    if dimension > policy.max_feature_dim:
        raise ValueError(
            f"program has {dimension} features; track limit is {policy.max_feature_dim}"
        )
    if info.depth > policy.max_depth:
        raise ValueError(f"program depth {info.depth} exceeds limit {policy.max_depth}")
    if info.node_count > policy.max_nodes:
        raise ValueError(
            f"program has {info.node_count} nodes; limit is {policy.max_nodes}"
        )
    if (
        policy.track is SearchTrack.DISCOVERY or policy.forbid_raw_pixels
    ) and "flatten_pixels" in info.operations:
        raise ValueError("raw pixels are not allowed in structural programs")
    return info
