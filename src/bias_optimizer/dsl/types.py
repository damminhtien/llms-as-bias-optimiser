"""Types used by the safe representation-program DSL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ValueType(StrEnum):
    IMAGE = "Image"
    BINARY_IMAGE = "BinaryImage"
    SKELETON = "Skeleton"
    GRAPH = "Graph"
    PATH_SET = "PathSet"
    SEQUENCE = "Sequence"
    ANGLE_SEQUENCE = "AngleSequence"
    VECTOR = "Vector"
    SCALAR = "Scalar"


@dataclass(frozen=True, slots=True)
class TypeInfo:
    """Static type and resource metadata inferred for one expression node."""

    value_type: ValueType
    dimension: int | None
    depth: int
    node_count: int
    operations: frozenset[str]
