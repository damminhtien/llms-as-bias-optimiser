"""Explicit allow-list and parameter validation for feature operators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

from bias_optimizer.domain.bias import OperatorSpec
from bias_optimizer.features.base import FeatureOperator
from bias_optimizer.features.baselines import RawPixelsOperator
from bias_optimizer.features.curvature import CurvatureOperator
from bias_optimizer.features.spatial import SpatialOperator
from bias_optimizer.features.stroke import (
    DirectionTransitionOperator,
    StrokeDirectionOperator,
)
from bias_optimizer.features.symmetry import SymmetryOperator
from bias_optimizer.features.topology import TopologyOperator

_TOPOLOGY_THRESHOLD_BOUNDS = (0.0, 1.0)


def _no_parameters(_: Mapping[str, Any]) -> None:
    return None


def _topology_parameters(params: Mapping[str, Any]) -> None:
    threshold = params.get("threshold", 0.5)
    if type(threshold) not in (int, float):
        raise TypeError("topology threshold must be a number")
    if not isfinite(threshold) or not (
        _TOPOLOGY_THRESHOLD_BOUNDS[0] < threshold < _TOPOLOGY_THRESHOLD_BOUNDS[1]
    ):
        raise ValueError("topology threshold must be strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class _OperatorDefinition:
    factory: Callable[[Mapping[str, Any]], FeatureOperator]
    allowed_parameters: frozenset[str] = frozenset()
    validate_parameters: Callable[[Mapping[str, Any]], None] = _no_parameters


def _create_raw_pixels(_: Mapping[str, Any]) -> FeatureOperator:
    return RawPixelsOperator()


def _create_topology(params: Mapping[str, Any]) -> FeatureOperator:
    return TopologyOperator(threshold=float(params.get("threshold", 0.5)))


def _create_spatial(_: Mapping[str, Any]) -> FeatureOperator:
    return SpatialOperator()


def _create_symmetry(_: Mapping[str, Any]) -> FeatureOperator:
    return SymmetryOperator()


def _create_stroke_direction(_: Mapping[str, Any]) -> FeatureOperator:
    return StrokeDirectionOperator()


def _create_curvature(_: Mapping[str, Any]) -> FeatureOperator:
    return CurvatureOperator()


def _create_direction_transition(_: Mapping[str, Any]) -> FeatureOperator:
    return DirectionTransitionOperator()


OPERATOR_NAMES = (
    "raw_pixels",
    "topology",
    "spatial",
    "symmetry",
    "stroke_direction",
    "curvature",
    "direction_transition",
)

_OPERATOR_DEFINITIONS = MappingProxyType(
    {
        "raw_pixels": _OperatorDefinition(_create_raw_pixels),
        "topology": _OperatorDefinition(
            _create_topology,
            allowed_parameters=frozenset({"threshold"}),
            validate_parameters=_topology_parameters,
        ),
        "spatial": _OperatorDefinition(_create_spatial),
        "symmetry": _OperatorDefinition(_create_symmetry),
        "stroke_direction": _OperatorDefinition(_create_stroke_direction),
        "curvature": _OperatorDefinition(_create_curvature),
        "direction_transition": _OperatorDefinition(_create_direction_transition),
    }
)


class OperatorRegistry:
    """Build only the feature operators approved for candidate specifications."""

    @property
    def names(self) -> tuple[str, ...]:
        return OPERATOR_NAMES

    def build(self, spec: OperatorSpec) -> FeatureOperator:
        if not isinstance(spec, OperatorSpec):
            raise TypeError("operator registry requires an OperatorSpec")
        definition = _OPERATOR_DEFINITIONS.get(spec.name)
        if definition is None:
            allowed = ", ".join(OPERATOR_NAMES)
            raise ValueError(
                f"unknown operator {spec.name!r}; allowed operators: {allowed}"
            )
        extra = set(spec.params) - definition.allowed_parameters
        if extra:
            raise ValueError(
                f"operator {spec.name!r} does not accept parameters: {sorted(extra)}"
            )
        definition.validate_parameters(spec.params)
        return definition.factory(spec.params)
