"""Tests for the fixed feature-operator allow-list."""

import pytest

from bias_optimizer.domain.bias import OperatorSpec
from bias_optimizer.features.baselines import RawPixelsOperator
from bias_optimizer.features.curvature import CurvatureOperator
from bias_optimizer.features.registry import OPERATOR_NAMES, OperatorRegistry
from bias_optimizer.features.spatial import SpatialOperator
from bias_optimizer.features.stroke import (
    DirectionTransitionOperator,
    StrokeDirectionOperator,
)
from bias_optimizer.features.symmetry import SymmetryOperator
from bias_optimizer.features.topology import TopologyOperator


def test_registry_contains_only_the_approved_operators() -> None:
    registry = OperatorRegistry()

    assert registry.names == (
        "raw_pixels",
        "topology",
        "spatial",
        "symmetry",
        "stroke_direction",
        "curvature",
        "direction_transition",
    )
    assert OPERATOR_NAMES == registry.names


@pytest.mark.parametrize(
    ("name", "operator_type"),
    [
        ("raw_pixels", RawPixelsOperator),
        ("topology", TopologyOperator),
        ("spatial", SpatialOperator),
        ("symmetry", SymmetryOperator),
        ("stroke_direction", StrokeDirectionOperator),
        ("curvature", CurvatureOperator),
        ("direction_transition", DirectionTransitionOperator),
    ],
)
def test_registry_builds_each_approved_operator(name: str, operator_type: type) -> None:
    operator = OperatorRegistry().build(OperatorSpec(name))

    assert isinstance(operator, operator_type)


def test_topology_accepts_a_bounded_threshold() -> None:
    operator = OperatorRegistry().build(OperatorSpec("topology", {"threshold": 0.4}))

    assert isinstance(operator, TopologyOperator)
    assert operator.threshold == 0.4


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.1, True])
def test_topology_rejects_out_of_bounds_or_non_numeric_threshold(
    threshold: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="threshold"):
        OperatorRegistry().build(OperatorSpec("topology", {"threshold": threshold}))


def test_registry_rejects_unknown_operators_and_untrusted_code_names() -> None:
    with pytest.raises(ValueError, match="unknown operator 'python_exec'"):
        OperatorRegistry().build(
            OperatorSpec("python_exec", {"source": "raise SystemExit()"})
        )
    with pytest.raises(ValueError, match="unknown operator 'hog'"):
        OperatorRegistry().build(OperatorSpec("hog"))


def test_registry_rejects_unapproved_parameters() -> None:
    with pytest.raises(ValueError, match="does not accept parameters"):
        OperatorRegistry().build(OperatorSpec("symmetry", {"threshold": 0.5}))
    with pytest.raises(ValueError, match="threshold2"):
        OperatorRegistry().build(OperatorSpec("topology", {"threshold2": 0.5}))


def test_registry_requires_an_operator_spec() -> None:
    with pytest.raises(TypeError, match="OperatorSpec"):
        OperatorRegistry().build("topology")  # type: ignore[arg-type]
