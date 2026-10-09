"""Tests for the fixed human-designed seed candidates."""

from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.features.registry import OperatorRegistry


def test_initial_biases_match_the_five_planned_operator_compositions() -> None:
    biases = initial_human_biases()

    assert len(biases) == 5
    assert [bias.name for bias in biases] == [
        "raw_pixels_control",
        "topology_only",
        "topology_plus_spatial",
        "topology_plus_curvature",
        "topology_direction_curvature",
    ]
    assert [tuple(operator.name for operator in bias.operators) for bias in biases] == [
        ("raw_pixels",),
        ("topology",),
        ("topology", "spatial"),
        ("topology", "curvature"),
        ("topology", "stroke_direction", "curvature"),
    ]


def test_every_human_seed_compiles_without_an_llm() -> None:
    registry = OperatorRegistry()

    for bias in initial_human_biases():
        assert all(registry.build(operator) for operator in bias.operators)
