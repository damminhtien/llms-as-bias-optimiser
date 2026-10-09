"""Finalist interventions preserve nuisance marginals by mechanism."""

from bias_optimizer.dsl.ast import Expr
from experiments.evaluate_v3_counterfactuals import _intervention_plan
from experiments.freeze_v3_finalists import _counterfactual_expectation


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _paths() -> Expr:
    return _node("paths", _node("graph", _node("skeletonize", _node("image"))))


def test_order_mechanism_targets_an_ordered_turn_sequence() -> None:
    program = _node(
        "autocorrelation",
        _node("delta_angle", _node("angles", _paths())),
    )

    plan = _intervention_plan(program, "ordered stroke or turn persistence")

    assert plan["kind"] == "sequence_order"
    assert plan["target_op"] == "delta_angle"
    assert "multisets" in plan["preserved"]


def test_spatial_mechanism_targets_events_before_conditioning() -> None:
    program = _node(
        "spatial_condition",
        _node("angles", _paths()),
        axis="vertical",
        regions=3,
    )

    plan = _intervention_plan(program, "spatial conditioning")

    assert plan["kind"] == "spatial_reassignment"
    assert plan["target_op"] == "angles"
    assert "event-to-location" in plan["destroyed"]


def test_cross_relational_mechanism_targets_joint_pairings() -> None:
    program = _node(
        "cross_histogram",
        _node("path_summary", _paths(), measure="is_loop"),
        _node("path_summary", _paths(), measure="centroid_y"),
    )

    plan = _intervention_plan(program, "graph and location interaction")

    assert plan["kind"] == "centroid_perturbation"
    assert plan["target_op"] == "path_summary"
    assert plan["target_measure"] == "centroid_y"
    assert "path count" in plan["preserved"]


def test_sequence_cross_histogram_is_a_cross_variable_relation() -> None:
    angles = _node("angles", _paths())
    program = _node("cross_histogram", angles, _node("delta_angle", angles))

    expectation = _counterfactual_expectation(program)
    plan = _intervention_plan(program, expectation["mechanism"])

    assert expectation["mechanism"] == "cross-variable relation"
    assert plan["kind"] == "cross_pair_shuffle"


def test_global_distribution_mechanism_uses_feature_identity_permutation() -> None:
    program = _node("histogram", _node("angles", _paths()), bins=8)

    plan = _intervention_plan(program, "global path geometry")

    assert plan["kind"] == "feature_identity_permutation"
    assert "exact marginal distribution" in plan["preserved"]


def test_derivative_histogram_is_not_mislabeled_as_order_persistence() -> None:
    program = _node(
        "histogram",
        _node("delta_angle", _node("angles", _paths())),
        bins=8,
    )

    expectation = _counterfactual_expectation(program)

    assert expectation["mechanism"] == "global path geometry"


def test_autocorrelation_freezes_an_order_specific_intervention() -> None:
    program = _node(
        "autocorrelation",
        _node("delta_angle", _node("angles", _paths())),
    )

    expectation = _counterfactual_expectation(program)
    plan = _intervention_plan(program, expectation["mechanism"])

    assert expectation["mechanism"] == "ordered stroke or turn persistence"
    assert plan["kind"] == "sequence_order"
    assert plan["target_op"] == "delta_angle"
