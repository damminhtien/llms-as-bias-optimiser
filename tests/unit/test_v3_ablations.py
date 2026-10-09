"""Mechanism ablations remove only their named program structure."""

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from experiments.evaluate_v3_ablations import ablate_program


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _paths() -> Expr:
    return _node("paths", _node("graph", _node("skeletonize", _node("image"))))


def test_no_spatial_ablation_replaces_conditioning_with_global_summary() -> None:
    program = _node(
        "spatial_condition",
        _node("angles", _paths()),
        regions=3,
        bins=6,
    )

    ablated, count = ablate_program(program, "no_spatial")

    assert count == 1
    assert ablated.op == "histogram"
    assert ProgramCompiler().compile(ablated).feature_dim == 6


def test_no_order_ablation_preserves_orderless_moments() -> None:
    program = _node(
        "autocorrelation",
        _node("delta_angle", _node("angles", _paths())),
        lags=[1, 2, 4],
    )

    ablated, count = ablate_program(program, "no_order")

    assert count == 2
    assert ablated.op == "moments"
    assert ablated.args[0].op == "angles"
    assert ProgramCompiler().compile(ablated).feature_dim == 3


def test_no_relation_ablation_keeps_marginals_as_separate_histograms() -> None:
    program = _node(
        "cross_histogram",
        _node("path_summary", _paths(), measure="is_loop"),
        _node("path_summary", _paths(), measure="centroid_y"),
        bins_x=3,
        bins_y=4,
    )

    ablated, count = ablate_program(program, "no_relation")

    assert count == 1
    assert ablated.op == "concat"
    assert [child.op for child in ablated.args] == ["histogram", "histogram"]
    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.DISCOVERY,
            max_feature_dim=128,
            max_depth=8,
        )
    )
    assert compiler.compile(ablated).feature_dim == 7


def test_no_relation_ablation_keeps_pairwise_inputs_but_removes_their_difference() -> (
    None
):
    program = _node(
        "pairwise_difference",
        _node(
            "spatial_condition",
            _node("angles", _paths()),
            regions=1,
            bins=3,
        ),
        _node(
            "spatial_condition",
            _node("path_summary", _paths(), measure="centroid_x"),
            regions=1,
            bins=3,
            high=64.0,
        ),
    )

    ablated, count = ablate_program(program, "no_relation")

    assert count == 3
    assert ablated.op == "concat"
    assert [child.op for child in ablated.args] == ["histogram", "histogram"]
    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.DISCOVERY,
            max_feature_dim=128,
            max_depth=8,
        )
    )
    assert compiler.compile(program).feature_dim == 3
    assert compiler.compile(ablated).feature_dim == 6


def test_no_curvature_ablation_uses_orientation_without_turn_derivative() -> None:
    program = _node(
        "autocorrelation",
        _node("delta_angle", _node("angles", _paths())),
        lags=[1, 2],
    )

    ablated, count = ablate_program(program, "no_curvature")

    assert count == 1
    assert ablated.args[0].op == "angles"
    assert ProgramCompiler().compile(ablated).feature_dim == 2
