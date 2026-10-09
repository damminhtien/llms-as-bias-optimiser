"""Typed relational DSL contracts and location-aware sequence behavior."""

from collections import Counter
from itertools import pairwise

import numpy as np
import pytest
from skimage.draw import line
from skimage.morphology import skeletonize

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import (
    ProgramCompilationError,
    ProgramCompiler,
    _perturb_centroid_values,
    _rewire_topology,
    _topology_signature,
)
from bias_optimizer.dsl.primitives import SequenceValue
from bias_optimizer.dsl.seeds import (
    generate_seed_programs,
    generate_v3_pilot_seed_programs,
)
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.novelty.descriptors import MECHANISM_FAMILIES, mechanism_family


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _paths() -> Expr:
    return _node("paths", _node("graph", _node("skeletonize", _node("image"))))


def _angle_turns() -> Expr:
    return _node("delta_angle", _node("angles", _paths()))


def _polyline_image(points: tuple[tuple[int, int], ...]) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.float32)
    for start, end in pairwise(points):
        rows, cols = line(*start, *end)
        image[rows, cols] = 1.0
    return image


def _translated_elbow(top: int) -> np.ndarray:
    return _polyline_image(((top, 5), (top, 12), (top + 7, 12)))


def _t_image(row: int) -> np.ndarray:
    image = np.zeros((28, 28), dtype=np.float32)
    image[row, 6:19] = 1.0
    image[row : row + 9, 12] = 1.0
    return image


def _spatial_turn_program() -> Expr:
    return _node(
        "spatial_condition",
        _angle_turns(),
        axis="vertical",
        regions=3,
        bins=8,
        low=-3.141593,
        high=3.141593,
    )


def test_relational_primitives_have_static_dimensions_and_types() -> None:
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    conditional = compiler.compile(_spatial_turn_program())
    paired = compiler.compile(
        _node(
            "cross_histogram",
            _node("path_summary", _paths(), measure="branch_endpoints"),
            _node("path_summary", _paths(), measure="centroid_y"),
            bins_x=3,
            bins_y=4,
            low_x=0,
            high_x=2,
            low_y=0,
            high_y=1,
        )
    )
    difference = compiler.compile(
        _node(
            "pairwise_difference",
            _node("spatial_split", _node("image"), rows=2, cols=3),
            _node("spatial_split", _node("image"), rows=3, cols=2),
        )
    )

    assert conditional.feature_dim == 24
    assert paired.feature_dim == 12
    assert difference.feature_dim == 6

    with pytest.raises(ProgramCompilationError, match="same dimension"):
        compiler.compile(
            _node(
                "pairwise_difference",
                _node("spatial_split", _node("image"), rows=2, cols=3),
                _node("spatial_split", _node("image"), rows=2, cols=2),
            )
        )


def test_spatial_condition_encodes_curvature_by_height() -> None:
    pipeline = ProgramCompiler().compile(_spatial_turn_program())

    top = pipeline.transform(_translated_elbow(3))
    bottom = pipeline.transform(_translated_elbow(17))

    assert top.shape == (24,)
    assert bottom.shape == (24,)
    assert np.isfinite(top).all() and np.isfinite(bottom).all()
    assert not np.array_equal(top, bottom)


def test_cross_histogram_captures_loop_position_and_branch_location() -> None:
    loop_position = ProgramCompiler().compile(
        _node(
            "cross_histogram",
            _node("path_summary", _paths(), measure="is_loop"),
            _node("path_summary", _paths(), measure="centroid_y"),
            bins_x=2,
            bins_y=4,
            low_x=0,
            high_x=1,
            low_y=0,
            high_y=1,
        )
    )
    branch_position = ProgramCompiler().compile(
        _node(
            "cross_histogram",
            _node("path_summary", _paths(), measure="branch_endpoints"),
            _node("path_summary", _paths(), measure="centroid_y"),
            bins_x=3,
            bins_y=4,
            low_x=0,
            high_x=2,
            low_y=0,
            high_y=1,
        )
    )
    top_loop = _polyline_image(((4, 4), (4, 10), (10, 10), (10, 4), (4, 4)))
    bottom_loop = _polyline_image(((16, 4), (16, 10), (22, 10), (22, 4), (16, 4)))
    top_branch = _t_image(4)
    bottom_branch = _t_image(15)

    assert not np.array_equal(
        loop_position.transform(top_loop), loop_position.transform(bottom_loop)
    )
    assert not np.array_equal(
        branch_position.transform(top_branch), branch_position.transform(bottom_branch)
    )


def test_centroid_counterfactual_breaks_loop_position_and_preserves_path_count() -> (
    None
):
    paths = _paths()
    program = _node(
        "cross_histogram",
        _node("path_summary", paths, measure="is_loop"),
        _node("path_summary", paths, measure="centroid_y"),
        bins_x=2,
        bins_y=4,
        low_x=0,
        high_x=1,
        low_y=0,
        high_y=1,
    )
    image = _polyline_image(((2, 3), (2, 10), (9, 10), (9, 3), (2, 3)))
    image[15:25, 20] = 1.0
    pipeline = ProgramCompiler().compile(program)

    original = pipeline.transform(image)
    counterfactual = pipeline.transform_with_centroid_perturbation(
        image, target_measure="centroid_y", seed=3
    )

    assert original.shape == counterfactual.shape == (8,)
    assert not np.array_equal(original, counterfactual)


def test_cross_histogram_aligns_different_sequence_lengths_by_location() -> None:
    angles = _node("angles", _paths())
    turns = _node("delta_angle", angles)
    program = _node(
        "cross_histogram",
        angles,
        turns,
        bins_x=6,
        bins_y=6,
        low_x=-3.141593,
        high_x=3.141593,
        low_y=-3.141593,
        high_y=3.141593,
    )
    pipeline = ProgramCompiler().compile(program)

    features = pipeline.transform(_t_image(8))

    assert features.shape == (36,)
    assert np.isfinite(features).all()
    assert features.sum() == pytest.approx(1.0)


def test_turn_autocorrelation_changes_when_sequence_order_is_shuffled() -> None:
    program = _node("autocorrelation", _angle_turns(), lags=[1, 2, 3])
    pipeline = ProgramCompiler().compile(program)
    image = _polyline_image(
        ((8, 3), (8, 6), (5, 9), (5, 13), (9, 16), (14, 16), (18, 12), (18, 8))
    )

    original = pipeline.transform(image)
    shuffled = pipeline.transform_with_sequence_shuffle(
        image, target_op="delta_angle", seed=17
    )

    assert original.shape == (3,)
    assert np.isfinite(original).all() and np.isfinite(shuffled).all()
    assert not np.array_equal(original, shuffled)


def test_spatial_reassignment_preserves_value_multiset_but_changes_conditioning() -> (
    None
):
    program = _spatial_turn_program()
    pipeline = ProgramCompiler().compile(program)
    image = _polyline_image(
        ((3, 3), (3, 15), (8, 15), (8, 5), (13, 5), (13, 18), (19, 18))
    )
    marginal = ProgramCompiler().compile(
        _node("histogram", _node("angles", _paths()), bins=8)
    )

    original = pipeline.transform(image)
    reassigned = pipeline.transform_with_spatial_reassignment(
        image, target_op="angles", seed=19
    )

    assert original.shape == reassigned.shape == (24,)
    np.testing.assert_array_equal(
        marginal.transform(image),
        marginal.transform_with_spatial_reassignment(
            image, target_op="angles", seed=19
        ),
    )
    assert not np.array_equal(original, reassigned)


def test_cross_pair_shuffle_preserves_marginals_but_changes_joint_histogram() -> None:
    angles = _node("angles", _paths())
    turns = _node("delta_angle", angles)
    pipeline = ProgramCompiler().compile(
        _node(
            "cross_histogram",
            angles,
            turns,
            bins_x=6,
            bins_y=6,
            low_x=-3.141593,
            high_x=3.141593,
            low_y=-3.141593,
            high_y=3.141593,
        )
    )
    image = _polyline_image(
        ((8, 3), (8, 6), (5, 9), (5, 13), (9, 16), (14, 16), (18, 16), (18, 8))
    )

    original = pipeline.transform(image)
    shuffled = pipeline.transform_with_cross_pair_shuffle(
        image, target_op="cross_histogram", seed=23
    )

    assert original.shape == shuffled.shape == (36,)
    assert original.sum() == pytest.approx(1.0)
    assert shuffled.sum() == pytest.approx(1.0)
    assert not np.array_equal(original, shuffled)


def test_centroid_perturbation_preserves_marginal_and_path_locations() -> None:
    locations = np.asarray([[0.2, 0.3], [0.7, 0.8]])
    summary = SequenceValue((np.asarray([0.2, 0.7]),), (locations,))

    perturbed = _perturb_centroid_values(summary, np.random.default_rng(3))

    np.testing.assert_array_equal(
        np.sort(summary.groups[0]), np.sort(perturbed.groups[0])
    )
    np.testing.assert_array_equal(summary.locations[0], perturbed.locations[0])
    assert not np.array_equal(summary.groups[0], perturbed.groups[0])


def test_topology_rewire_preserves_foreground_count_and_changes_cycle_rank() -> None:
    loop = _polyline_image(((4, 4), (4, 20), (20, 20), (20, 4), (4, 4)))
    skeleton_image = skeletonize(loop.astype(bool))
    rewired_mask = _rewire_topology(skeleton_image, np.random.default_rng(29))
    assert np.count_nonzero(skeleton_image) == np.count_nonzero(rewired_mask)
    assert _topology_signature(skeleton_image) != _topology_signature(rewired_mask)

    program = _node(
        "cycle_rank",
        _node("graph", _node("skeletonize", _node("image"))),
    )
    pipeline = ProgramCompiler().compile(program)

    original = pipeline.transform(loop)
    rewired = pipeline.transform_with_topology_rewire(
        loop, target_op="skeletonize", seed=29
    )

    assert original[0] != rewired[0]


def test_order_sensitive_seed_programs_are_valid_and_shuffle_sensitive() -> None:
    programs = generate_seed_programs(3)
    turn_image = _polyline_image(
        ((8, 3), (8, 6), (5, 9), (5, 13), (9, 16), (14, 16), (18, 12), (18, 8))
    )
    graph_image = _t_image(8)

    for index, (program, target_op, image) in enumerate(
        (
            (programs[0].program, "delta_angle", turn_image),
            (programs[1].program, "degree_sequence", graph_image),
            (programs[2].program, "degree_sequence", graph_image),
        )
    ):
        pipeline = ProgramCompiler().compile(program)
        original = pipeline.transform(image)
        shuffled = pipeline.transform_with_sequence_shuffle(
            image, target_op=target_op, seed=31 + index
        )
        assert not np.array_equal(original, shuffled)


def test_v3_pilot_uses_ten_explicit_mechanism_balanced_seeds() -> None:
    seeds = generate_v3_pilot_seed_programs()

    assert len(seeds) == 10
    assert len({bias.program.to_json() for bias in seeds}) == 10
    assert Counter(mechanism_family(bias.program) for bias in seeds) == Counter(
        {
            "order_sensitive": 3,
            "spatial_relational": 2,
            "graph_relational": 2,
            "free_exploration": 3,
        }
    )
    assert set(MECHANISM_FAMILIES) == set(
        Counter(mechanism_family(bias.program) for bias in seeds)
    )
