"""Tests for graph paths and handwriting-dynamics representations."""

import networkx as nx
import numpy as np
import pytest

from bias_optimizer.features import (
    CurvatureOperator,
    DirectionTransitionOperator,
    StrokeDirectionOperator,
    detect_endpoints_and_junctions,
    extract_graph_paths,
    skeleton_to_graph,
)


def test_skeleton_graph_omits_redundant_diagonal_shortcut() -> None:
    skeleton = np.zeros((8, 8), dtype=np.bool_)
    skeleton[2, 2:4] = True
    skeleton[3, 3] = True

    graph = skeleton_to_graph(skeleton)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert not graph.has_edge((2, 2), (3, 3))


def test_graph_detects_endpoints_junctions_and_extracts_branch_paths() -> None:
    skeleton = np.zeros((9, 9), dtype=np.bool_)
    skeleton[2:7, 4] = True
    skeleton[2, 2:7] = True
    graph = skeleton_to_graph(skeleton)

    endpoints, junctions = detect_endpoints_and_junctions(graph)
    paths = extract_graph_paths(graph)

    assert endpoints == ((2, 2), (2, 6), (6, 4))
    assert junctions == ((2, 4),)
    assert len(paths) == 3
    assert all(len(path) >= 2 for path in paths)
    assert all((2, 4) in (path[0], path[-1]) for path in paths)


def test_graph_paths_preserve_closed_loops() -> None:
    skeleton = np.zeros((8, 8), dtype=np.bool_)
    skeleton[2, 2:6] = True
    skeleton[5, 2:6] = True
    skeleton[2:6, 2] = True
    skeleton[2:6, 5] = True

    paths = extract_graph_paths(skeleton_to_graph(skeleton))

    assert len(paths) == 1
    assert paths[0][0] == paths[0][-1]
    assert len(paths[0]) == 13


def test_skeleton_graph_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        skeleton_to_graph(np.zeros((2, 3, 4), dtype=np.bool_))


def test_stroke_direction_is_normalized_and_reversal_invariant() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[14, 6:22] = 1

    features = StrokeDirectionOperator().transform(image)

    assert features.shape == (8,)
    assert features.dtype == np.float32
    assert features[0] == features[4] == 0.5
    assert features.sum() == pytest.approx(1)
    assert np.isfinite(features).all()


def test_curvature_and_direction_transitions_capture_an_elbow() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[14, 5:15] = 1
    image[14:23, 14] = 1

    curvature = CurvatureOperator().transform(image)
    transitions = DirectionTransitionOperator().transform(image).reshape(8, 8)

    assert curvature.shape == (8,)
    assert curvature[3] == curvature[5]
    assert curvature[3] > 0
    assert curvature.sum() == pytest.approx(1)
    assert transitions.shape == (8, 8)
    assert transitions[0, 1] == transitions[5, 4]
    assert transitions[1, 2] == transitions[6, 5]
    assert transitions[0, 1] > 0 and transitions[1, 2] > 0
    assert transitions.sum() == pytest.approx(1)
    assert np.isfinite(curvature).all() and np.isfinite(transitions).all()


@pytest.mark.parametrize(
    ("operator", "dimension"),
    [
        (StrokeDirectionOperator(), 8),
        (CurvatureOperator(), 8),
        (DirectionTransitionOperator(), 64),
    ],
)
def test_dynamics_operators_return_zeros_for_blank_images(
    operator: object, dimension: int
) -> None:
    features = operator.transform(np.zeros((28, 28), dtype=np.float32))  # type: ignore[attr-defined]

    np.testing.assert_array_equal(features, np.zeros(dimension, dtype=np.float32))
    assert np.isfinite(features).all()


def test_path_extraction_accepts_plain_networkx_graph() -> None:
    graph = nx.path_graph([(0, 0), (0, 1), (0, 2)])

    assert extract_graph_paths(graph) == (((0, 0), (0, 1), (0, 2)),)
