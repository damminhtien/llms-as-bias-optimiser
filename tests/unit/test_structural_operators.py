"""Tests for first-order structural feature operators."""

import numpy as np
import pytest

from bias_optimizer.features import (
    SpatialOperator,
    SymmetryOperator,
    TopologyOperator,
    skeletonize_image,
)


def test_skeletonization_returns_thin_binary_image() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[8:20, 12:16] = 1

    skeleton = skeletonize_image(image)

    assert skeleton.shape == (28, 28)
    assert skeleton.dtype == np.bool_
    assert skeleton.any()
    assert skeleton.sum() < np.count_nonzero(image)


def test_skeletonization_validates_threshold_and_image() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    with pytest.raises(ValueError, match="threshold"):
        skeletonize_image(image, threshold=1.1)
    with pytest.raises(ValueError, match="28x28"):
        skeletonize_image(np.zeros((10, 10), dtype=np.float32))


def test_topology_counts_components_holes_endpoints_and_junctions() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[5:12, 5:12] = 1
    image[17:24, 17:24] = 1
    image[7:10, 7:10] = 0

    features = TopologyOperator().transform(image)

    assert features.shape == (4,)
    assert features.dtype == np.float32
    assert features.tolist()[:2] == [2.0, 1.0]
    assert np.isfinite(features).all()


def test_topology_counts_skeleton_endpoints_and_junctions() -> None:
    t_shape = np.zeros((28, 28), dtype=np.float32)
    t_shape[7:21, 13:15] = 1
    t_shape[7:9, 7:21] = 1

    features = TopologyOperator().transform(t_shape)

    assert features[2] == 3
    assert features[3] == 1


def test_spatial_operator_returns_top_middle_bottom_ink_fractions() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[0, :2] = 1
    image[13, 0] = 1
    image[27, 0] = 1

    features = SpatialOperator().transform(image)

    np.testing.assert_allclose(features, [0.5, 0.25, 0.25])
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(
        SpatialOperator().transform(np.zeros_like(image)), np.zeros(3)
    )


def test_symmetry_scores_identical_and_reflected_images() -> None:
    symmetric = np.zeros((28, 28), dtype=np.float32)
    symmetric[5:23, 10:18] = 1
    asymmetric = symmetric.copy()
    asymmetric[10:18, 10:14] = 0

    scores = SymmetryOperator().transform(symmetric)
    asymmetric_scores = SymmetryOperator().transform(asymmetric)

    np.testing.assert_allclose(scores, [1, 1])
    assert asymmetric_scores[0] < scores[0]
    assert asymmetric_scores[1] == scores[1]
    np.testing.assert_array_equal(
        SymmetryOperator().transform(np.zeros_like(symmetric)), np.ones(2)
    )


@pytest.mark.parametrize(
    "operator", [TopologyOperator(), SpatialOperator(), SymmetryOperator()]
)
def test_structural_operators_reject_invalid_images(operator: object) -> None:
    with pytest.raises(ValueError, match="28x28"):
        operator.transform(np.zeros((12, 12), dtype=np.float32))  # type: ignore[attr-defined]
