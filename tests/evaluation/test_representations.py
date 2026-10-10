from __future__ import annotations

import numpy as np

from bias_optimizer.evaluation.representations import (
    HOGPCARepresentation,
    build_concatenated_representation,
    build_frozen_registry,
)
from bias_optimizer.features.baselines import HOGOperator


def test_frozen_representation_registry_has_protocol_dimensions() -> None:
    registry = build_frozen_registry()
    assert {name: definition.dimension for name, definition in registry.items()} == {
        "angle_centroid_pairwise": 30,
        "regional_turn_histogram": 60,
        "horizontal_turn_spatial": 30,
        "regional_turn_profile": 36,
        "vertical_turn_gradient": 42,
        "cycle_angle_hist": 16,
        "zoning_30d": 30,
        "raw_pixels": 784,
        "hog": 1296,
        "hog_pca_30d": 30,
        "hog_pca_60d": 60,
    }


def test_pca_fits_only_the_supplied_training_images() -> None:
    rng = np.random.default_rng(29)
    training = rng.random((40, 28, 28), dtype=np.float32)
    evaluation = np.clip(1.0 - rng.random((8, 28, 28), dtype=np.float32), 0, 1)
    representation = HOGPCARepresentation("hog_pca_30d", 30)

    representation.fit(training, np.arange(len(training), dtype=np.int64))
    expected_train_mean = HOGOperator().transform(training[0])
    assert representation._pca is not None
    train_features = np.stack([HOGOperator().transform(image) for image in training])
    all_features = np.concatenate(
        [train_features, np.stack([HOGOperator().transform(image) for image in evaluation])]
    )
    np.testing.assert_allclose(representation._pca.mean_, train_features.mean(axis=0), atol=1e-6)
    assert not np.allclose(representation._pca.mean_, all_features.mean(axis=0), atol=1e-3)
    state = representation.fitted_state_id

    assert representation.transform(training).shape == (40, 30)
    assert representation.transform(evaluation).shape == (8, 30)
    assert representation.fitted_state_id == state
    assert expected_train_mean.shape == (1296,)


def test_concatenation_keeps_rows_aligned() -> None:
    registry = build_frozen_registry()
    definition = build_concatenated_representation(
        registry, "hog", "angle_centroid_pairwise"
    )
    representation = definition.create()
    images = np.zeros((3, 28, 28), dtype=np.float32)
    representation.fit(images)
    features = representation.transform(images)
    assert definition.dimension == 1326
    assert features.shape == (3, 1326)
