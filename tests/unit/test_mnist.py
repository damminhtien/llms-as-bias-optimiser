from __future__ import annotations

import numpy as np
import pytest

from bias_optimizer.data.mnist import (
    MNISTFinalData,
    MNISTSearchData,
    split_search_data,
)


def _synthetic_digits() -> tuple[np.ndarray, np.ndarray]:
    labels = np.tile(np.arange(10, dtype=np.int64), 20)
    images = np.zeros((len(labels), 28, 28), dtype=np.float32)
    images[:, 0, 0] = np.arange(len(labels), dtype=np.float32)
    return images, labels


def test_search_split_is_reproducible_and_stratified() -> None:
    images, labels = _synthetic_digits()

    first = split_search_data(images, labels, validation_size=40, seed=23)
    second = split_search_data(images, labels, validation_size=40, seed=23)

    assert isinstance(first, MNISTSearchData)
    np.testing.assert_array_equal(first.train_images, second.train_images)
    np.testing.assert_array_equal(first.validation_images, second.validation_images)
    np.testing.assert_array_equal(first.train_labels, second.train_labels)
    np.testing.assert_array_equal(first.validation_labels, second.validation_labels)
    np.testing.assert_array_equal(np.bincount(first.validation_labels), [4] * 10)
    assert not hasattr(first, "test_images")
    assert not first.train_images.flags.writeable


def test_training_subset_is_reproducible_and_has_requested_size() -> None:
    images, labels = _synthetic_digits()
    search_data = split_search_data(images, labels, validation_size=20, seed=7)

    first_images, first_labels = search_data.sample_training_data(50, seed=11)
    second_images, second_labels = search_data.sample_training_data(50, seed=11)

    assert first_images.shape == (50, 28, 28)
    np.testing.assert_array_equal(first_images, second_images)
    np.testing.assert_array_equal(first_labels, second_labels)


def test_final_sampler_returns_full_training_partition_without_copying() -> None:
    images, labels = _synthetic_digits()
    final_data = MNISTFinalData(
        train_images=images,
        train_labels=labels,
        test_images=images[:20],
        test_labels=labels[:20],
    )

    train_images, train_labels = final_data.sample_training_data(len(labels))

    assert train_images is final_data.train_images
    assert train_labels is final_data.train_labels


def test_final_sampler_indices_match_sampled_training_data() -> None:
    images, labels = _synthetic_digits()
    final_data = MNISTFinalData(
        train_images=images,
        train_labels=labels,
        test_images=images[:20],
        test_labels=labels[:20],
    )

    indices = final_data.sample_training_indices(50, seed=23)
    sampled_images, sampled_labels = final_data.sample_training_data(50, seed=23)

    np.testing.assert_array_equal(sampled_images, final_data.train_images[indices])
    np.testing.assert_array_equal(sampled_labels, final_data.train_labels[indices])
    np.testing.assert_array_equal(
        final_data.sample_training_indices(50, seed=23), indices
    )


@pytest.mark.parametrize("validation_size", [0, 200])
def test_search_split_rejects_invalid_validation_size(validation_size: int) -> None:
    images, labels = _synthetic_digits()

    with pytest.raises(ValueError):
        split_search_data(
            images,
            labels,
            validation_size=validation_size,
        )
