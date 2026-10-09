from __future__ import annotations

import numpy as np
import pytest

from bias_optimizer.features.baselines import (
    DownsampledPixelsOperator,
    HOGOperator,
    RawPixelsOperator,
)


def test_raw_pixels_preserve_all_normalized_pixels() -> None:
    image = np.linspace(0.0, 1.0, 28 * 28, dtype=np.float32).reshape(28, 28)

    features = RawPixelsOperator().transform(image)

    assert features.shape == (784,)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(features, image.ravel())


def test_downsampled_pixels_average_2x2_blocks() -> None:
    image = np.arange(784, dtype=np.float32).reshape(28, 28) / 783

    features = DownsampledPixelsOperator().transform(image)

    expected = image.reshape(14, 2, 14, 2).mean(axis=(1, 3)).ravel()
    assert features.shape == (196,)
    np.testing.assert_allclose(features, expected)


def test_hog_output_is_fixed_size_and_finite() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[5:23, 12:16] = 1.0

    features = HOGOperator().transform(image)

    assert features.shape == (1_296,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan])
def test_raw_pixels_reject_invalid_images(value: float) -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[0, 0] = value

    with pytest.raises(ValueError):
        RawPixelsOperator().transform(image)
