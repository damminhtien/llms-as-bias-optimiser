"""Validation shared by deterministic MNIST feature operators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bias_optimizer.features.base import Image

IMAGE_SHAPE = (28, 28)


def validated_image(image: Image) -> NDArray[np.float32]:
    """Return a validated float32 MNIST image."""
    array = np.asarray(image, dtype=np.float32)
    if array.shape != IMAGE_SHAPE:
        raise ValueError("MNIST operators expect 28x28 images")
    if not np.isfinite(array).all():
        raise ValueError("image must contain only finite values")
    if np.any((array < 0) | (array > 1)):
        raise ValueError("normalized image pixels must be in [0, 1]")
    return array
