"""Utilities for turning a normalized MNIST image into a thin binary stroke."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from bias_optimizer.features.base import Image
from bias_optimizer.features._image import validated_image


def skeletonize_image(image: Image, threshold: float = 0.5) -> NDArray[np.bool_]:
    """Threshold and skeletonize an MNIST image deterministically."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    binary = validated_image(image) >= threshold
    return np.asarray(skeletonize(binary), dtype=np.bool_)
