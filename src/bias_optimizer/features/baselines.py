"""Deterministic pixel and gradient feature operators for MNIST baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog

from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image



@dataclass(frozen=True, slots=True)
class RawPixelsOperator:
    """Use all normalized grayscale pixels as the feature vector."""

    def transform(self, image: Image) -> FeatureVector:
        return validated_image(image).reshape(-1).copy()


@dataclass(frozen=True, slots=True)
class DownsampledPixelsOperator:
    """Average each 2x2 pixel block to produce a 14x14 representation."""

    def transform(self, image: Image) -> FeatureVector:
        array = validated_image(image)
        downsampled = array.reshape(14, 2, 14, 2).mean(axis=(1, 3))
        return np.asarray(downsampled.reshape(-1), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class HOGOperator:
    """Histogram of oriented gradients with fixed, reproducible settings."""

    orientations: int = 9
    pixels_per_cell: tuple[int, int] = (4, 4)
    cells_per_block: tuple[int, int] = (2, 2)

    def __post_init__(self) -> None:
        if self.orientations <= 0:
            raise ValueError("orientations must be positive")
        if any(size <= 0 for size in (*self.pixels_per_cell, *self.cells_per_block)):
            raise ValueError("cell and block dimensions must be positive")

    def transform(self, image: Image) -> FeatureVector:
        array = validated_image(image)
        features = hog(
            array,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True,
            channel_axis=None,
        )
        return np.asarray(features, dtype=np.float32)
