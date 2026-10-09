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

    @property
    def feature_dim(self) -> int:
        return 28 * 28

    def transform(self, image: Image) -> FeatureVector:
        return validated_image(image).reshape(-1).copy()


@dataclass(frozen=True, slots=True)
class DownsampledPixelsOperator:
    """Average each 2x2 pixel block to produce a 14x14 representation."""

    @property
    def feature_dim(self) -> int:
        return 14 * 14

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

    @property
    def feature_dim(self) -> int:
        cell_rows = 28 // self.pixels_per_cell[0]
        cell_cols = 28 // self.pixels_per_cell[1]
        block_rows = cell_rows - self.cells_per_block[0] + 1
        block_cols = cell_cols - self.cells_per_block[1] + 1
        if block_rows <= 0 or block_cols <= 0:
            raise ValueError("HOG cells_per_block exceed the image cell dimensions")
        return (
            block_rows
            * block_cols
            * self.cells_per_block[0]
            * self.cells_per_block[1]
            * self.orientations
        )

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
