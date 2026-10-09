"""Coarse vertical occupancy features for MNIST images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image


@dataclass(frozen=True, slots=True)
class SpatialOperator:
    """Return ink-mass fractions in top, middle, and bottom image bands."""

    def transform(self, image: Image) -> FeatureVector:
        array = validated_image(image)
        total_mass = float(array.sum())
        if total_mass == 0:
            return np.zeros(3, dtype=np.float32)
        band_masses = np.asarray(
            [array[:9].sum(), array[9:19].sum(), array[19:].sum()],
            dtype=np.float64,
        )
        return np.asarray(band_masses / total_mass, dtype=np.float32)
