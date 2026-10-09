"""Reflection symmetry features for normalized MNIST images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image


@dataclass(frozen=True, slots=True)
class SymmetryOperator:
    """Return left-right and top-bottom reflection similarity scores."""

    @property
    def feature_dim(self) -> int:
        return 2

    def transform(self, image: Image) -> FeatureVector:
        array = validated_image(image)
        mass = float(array.sum())
        if mass == 0:
            return np.ones(2, dtype=np.float32)
        left_right = 1 - np.abs(array - array[:, ::-1]).sum() / (2 * mass)
        top_bottom = 1 - np.abs(array - array[::-1, :]).sum() / (2 * mass)
        return np.asarray([left_right, top_bottom], dtype=np.float32)
