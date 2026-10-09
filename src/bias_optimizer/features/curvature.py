"""Curvature histograms from skeleton paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bias_optimizer.features._stroke_math import (
    DIRECTION_BINS,
    is_closed_path,
    path_angles_for_paths,
    wrap_angle,
)
from bias_optimizer.features.base import FeatureVector, Image
from bias_optimizer.features.stroke import _image_paths


@dataclass(frozen=True, slots=True)
class CurvatureOperator:
    """Return an orientation-neutral histogram of signed local turning angles."""

    def transform(self, image: Image) -> FeatureVector:
        histogram = np.zeros(DIRECTION_BINS, dtype=np.float64)
        paths = _image_paths(image)
        for path, angles in zip(paths, path_angles_for_paths(paths), strict=True):
            if angles.size < 2:
                continue
            turns = wrap_angle(np.diff(angles))
            if is_closed_path(path):
                turns = np.append(turns, wrap_angle(angles[0] - angles[-1]))
            bins = (
                np.floor((turns + np.pi) * DIRECTION_BINS / (2 * np.pi)).astype(
                    np.int64
                )
                % DIRECTION_BINS
            )
            reverse_bins = (
                np.floor((-turns + np.pi) * DIRECTION_BINS / (2 * np.pi)).astype(
                    np.int64
                )
                % DIRECTION_BINS
            )
            np.add.at(histogram, bins, 1)
            np.add.at(histogram, reverse_bins, 1)
        total = float(histogram.sum())
        if total:
            histogram /= total
        return np.asarray(histogram, dtype=np.float32)
