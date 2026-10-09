"""Orientation and transition histograms from skeleton graph paths."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from bias_optimizer.features._stroke_math import (
    DIRECTION_BINS,
    is_closed_path,
    path_angles_for_paths,
    quantize_direction,
)
from bias_optimizer.features.base import FeatureVector, Image
from bias_optimizer.features.skeleton import skeletonize_image
from bias_optimizer.features.skeleton_graph import (
    extract_graph_paths,
    skeleton_to_graph,
)


def _image_paths(image: Image) -> tuple[tuple[tuple[int, int], ...], ...]:
    skeleton = skeletonize_image(image)
    graph = skeleton_to_graph(skeleton)
    return extract_graph_paths(graph)


@dataclass(frozen=True, slots=True)
class StrokeDirectionOperator:
    """Return an orientation-neutral histogram over eight compass directions."""

    @property
    def feature_dim(self) -> int:
        return DIRECTION_BINS

    def transform(self, image: Image) -> FeatureVector:
        histogram = np.zeros(DIRECTION_BINS, dtype=np.float64)
        for angles in path_angles_for_paths(_image_paths(image)):
            bins = quantize_direction(angles)
            for direction in bins:
                histogram[direction] += 1
                histogram[(direction + DIRECTION_BINS // 2) % DIRECTION_BINS] += 1
        total = float(histogram.sum())
        if total:
            histogram /= total
        return np.asarray(histogram, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DirectionTransitionOperator:
    """Return a normalized 8x8 histogram of consecutive path directions."""

    @property
    def feature_dim(self) -> int:
        return DIRECTION_BINS * DIRECTION_BINS

    def transform(self, image: Image) -> FeatureVector:
        transitions = np.zeros((DIRECTION_BINS, DIRECTION_BINS), dtype=np.float64)
        paths = _image_paths(image)
        for path, angles in zip(paths, path_angles_for_paths(paths), strict=True):
            if angles.size < 2:
                continue
            bins = quantize_direction(angles)
            pairs = list(pairwise(bins))
            if is_closed_path(path):
                pairs.append((bins[-1], bins[0]))
            for first, second in pairs:
                transitions[first, second] += 1
                reverse_first = (second + DIRECTION_BINS // 2) % DIRECTION_BINS
                reverse_second = (first + DIRECTION_BINS // 2) % DIRECTION_BINS
                transitions[reverse_first, reverse_second] += 1
        total = float(transitions.sum())
        if total:
            transitions /= total
        return np.asarray(transitions.reshape(-1), dtype=np.float32)
