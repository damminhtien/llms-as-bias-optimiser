"""Topological summaries of binarized MNIST strokes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image
from bias_optimizer.features.skeleton import skeletonize_image
from bias_optimizer.features.skeleton_graph import (
    detect_endpoints_and_junctions,
    skeleton_to_graph,
)

_FOREGROUND_CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)
_BACKGROUND_CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask, structure=_FOREGROUND_CONNECTIVITY)
    return int(count)


def _hole_count(mask: np.ndarray) -> int:
    background_labels, _ = ndimage.label(~mask, structure=_BACKGROUND_CONNECTIVITY)
    border_labels = np.unique(
        np.concatenate(
            (
                background_labels[0, :],
                background_labels[-1, :],
                background_labels[:, 0],
                background_labels[:, -1],
            )
        )
    )
    labels = np.unique(background_labels)
    enclosed_labels = (labels != 0) & ~np.isin(labels, border_labels)
    return int(np.count_nonzero(enclosed_labels))


@dataclass(frozen=True, slots=True)
class TopologyOperator:
    """Return foreground components, holes, skeleton endpoints, and junctions."""

    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")

    def transform(self, image: Image) -> FeatureVector:
        binary = validated_image(image) >= self.threshold
        skeleton = skeletonize_image(image, threshold=self.threshold)
        endpoints, junctions = detect_endpoints_and_junctions(
            skeleton_to_graph(skeleton)
        )
        return np.asarray(
            [
                _component_count(binary),
                _hole_count(binary),
                len(endpoints),
                len(junctions),
            ],
            dtype=np.float32,
        )
