"""Shared angle calculations for skeleton-path feature operators."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from bias_optimizer.features.skeleton_graph import Pixel

DIRECTION_BINS = 8


def path_angles(path: tuple[Pixel, ...]) -> NDArray[np.float64]:
    if len(path) < 2:
        return np.empty(0, dtype=np.float64)
    points = np.asarray(path, dtype=np.float64)
    delta_y = np.diff(points[:, 0])
    delta_x = np.diff(points[:, 1])
    return np.arctan2(delta_y, delta_x)


def path_angles_for_paths(
    paths: Iterable[tuple[Pixel, ...]],
) -> tuple[NDArray[np.float64], ...]:
    return tuple(path_angles(path) for path in paths)


def quantize_direction(angles: NDArray[np.float64]) -> NDArray[np.int64]:
    bins = np.floor(np.mod(angles, 2 * np.pi) * DIRECTION_BINS / (2 * np.pi))
    return np.asarray(bins, dtype=np.int64) % DIRECTION_BINS


def wrap_angle(angle: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.asarray((np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi)


def is_closed_path(path: tuple[Pixel, ...]) -> bool:
    return len(path) > 2 and path[0] == path[-1]
