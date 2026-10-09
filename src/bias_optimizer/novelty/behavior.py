"""Behavioral distances between candidate representations on fixed probes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


def pairwise_distance_profile(features: NDArray[np.float32]) -> NDArray[np.float64]:
    """Return the representation's pairwise sample geometry, independent of width."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("features must be a matrix with at least two probe rows")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    return pdist(matrix, metric="euclidean")


def distance_profile_similarity(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> float:
    """Compare two representations by correlation of probe-sample distances."""
    left_values = np.asarray(left, dtype=np.float64).reshape(-1)
    right_values = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_values.shape != right_values.shape or not len(left_values):
        raise ValueError("distance profiles must be non-empty and equally sized")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("distance profiles must be finite")
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0 if left_norm == right_norm else 0.0
    similarity = float(
        np.clip(
            np.dot(left_centered, right_centered) / (left_norm * right_norm), -1.0, 1.0
        )
    )
    return 1.0 if np.isclose(similarity, 1.0, atol=1e-12) else similarity


def behavioral_novelty(
    profile: NDArray[np.float64],
    archive_profiles: dict[str, NDArray[np.float64]],
    *,
    candidate_id: str,
) -> tuple[float, str | None, float]:
    """Score dissimilarity to the closest archived behavior, excluding self."""
    comparisons = [
        (distance_profile_similarity(profile, other), other_id)
        for other_id, other in archive_profiles.items()
        if other_id != candidate_id
    ]
    if not comparisons:
        return 1.0, None, 0.0
    similarity, nearest_id = max(comparisons, key=lambda item: item[0])
    return 1.0 - max(0.0, similarity), nearest_id, similarity
