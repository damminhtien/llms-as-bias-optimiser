"""Frozen test-time transformations and nuisance-control calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter, minimum_filter, shift
from skimage.transform import rotate

from bias_optimizer.evaluation.representations import (
    ProgramRepresentation,
    Representation,
)

TRANSFORMATIONS = (
    "translate_x_-2", "translate_x_-1", "translate_x_+1", "translate_x_+2",
    "translate_y_-2", "translate_y_-1", "translate_y_+1", "translate_y_+2",
    "rotate_-10", "rotate_-5", "rotate_+5", "rotate_+10",
    "dilation_1px", "erosion_1px", "gaussian_noise_sigma_0.05",
)


def transform_images(images: NDArray, name: str, *, seed: int = 0) -> NDArray[np.float32]:
    """Apply one frozen transformation without changing row order or labels."""
    batch = _validate_images(images)
    if name not in TRANSFORMATIONS:
        raise ValueError(f"unknown frozen transformation {name!r}")
    result = np.empty_like(batch)
    if name.startswith("translate_x_"):
        dx = int(name.rsplit("_", maxsplit=1)[1])
        for index, image in enumerate(batch):
            result[index] = shift(image, (0, dx), order=1, mode="constant", cval=0.0, prefilter=False)
    elif name.startswith("translate_y_"):
        dy = int(name.rsplit("_", maxsplit=1)[1])
        for index, image in enumerate(batch):
            result[index] = shift(image, (dy, 0), order=1, mode="constant", cval=0.0, prefilter=False)
    elif name.startswith("rotate_"):
        degrees = float(name.removeprefix("rotate_"))
        for index, image in enumerate(batch):
            result[index] = rotate(np.array(image, copy=True), degrees, resize=False, preserve_range=True, order=1, mode="constant", cval=0.0)
    elif name == "dilation_1px":
        result[:] = maximum_filter(batch, size=(1, 3, 3), mode="constant", cval=0.0)
    elif name == "erosion_1px":
        result[:] = minimum_filter(batch, size=(1, 3, 3), mode="constant", cval=0.0)
    else:
        rng = np.random.default_rng(seed)
        result[:] = batch + rng.normal(0.0, 0.05, size=batch.shape).astype(np.float32)
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def spatial_reassignment_features(
    representation: Representation,
    images: NDArray,
    *,
    target_op: str,
    seed: int,
) -> NDArray[np.float32]:
    """Apply the frozen V3 event-location destroyer to each image."""
    if not isinstance(representation, ProgramRepresentation):
        raise TypeError("spatial reassignment is defined only for frozen V3 programs")
    return representation.transform_spatial_reassignment(
        images, target_op=target_op, seed=seed
    )


def calibrate_noise_sigma(
    representation: Representation,
    train_images: NDArray,
    *,
    target_features: NDArray,
    seed: int,
    candidates: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35),
) -> dict[str, float]:
    """Match nuisance-noise feature displacement using training images only."""
    images = _validate_images(train_images)
    if len(images) != len(target_features):
        raise ValueError("target intervention features must align with training images")
    clean = representation.transform(images)
    target_distance = _rms_feature_displacement(clean, target_features)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=images.shape).astype(np.float32)
    best: tuple[float, float] | None = None
    for sigma in candidates:
        noisy = np.clip(images + sigma * noise, 0.0, 1.0)
        displaced = representation.transform(noisy)
        distance = _rms_feature_displacement(clean, displaced)
        error = abs(distance - target_distance)
        if best is None or error < best[0]:
            best = (error, sigma)
    assert best is not None
    return {"sigma": float(best[1]), "target_feature_rms": target_distance}


def gaussian_noise_images(images: NDArray, sigma: float, *, seed: int) -> NDArray[np.float32]:
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    batch = _validate_images(images)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=batch.shape).astype(np.float32)
    return np.clip(batch + noise, 0.0, 1.0)


def _rms_feature_displacement(left: NDArray, right: NDArray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("feature matrices must have the same two-dimensional shape")
    return float(np.sqrt(np.mean(np.square(first - second))))


def _validate_images(images: NDArray) -> NDArray[np.float32]:
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (28, 28) or not np.isfinite(batch).all():
        raise ValueError("images must be a finite (N, 28, 28) array")
    if np.any((batch < 0) | (batch > 1)):
        raise ValueError("images must be normalized to [0, 1]")
    return batch
