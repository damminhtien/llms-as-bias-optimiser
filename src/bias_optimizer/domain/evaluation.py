"""Measured results from training and predicting with the fixed learner."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log1p

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Metrics for one fixed-feature train/evaluation run."""

    accuracy: float
    confusion_matrix: NDArray[np.int64]
    training_time_ms: float
    inference_time_ms: float

    def __post_init__(self) -> None:
        if not isfinite(self.accuracy) or not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be finite and between 0 and 1")
        for name in ("training_time_ms", "inference_time_ms"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        matrix = np.array(self.confusion_matrix, dtype=np.int64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("confusion_matrix must be square")
        if np.any(matrix < 0):
            raise ValueError("confusion_matrix cannot contain negative counts")
        matrix.setflags(write=False)
        object.__setattr__(self, "confusion_matrix", matrix)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Validation results; the 5,000-sample score is optional for screening."""

    accuracy_500: float
    accuracy_5000: float | None
    feature_dim: int
    feature_runtime_ms: float
    training_runtime_ms: float
    inference_runtime_ms: float
    confusion_matrix: NDArray[np.int64]

    def __post_init__(self) -> None:
        if not isfinite(self.accuracy_500) or not 0.0 <= self.accuracy_500 <= 1.0:
            raise ValueError("accuracy_500 must be finite and between 0 and 1")
        if self.accuracy_5000 is not None and (
            not isfinite(self.accuracy_5000) or not 0.0 <= self.accuracy_5000 <= 1.0
        ):
            raise ValueError("accuracy_5000 must be finite and between 0 and 1")
        if type(self.feature_dim) is not int or self.feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer")
        for name in (
            "feature_runtime_ms",
            "training_runtime_ms",
            "inference_runtime_ms",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        matrix = np.array(self.confusion_matrix, dtype=np.int64, copy=True)
        if matrix.shape != (10, 10):
            raise ValueError("confusion_matrix must have shape (10, 10)")
        if np.any(matrix < 0):
            raise ValueError("confusion_matrix cannot contain negative counts")
        matrix.setflags(write=False)
        object.__setattr__(self, "confusion_matrix", matrix)

    @property
    def runtime_ms(self) -> float:
        return (
            self.feature_runtime_ms
            + self.training_runtime_ms
            + self.inference_runtime_ms
        )

    @property
    def ranking_score(self) -> float:
        """Use 500-sample quality for screening, then 0.6/0.4 for full results."""
        accuracy = (
            self.accuracy_500
            if self.accuracy_5000 is None
            else 0.6 * self.accuracy_500 + 0.4 * self.accuracy_5000
        )
        return (
            accuracy - 0.001 * log1p(self.feature_dim) - 0.0001 * log1p(self.runtime_ms)
        )
