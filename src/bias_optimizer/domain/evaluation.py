"""Measured results from training and predicting with the fixed learner."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

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
