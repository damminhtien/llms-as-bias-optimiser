"""A fixed StandardScaler plus multinomial logistic-regression learner."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class LearnerConfig:
    """Fixed classifier settings, independent of candidate bias proposals."""

    regularization_c: float = 1.0
    max_iter: int = 1_000
    seed: int = 42

    def __post_init__(self) -> None:
        if not isfinite(self.regularization_c) or self.regularization_c <= 0:
            raise ValueError("regularization_c must be finite and positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


def _feature_matrix(values: NDArray, name: str) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _labels(values: NDArray, expected_size: int) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != expected_size:
        raise ValueError("labels must be a 1D array aligned with the feature rows")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must have an integer dtype")
    return np.asarray(labels, dtype=np.int64)


class Learner:
    """Train and predict with one fixed standardization/classification pipeline."""

    def __init__(self, config: LearnerConfig = LearnerConfig()) -> None:
        self._config = config
        self._model: Pipeline | None = None

    def fit(self, train_x: NDArray, train_y: NDArray) -> None:
        features = _feature_matrix(train_x, "train_x")
        labels = _labels(train_y, len(features))
        if np.unique(labels).size < 2:
            raise ValueError("training data must contain at least two classes")

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self._config.regularization_c,
                max_iter=self._config.max_iter,
                random_state=self._config.seed,
                solver="lbfgs",
                tol=1e-4,
            ),
        )
        model.fit(features, labels)
        self._model = model

    def predict(self, eval_x: NDArray) -> NDArray[np.int64]:
        if self._model is None:
            raise RuntimeError("fit must be called before predict")
        features = _feature_matrix(eval_x, "eval_x")
        return np.asarray(self._model.predict(features), dtype=np.int64)

    def fit_predict(
        self,
        train_x: NDArray,
        train_y: NDArray,
        eval_x: NDArray,
    ) -> NDArray[np.int64]:
        """Fit on training features and return predictions for evaluation rows."""
        self.fit(train_x, train_y)
        return self.predict(eval_x)
