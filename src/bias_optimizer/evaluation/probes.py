"""Fixed, independently instantiated learner probes for bias evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Probe:
    """A standardized classifier wrapper with a fixed estimator configuration."""

    def __init__(self, name: str, estimator: object) -> None:
        self._name = name
        self._model = make_pipeline(StandardScaler(), estimator)
        self._fitted = False

    @property
    def name(self) -> str:
        return self._name

    def fit(self, features: NDArray, labels: NDArray) -> None:
        matrix = _matrix(features, "features")
        target = _labels(labels, len(matrix))
        if np.unique(target).size < 2:
            raise ValueError("training data must contain at least two classes")
        self._model.fit(matrix, target)
        self._fitted = True

    def predict(self, features: NDArray) -> NDArray[np.int64]:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        matrix = _matrix(features, "features")
        return np.asarray(self._model.predict(matrix), dtype=np.int64)

    def predict_proba(self, features: NDArray) -> NDArray[np.float64] | None:
        if not self._fitted:
            raise RuntimeError("fit must be called before predict_proba")
        estimator = self._model[-1]
        if getattr(estimator, "probability", True) is False:
            return None
        if not hasattr(estimator, "predict_proba"):
            return None
        matrix = _matrix(features, "features")
        try:
            probabilities = np.asarray(self._model.predict_proba(matrix), dtype=np.float64)
        except AttributeError:
            return None
        totals = probabilities.sum(axis=1, keepdims=True)
        if np.any(totals <= 0) or not np.isfinite(totals).all():
            raise ValueError("probe returned invalid probability totals")
        return probabilities / totals


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    name: str
    primary: bool

    def create(self, *, seed: int) -> Probe:
        if self.name == "linear_logreg":
            estimator = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=1_000, random_state=seed
            )
        elif self.name == "rbf_svm":
            estimator = SVC(
                kernel="rbf", C=10.0, gamma="scale", random_state=seed
            )
        elif self.name == "small_mlp":
            estimator = MLPClassifier(
                hidden_layer_sizes=(128,),
                alpha=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=300,
                random_state=seed,
            )
        elif self.name == "knn":
            estimator = KNeighborsClassifier(n_neighbors=5, weights="distance")
        else:
            raise ValueError(f"unknown frozen probe {self.name!r}")
        return Probe(self.name, estimator)


PROBE_DEFINITIONS = {
    "linear_logreg": ProbeDefinition("linear_logreg", primary=True),
    "rbf_svm": ProbeDefinition("rbf_svm", primary=True),
    "small_mlp": ProbeDefinition("small_mlp", primary=True),
    "knn": ProbeDefinition("knn", primary=False),
}


def _matrix(values: NDArray, name: str) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def _labels(values: NDArray, expected: int) -> NDArray[np.int64]:
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != expected:
        raise ValueError("labels must be 1D and aligned with feature rows")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels must have an integer dtype")
    return np.asarray(labels, dtype=np.int64)
