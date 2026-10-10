"""Immutable records shared by the V3.1 evaluation and reporting layers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EvaluationKey:
    """Identity of one representation/probe fit and its paired evaluation set."""

    representation: str
    probe: str
    dataset: str
    train_size: int
    seed: int
    train_split_id: str
    evaluation_split_id: str

    def __post_init__(self) -> None:
        for name in ("representation", "probe", "dataset", "train_split_id", "evaluation_split_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.train_size <= 0:
            raise ValueError("train_size must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One run, including aligned predictions needed for paired statistics."""

    key: EvaluationKey
    feature_dim: int
    accuracy: float
    error_rate: float
    train_time_ms: float
    inference_time_ms: float
    feature_time_ms: float
    predictions: NDArray[np.int64]
    targets: NDArray[np.int64]
    confusion_matrix: NDArray[np.int64]
    evaluation_indices: NDArray[np.int64]
    log_loss: float | None = None

    def __post_init__(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        for name in ("accuracy", "error_rate", "train_time_ms", "inference_time_ms", "feature_time_ms"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.accuracy > 1 or self.error_rate > 1:
            raise ValueError("accuracy and error_rate must be at most one")
        predictions = np.asarray(self.predictions, dtype=np.int64)
        targets = np.asarray(self.targets, dtype=np.int64)
        indices = np.asarray(self.evaluation_indices, dtype=np.int64)
        matrix = np.asarray(self.confusion_matrix, dtype=np.int64)
        if predictions.ndim != 1 or targets.shape != predictions.shape:
            raise ValueError("predictions and targets must be aligned 1D arrays")
        if indices.shape != predictions.shape:
            raise ValueError("evaluation indices must align with predictions")
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("confusion_matrix must be a square matrix")
        if matrix.sum() != len(targets):
            raise ValueError("confusion_matrix total must match the evaluation size")
        if self.log_loss is not None and (not isfinite(self.log_loss) or self.log_loss < 0):
            raise ValueError("log_loss must be finite and non-negative")
        for value in (predictions, targets, indices, matrix):
            value.setflags(write=False)
        object.__setattr__(self, "predictions", predictions)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "evaluation_indices", indices)
        object.__setattr__(self, "confusion_matrix", matrix)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": {
                "representation": self.key.representation,
                "probe": self.key.probe,
                "dataset": self.key.dataset,
                "train_size": self.key.train_size,
                "seed": self.key.seed,
                "train_split_id": self.key.train_split_id,
                "evaluation_split_id": self.key.evaluation_split_id,
            },
            "feature_dim": self.feature_dim,
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "train_time_ms": self.train_time_ms,
            "inference_time_ms": self.inference_time_ms,
            "feature_time_ms": self.feature_time_ms,
            "log_loss": self.log_loss,
            "predictions": self.predictions.tolist(),
            "targets": self.targets.tolist(),
            "evaluation_indices": self.evaluation_indices.tolist(),
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> EvaluationResult:
        key_data = value["key"]
        if not isinstance(key_data, dict):
            raise TypeError("serialized evaluation key must be an object")
        key = EvaluationKey(**key_data)
        return cls(
            key=key,
            feature_dim=int(value["feature_dim"]),
            accuracy=float(value["accuracy"]),
            error_rate=float(value["error_rate"]),
            train_time_ms=float(value["train_time_ms"]),
            inference_time_ms=float(value["inference_time_ms"]),
            feature_time_ms=float(value["feature_time_ms"]),
            predictions=np.asarray(value["predictions"], dtype=np.int64),
            targets=np.asarray(value["targets"], dtype=np.int64),
            confusion_matrix=np.asarray(value["confusion_matrix"], dtype=np.int64),
            evaluation_indices=np.asarray(value["evaluation_indices"], dtype=np.int64),
            log_loss=None if value.get("log_loss") is None else float(value["log_loss"]),
        )
