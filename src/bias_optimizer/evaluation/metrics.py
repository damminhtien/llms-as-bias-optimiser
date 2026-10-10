"""Classification, learning-curve, and paired-difference summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def classification_metrics(
    targets: NDArray,
    predictions: NDArray,
    *,
    class_labels: NDArray,
    probabilities: NDArray | None = None,
) -> dict[str, object]:
    truth = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(class_labels, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("targets and predictions must be aligned 1D arrays")
    score = float(accuracy_score(truth, predicted))
    loss: float | None = None
    if probabilities is not None:
        probability_matrix = np.asarray(probabilities, dtype=np.float64)
        if probability_matrix.shape != (len(truth), len(labels)):
            raise ValueError("probabilities must align with targets and class_labels")
        loss = float(log_loss(truth, probability_matrix, labels=labels))
    return {
        "accuracy": score,
        "error_rate": 1.0 - score,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels),
        "log_loss": loss,
    }


def learning_curve_aulc(
    accuracies: Mapping[int, float] | Sequence[tuple[int, float]],
) -> float:
    """Integrate accuracy over log training size with trapezoidal integration."""
    items = list(accuracies.items()) if isinstance(accuracies, Mapping) else list(accuracies)
    items.sort(key=lambda item: item[0])
    sizes = np.asarray([item[0] for item in items], dtype=np.float64)
    scores = np.asarray([item[1] for item in items], dtype=np.float64)
    if len(sizes) < 2 or np.any(sizes <= 0) or np.any(np.diff(sizes) <= 0):
        raise ValueError("AULC requires at least two increasing positive training sizes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("accuracies must be finite values in [0, 1]")
    return float(np.trapezoid(scores, x=np.log(sizes)))


def nonlinear_probe_gain(linear_accuracy: float, nonlinear_accuracy: float) -> float:
    """Return probe-accessible task-performance gain over the linear probe."""
    return float(nonlinear_accuracy - linear_accuracy)


def pareto_frontier(points: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep points not dominated in lower dimension and higher accuracy."""
    unique = sorted({(int(d), float(a)) for d, a in points}, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[int, float]] = []
    best_accuracy = -np.inf
    for dimension, accuracy in unique:
        if accuracy > best_accuracy:
            frontier.append((dimension, accuracy))
            best_accuracy = accuracy
    return frontier
