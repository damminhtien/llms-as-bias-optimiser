"""Metrics for fixed feature matrices; representation extraction is separate."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix

from bias_optimizer.domain.evaluation import ModelEvaluation
from bias_optimizer.ml.learner import Learner, LearnerConfig

_DIGIT_LABELS = np.arange(10, dtype=np.int64)


class Evaluator:
    """Measure the fixed learner without allowing candidate code to set fitness."""

    def __init__(self, learner_config: LearnerConfig = LearnerConfig()) -> None:
        self._learner_config = learner_config

    def evaluate_features(
        self,
        train_x: NDArray,
        train_y: NDArray,
        eval_x: NDArray,
        eval_y: NDArray,
    ) -> ModelEvaluation:
        """Return deterministic classification metrics and measured runtimes."""
        expected = np.asarray(eval_y)
        if expected.ndim != 1 or len(expected) != len(eval_x):
            raise ValueError("eval_y must align with eval_x")
        if not np.issubdtype(expected.dtype, np.integer):
            raise TypeError("eval_y must have an integer dtype")
        if len(expected) == 0:
            raise ValueError("evaluation data cannot be empty")

        learner = Learner(self._learner_config)
        fit_start = perf_counter()
        learner.fit(train_x, train_y)
        training_time_ms = (perf_counter() - fit_start) * 1_000

        predict_start = perf_counter()
        predictions = learner.predict(eval_x)
        inference_time_ms = (perf_counter() - predict_start) * 1_000

        return ModelEvaluation(
            accuracy=float(accuracy_score(expected, predictions)),
            confusion_matrix=confusion_matrix(
                expected,
                predictions,
                labels=_DIGIT_LABELS,
            ),
            training_time_ms=training_time_ms,
            inference_time_ms=inference_time_ms,
        )
