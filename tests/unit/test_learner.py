from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.ml.learner import Learner, LearnerConfig


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, labels = make_classification(
        n_samples=180,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=19,
    )
    return features[:120], labels[:120], features[120:], labels[120:]


def test_fixed_learner_predictions_are_reproducible(
    classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    train_x, train_y, eval_x, _ = classification_data
    config = LearnerConfig(seed=31)

    first = Learner(config).fit_predict(train_x, train_y, eval_x)
    second = Learner(config).fit_predict(train_x, train_y, eval_x)

    np.testing.assert_array_equal(first, second)


def test_evaluator_returns_metrics_and_timings(
    classification_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    train_x, train_y, eval_x, eval_y = classification_data

    result = Evaluator().evaluate_features(train_x, train_y, eval_x, eval_y)

    assert 0 <= result.accuracy <= 1
    assert result.confusion_matrix.shape == (10, 10)
    assert result.confusion_matrix.sum() == len(eval_y)
    assert result.training_time_ms >= 0
    assert result.inference_time_ms >= 0
    assert not result.confusion_matrix.flags.writeable


def test_learner_rejects_non_finite_features() -> None:
    train_x = np.array([[0.0], [np.nan]], dtype=np.float32)
    train_y = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="finite"):
        Learner().fit(train_x, train_y)
