"""Tests for BiasSpec evaluation over the search split."""

import numpy as np
import pytest

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.data.mnist import MNISTSearchData
from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.ml.evaluator import Evaluator


def _search_data() -> MNISTSearchData:
    train_labels = np.arange(5_000, dtype=np.int64) % 10
    train_images = np.zeros((5_000, 28, 28), dtype=np.float32)
    for digit in range(10):
        train_images[train_labels == digit, digit * 2, :] = 1

    validation_labels = np.arange(100, dtype=np.int64) % 10
    validation_images = np.zeros((100, 28, 28), dtype=np.float32)
    for digit in range(10):
        validation_images[validation_labels == digit, digit * 2, :] = 1

    return MNISTSearchData(
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        seed=42,
    )


def _raw_pixels_bias() -> BiasSpec:
    return BiasSpec(
        name="raw_pixel_control",
        hypothesis="The fixed learner should separate these synthetic controls.",
        operators=(OperatorSpec("raw_pixels"),),
        prediction="Accuracy should be one on the deterministic split.",
        falsification="Reject if validation accuracy falls below one.",
    )


def test_evaluator_runs_500_and_5000_training_sizes_and_caches_features(
    tmp_path,
) -> None:
    evaluator = Evaluator(
        search_data=_search_data(),
        feature_cache=FeatureCache(tmp_path),
    )
    bias = _raw_pixels_bias()

    result = evaluator.evaluate(bias)
    repeated = evaluator.evaluate(bias)

    assert result.accuracy_500 == pytest.approx(1)
    assert result.accuracy_5000 == pytest.approx(1)
    assert result.feature_dim == 784
    assert result.confusion_matrix.shape == (10, 10)
    assert result.confusion_matrix.sum() == 100
    assert result.feature_runtime_ms >= 0
    assert result.training_runtime_ms >= 0
    assert result.inference_runtime_ms >= 0
    assert repeated.accuracy_500 == result.accuracy_500
    assert repeated.accuracy_5000 == result.accuracy_5000
    assert len(list(tmp_path.rglob("*.npy"))) == 3


def test_evaluator_has_no_test_data_in_its_search_split(tmp_path) -> None:
    search_data = _search_data()
    evaluator = Evaluator(
        search_data=search_data,
        feature_cache=FeatureCache(tmp_path),
    )

    assert not hasattr(search_data, "test_images")
    result = evaluator.evaluate(_raw_pixels_bias())
    assert result.accuracy_5000 == pytest.approx(1)


def test_invalid_bias_is_rejected_before_loading_dataset(monkeypatch, tmp_path) -> None:
    import bias_optimizer.ml.evaluator as evaluator_module

    def fail_if_loaded(_config):
        raise AssertionError("dataset should not load before bias compilation")

    monkeypatch.setattr(evaluator_module, "load_mnist_search_data", fail_if_loaded)
    evaluator = Evaluator(feature_cache=FeatureCache(tmp_path))
    invalid_bias = BiasSpec(
        name="invalid",
        hypothesis="unknown operators should fail compilation",
        operators=(OperatorSpec("unknown_operator"),),
        prediction="none",
        falsification="none",
    )

    with pytest.raises(ValueError, match="unknown operator"):
        evaluator.evaluate(invalid_bias)


def test_evaluation_ranking_score_uses_accuracy_dimension_and_runtime() -> None:
    result = Evaluation(
        accuracy_500=0.8,
        accuracy_5000=0.9,
        feature_dim=784,
        feature_runtime_ms=10,
        training_runtime_ms=20,
        inference_runtime_ms=5,
        confusion_matrix=np.zeros((10, 10), dtype=np.int64),
    )
    expected = 0.6 * 0.8 + 0.4 * 0.9 - 0.001 * np.log1p(784) - 0.0001 * np.log1p(35)

    assert result.ranking_score == pytest.approx(expected)
    assert result.runtime_ms == 35
    assert not result.confusion_matrix.flags.writeable


def test_evaluation_rejects_invalid_metrics_and_confusion_shape() -> None:
    with pytest.raises(ValueError, match="accuracy_500"):
        Evaluation(
            accuracy_500=1.1,
            accuracy_5000=0.9,
            feature_dim=5,
            feature_runtime_ms=0,
            training_runtime_ms=0,
            inference_runtime_ms=0,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="10, 10"):
        Evaluation(
            accuracy_500=0.8,
            accuracy_5000=0.9,
            feature_dim=5,
            feature_runtime_ms=0,
            training_runtime_ms=0,
            inference_runtime_ms=0,
            confusion_matrix=np.zeros((2, 2), dtype=np.int64),
        )
