"""Fixed learner metrics and BiasSpec evaluation on search data only."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, confusion_matrix

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.compiler.bias_compiler import BiasCompiler
from bias_optimizer.data.mnist import (
    MNISTDataConfig,
    MNISTSearchData,
    load_mnist_search_data,
)
from bias_optimizer.domain.bias import BiasSpec, bias_spec_hash
from bias_optimizer.domain.evaluation import Evaluation, ModelEvaluation
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.features.base import CompiledRepresentation
from bias_optimizer.features.pipeline import BatchFeatureExtractor
from bias_optimizer.ml.learner import Learner, LearnerConfig

_DIGIT_LABELS = np.arange(10, dtype=np.int64)
_TRAIN_SIZES = (500, 5_000)


class Evaluator:
    """Evaluate candidate representations on train/validation data, never test."""

    def __init__(
        self,
        learner_config: LearnerConfig | None = None,
        *,
        data_config: MNISTDataConfig | None = None,
        search_data: MNISTSearchData | None = None,
        feature_cache: FeatureCache | None = None,
        subexpression_cache: SubexpressionCache | None = None,
        compiler: BiasCompiler | None = None,
    ) -> None:
        self._learner_config = (
            learner_config if learner_config is not None else LearnerConfig()
        )
        self._data_config = (
            data_config if data_config is not None else MNISTDataConfig()
        )
        self._search_data = search_data
        self._compiler = compiler if compiler is not None else BiasCompiler()
        self._feature_extractor = BatchFeatureExtractor(
            feature_cache if feature_cache is not None else FeatureCache(),
            subexpression_cache=(
                subexpression_cache
                if subexpression_cache is not None
                else SubexpressionCache()
            ),
        )

    @property
    def subexpression_cache_metrics(self) -> dict[str, int | float]:
        """Expose bounded shared-cache counters for run-level reports."""
        cache = self._feature_extractor.subexpression_cache
        return cache.metrics if cache is not None else {}

    def evaluate(self, bias: BiasSpec) -> Evaluation:
        """Compile and score a bias with two fixed train sizes on validation data."""
        if not isinstance(bias, BiasSpec):
            raise TypeError("Evaluator.evaluate requires a BiasSpec")
        pipeline = self._compiler.compile(bias)
        return self.evaluate_pipeline(
            pipeline,
            cache_key=bias_spec_hash(bias),
            cache_namespace="mnist_v1",
        )

    def evaluate_pipeline(
        self,
        pipeline: CompiledRepresentation,
        *,
        cache_key: str,
        cache_namespace: str,
    ) -> Evaluation:
        """Evaluate any fixed-width compiled representation on search splits only."""
        if not callable(getattr(pipeline, "transform", None)):
            raise TypeError("pipeline must implement transform()")
        if type(getattr(pipeline, "feature_dim", None)) is not int:
            raise TypeError("pipeline must declare an integer feature_dim")
        if not cache_key or not cache_namespace:
            raise ValueError("cache_key and cache_namespace must be non-empty")
        data = self._get_search_data()
        train_sets = {
            size: data.sample_training_data(size, seed=self._learner_config.seed)
            for size in _TRAIN_SIZES
        }

        feature_start = perf_counter()
        validation_features = self._feature_extractor.transform(
            pipeline,
            data.validation_images,
            bias_hash=cache_key,
            dataset_key=(
                f"{cache_namespace}/search_validation/split_{data.seed}/"
                f"n_{len(data.validation_labels)}"
            ),
        )
        train_features: dict[int, NDArray[np.float32]] = {}
        for size, (images, _) in train_sets.items():
            train_features[size] = self._feature_extractor.transform(
                pipeline,
                images,
                bias_hash=cache_key,
                dataset_key=(
                    f"{cache_namespace}/search_train/split_{data.seed}/"
                    f"sample_{self._learner_config.seed}/n_{size}"
                ),
            )
        feature_runtime_ms = (perf_counter() - feature_start) * 1_000

        evaluations: dict[int, ModelEvaluation] = {}
        for size, (_, labels) in train_sets.items():
            evaluations[size] = self.evaluate_features(
                train_features[size],
                labels,
                validation_features,
                data.validation_labels,
            )

        return Evaluation(
            accuracy_500=evaluations[500].accuracy,
            accuracy_5000=evaluations[5_000].accuracy,
            feature_dim=pipeline.feature_dim,
            feature_runtime_ms=feature_runtime_ms,
            training_runtime_ms=sum(
                result.training_time_ms for result in evaluations.values()
            ),
            inference_runtime_ms=sum(
                result.inference_time_ms for result in evaluations.values()
            ),
            confusion_matrix=evaluations[5_000].confusion_matrix,
        )

    def evaluate_features(
        self,
        train_x: NDArray,
        train_y: NDArray,
        eval_x: NDArray,
        eval_y: NDArray,
    ) -> ModelEvaluation:
        """Return deterministic classification metrics for fixed feature matrices."""
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

    def _get_search_data(self) -> MNISTSearchData:
        if self._search_data is None:
            self._search_data = load_mnist_search_data(self._data_config)
        return self._search_data


class ProgramEvaluator:
    """Evaluate typed DSL programs with the existing frozen learner protocol."""

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        *,
        compiler: ProgramCompiler | None = None,
    ) -> None:
        self._evaluator = evaluator if evaluator is not None else Evaluator()
        self._compiler = compiler if compiler is not None else ProgramCompiler()

    def evaluate(self, bias: ProgramBiasSpec) -> Evaluation:
        if not isinstance(bias, ProgramBiasSpec):
            raise TypeError("ProgramEvaluator.evaluate requires a ProgramBiasSpec")
        pipeline = self._compiler.compile(bias.program)
        return self._evaluator.evaluate_pipeline(
            pipeline,
            cache_key=program_bias_hash(bias),
            cache_namespace=(
                f"mnist_v2_{self._compiler.constraints.track.value}_program"
            ),
        )

    @property
    def subexpression_cache_metrics(self) -> dict[str, int | float]:
        return self._evaluator.subexpression_cache_metrics
