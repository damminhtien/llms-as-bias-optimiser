"""Generic split-safe runner for representation × probe × dataset evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.metrics import classification_metrics
from bias_optimizer.evaluation.probes import PROBE_DEFINITIONS, Probe
from bias_optimizer.evaluation.protocol import EvaluationDataset, SplitRegistry
from bias_optimizer.evaluation.representations import (
    Representation,
    RepresentationDefinition,
)
from bias_optimizer.evaluation.schema import EvaluationKey, EvaluationResult


@dataclass(slots=True)
class FittedEvaluation:
    """Clean-fitted objects reused for original and transformed evaluation inputs."""

    key: EvaluationKey
    representation: Representation
    probe: Probe
    feature_dim: int
    class_labels: NDArray[np.int64]
    feature_time_ms: float
    train_time_ms: float
    train_indices: NDArray[np.int64]

    def predict_features(self, features: NDArray) -> tuple[NDArray[np.int64], float]:
        started = perf_counter()
        predictions = self.probe.predict(features)
        return predictions, (perf_counter() - started) * 1_000

    def predict(self, images: NDArray) -> tuple[NDArray[np.int64], NDArray | None, float]:
        started = perf_counter()
        features = self.representation.transform(images)
        feature_time = (perf_counter() - started) * 1_000
        predictions, inference_time = self.predict_features(features)
        probabilities = self.probe.predict_proba(features)
        return predictions, probabilities, feature_time + inference_time


class EvaluationRunner:
    """Create fresh fitted objects and retain paired predictions for every run."""

    def __init__(
        self,
        representations: dict[str, RepresentationDefinition],
        *,
        cache: FeatureMatrixCache | None = None,
        split_registry: SplitRegistry | None = None,
    ) -> None:
        self.representations = dict(representations)
        self.cache = cache or FeatureMatrixCache()
        self.split_registry = split_registry or SplitRegistry()

    def fit_model(
        self,
        *,
        dataset: EvaluationDataset,
        representation: str,
        probe: str,
        train_size: int,
        seed: int,
    ) -> FittedEvaluation:
        definition = self.representations[representation]
        probe_definition = PROBE_DEFINITIONS[probe]
        indices = self.split_registry.indices(dataset.train_labels, train_size, seed)
        train_images = dataset.train_images[indices]
        train_labels = dataset.train_labels[indices]
        train_split_id = self.split_registry.split_id(dataset, indices)
        key = EvaluationKey(
            representation=representation,
            probe=probe,
            dataset=dataset.name,
            train_size=train_size,
            seed=seed,
            train_split_id=train_split_id,
            evaluation_split_id=dataset.evaluation_split_id,
        )
        fitted_representation = definition.create()
        feature_start = perf_counter()
        fitted_representation.fit(train_images, train_labels)
        state_id = fitted_representation.fitted_state_id
        train_features = self._features(
            dataset=dataset,
            definition=definition,
            representation=fitted_representation,
            images=train_images,
            split_id=train_split_id,
            train_size=train_size,
            seed=seed,
            state_id=state_id,
            role="train",
        )
        feature_time = (perf_counter() - feature_start) * 1_000
        fitted_probe = probe_definition.create(seed=seed)
        fit_start = perf_counter()
        fitted_probe.fit(train_features, train_labels)
        train_time = (perf_counter() - fit_start) * 1_000
        classes = np.unique(dataset.train_labels)
        return FittedEvaluation(
            key=key,
            representation=fitted_representation,
            probe=fitted_probe,
            feature_dim=definition.dimension,
            class_labels=classes,
            feature_time_ms=feature_time,
            train_time_ms=train_time,
            train_indices=indices,
        )

    def evaluate(
        self,
        *,
        dataset: EvaluationDataset,
        representation: str,
        probe: str,
        train_size: int,
        seed: int,
    ) -> EvaluationResult:
        model = self.fit_model(
            dataset=dataset,
            representation=representation,
            probe=probe,
            train_size=train_size,
            seed=seed,
        )
        feature_start = perf_counter()
        evaluation_features = self._features(
            dataset=dataset,
            definition=self.representations[representation],
            representation=model.representation,
            images=dataset.evaluation_images,
            split_id=dataset.evaluation_split_id,
            train_size=train_size,
            seed=seed,
            state_id=model.representation.fitted_state_id,
            role="evaluation",
        )
        feature_time = model.feature_time_ms + (perf_counter() - feature_start) * 1_000
        inference_start = perf_counter()
        predictions = model.probe.predict(evaluation_features)
        inference_time = (perf_counter() - inference_start) * 1_000
        probabilities = model.probe.predict_proba(evaluation_features)
        computed = classification_metrics(
            dataset.evaluation_labels,
            predictions,
            class_labels=model.class_labels,
            probabilities=probabilities,
        )
        return EvaluationResult(
            key=model.key,
            feature_dim=model.feature_dim,
            accuracy=float(computed["accuracy"]),
            error_rate=float(computed["error_rate"]),
            train_time_ms=model.train_time_ms,
            inference_time_ms=inference_time,
            feature_time_ms=feature_time,
            predictions=predictions,
            targets=dataset.evaluation_labels,
            confusion_matrix=computed["confusion_matrix"],
            evaluation_indices=dataset.evaluation_indices,
            log_loss=computed["log_loss"],
        )

    def _features(
        self,
        *,
        dataset: EvaluationDataset,
        definition: RepresentationDefinition,
        representation: Representation,
        images: NDArray,
        split_id: str,
        train_size: int,
        seed: int,
        state_id: str,
        role: str,
    ) -> NDArray[np.float32]:
        key = self.cache.cache_key(
            dataset_fingerprint=dataset.fingerprint,
            split_id=split_id,
            representation_id=definition.name,
            representation_version=definition.version,
            train_size=train_size,
            seed=seed,
            fitted_state_id=state_id,
            role=role,
        )
        return self.cache.get_or_compute(
            key=key,
            compute=lambda: representation.transform(images),
            expected_rows=len(images),
            expected_columns=definition.dimension,
        )
