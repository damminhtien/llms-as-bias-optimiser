"""Composable feature pipelines and batched MNIST transformation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.features._image import IMAGE_SHAPE, validated_image
from bias_optimizer.features.base import FeatureOperator, FeatureVector, Image


@dataclass(frozen=True, slots=True)
class FeaturePipeline:
    """Concatenate finite features from a fixed ordered operator sequence."""

    operators: tuple[FeatureOperator, ...]

    def __post_init__(self) -> None:
        operators = tuple(self.operators)
        if not operators:
            raise ValueError("feature pipeline must contain at least one operator")
        for operator in operators:
            if not callable(getattr(operator, "transform", None)):
                raise TypeError("pipeline operators must implement transform()")
            dimension = getattr(operator, "feature_dim", None)
            if type(dimension) is not int or dimension <= 0:
                raise ValueError(
                    "pipeline operators must declare a positive feature_dim"
                )
        object.__setattr__(self, "operators", operators)

    @property
    def feature_dim(self) -> int:
        return sum(operator.feature_dim for operator in self.operators)

    def transform(self, image: Image) -> FeatureVector:
        validated = validated_image(image)
        parts: list[FeatureVector] = []
        for operator in self.operators:
            features = np.asarray(operator.transform(validated), dtype=np.float32)
            if features.ndim != 1 or features.size != operator.feature_dim:
                raise ValueError(
                    f"{type(operator).__name__} returned shape {features.shape}; "
                    f"expected ({operator.feature_dim},)"
                )
            if not np.isfinite(features).all():
                raise ValueError(
                    f"{type(operator).__name__} returned non-finite features"
                )
            parts.append(features)
        return np.concatenate(parts).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class BatchFeatureExtractor:
    """Transform image batches and optionally cache matrices by bias and split."""

    cache: FeatureCache | None = None
    subexpression_cache: SubexpressionCache | None = None

    def transform(
        self,
        pipeline: FeaturePipeline,
        images: NDArray[np.float32],
        *,
        bias_hash: str | None = None,
        dataset_key: str | None = None,
    ) -> NDArray[np.float32]:
        """Transform images; dataset_key must identify this exact split/subset."""
        batch = self._validate_batch(images)
        if self.cache is not None:
            if bias_hash is None or dataset_key is None:
                raise ValueError("cached extraction requires bias_hash and dataset_key")
            cached = self.cache.get(bias_hash, dataset_key)
            if cached is not None:
                expected_shape = (len(batch), pipeline.feature_dim)
                if cached.shape != expected_shape:
                    raise ValueError(
                        f"cached feature matrix has shape {cached.shape}; "
                        f"expected {expected_shape}"
                    )
                return cached

        matrix = np.empty((len(batch), pipeline.feature_dim), dtype=np.float32)
        transform_with_cache = getattr(pipeline, "transform_with_cache", None)
        for index, image in enumerate(batch):
            if self.subexpression_cache is not None and callable(transform_with_cache):
                if dataset_key is None:
                    raise ValueError(
                        "cross-candidate caching requires an exact dataset_key"
                    )
                matrix[index] = transform_with_cache(
                    image,
                    cache=self.subexpression_cache,
                    sample_key=f"{dataset_key}:{index}",
                )
            else:
                matrix[index] = pipeline.transform(image)
        if self.cache is not None:
            self.cache.set(bias_hash, dataset_key, matrix)
        return matrix

    @staticmethod
    def _validate_batch(images: NDArray[np.float32]) -> NDArray[np.float32]:
        batch = np.asarray(images, dtype=np.float32)
        if batch.ndim != 3 or batch.shape[1:] != IMAGE_SHAPE:
            raise ValueError("image batches must have shape (N, 28, 28)")
        if not np.isfinite(batch).all():
            raise ValueError("image batches must contain only finite values")
        if np.any((batch < 0) | (batch > 1)):
            raise ValueError("normalized image pixels must be in [0, 1]")
        return batch
