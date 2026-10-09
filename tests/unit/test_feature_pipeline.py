"""Tests for composed feature transformations and cached batch extraction."""

from dataclasses import dataclass

import numpy as np
import pytest

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.domain.bias import BiasSpec, OperatorSpec, bias_spec_hash
from bias_optimizer.features.pipeline import BatchFeatureExtractor, FeaturePipeline
from bias_optimizer.features.registry import OperatorRegistry
from bias_optimizer.features.spatial import SpatialOperator
from bias_optimizer.features.symmetry import SymmetryOperator


@dataclass
class _CountingOperator:
    calls: int = 0

    @property
    def feature_dim(self) -> int:
        return 1

    def transform(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.asarray([image.sum()], dtype=np.float32)


@dataclass(frozen=True)
class _InvalidOperator:
    result: np.ndarray

    @property
    def feature_dim(self) -> int:
        return 1

    def transform(self, image: np.ndarray) -> np.ndarray:
        return self.result


def _bias_spec() -> BiasSpec:
    return BiasSpec(
        name="spatial_and_symmetry",
        hypothesis="Coarse position and reflection encode complementary structure.",
        operators=(OperatorSpec("spatial"), OperatorSpec("symmetry")),
        prediction="The combined representation improves low-data accuracy.",
        falsification="Reject if validation accuracy does not improve.",
    )


def test_pipeline_concatenates_operator_outputs_in_declared_order() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    image[0, :2] = 1
    image[13, 0] = 1
    image[27, 0] = 1
    pipeline = FeaturePipeline((SpatialOperator(), SymmetryOperator()))

    features = pipeline.transform(image)

    np.testing.assert_allclose(
        features,
        np.concatenate(
            (SpatialOperator().transform(image), SymmetryOperator().transform(image))
        ),
    )
    assert pipeline.feature_dim == 5
    assert features.shape == (pipeline.feature_dim,)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_pipeline_rejects_empty_operators() -> None:
    with pytest.raises(ValueError, match="at least one operator"):
        FeaturePipeline(())


def test_every_registered_operator_declares_its_transform_dimension() -> None:
    image = np.zeros((28, 28), dtype=np.float32)
    registry = OperatorRegistry()

    for name in registry.names:
        operator = registry.build(OperatorSpec(name))
        assert operator.transform(image).shape == (operator.feature_dim,)


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (np.asarray([np.nan], dtype=np.float32), "non-finite"),
        (np.asarray([1, 2], dtype=np.float32), r"expected \(1,\)"),
        (np.asarray([[1]], dtype=np.float32), r"expected \(1,\)"),
    ],
)
def test_pipeline_rejects_invalid_operator_results(
    result: np.ndarray, message: str
) -> None:
    pipeline = FeaturePipeline((_InvalidOperator(result),))

    with pytest.raises(ValueError, match=message):
        pipeline.transform(np.zeros((28, 28), dtype=np.float32))


def test_pipeline_rejects_invalid_images_before_operator_execution() -> None:
    pipeline = FeaturePipeline((SpatialOperator(),))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pipeline.transform(np.full((28, 28), 2, dtype=np.float32))


def test_batch_extractor_transforms_batches_and_handles_empty_batches() -> None:
    pipeline = FeaturePipeline((SpatialOperator(), SymmetryOperator()))
    images = np.zeros((3, 28, 28), dtype=np.float32)

    features = BatchFeatureExtractor().transform(pipeline, images)
    empty = BatchFeatureExtractor().transform(
        pipeline, np.empty((0, 28, 28), dtype=np.float32)
    )

    assert features.shape == (3, pipeline.feature_dim)
    assert empty.shape == (0, pipeline.feature_dim)
    assert np.isfinite(features).all()


def test_feature_cache_keys_by_bias_and_dataset_and_reuses_matrix(tmp_path) -> None:
    bias = _bias_spec()
    bias_hash = bias_spec_hash(bias)
    counter = _CountingOperator()
    pipeline = FeaturePipeline((counter,))
    extractor = BatchFeatureExtractor(FeatureCache(tmp_path))
    images = np.zeros((2, 28, 28), dtype=np.float32)

    first = extractor.transform(
        pipeline, images, bias_hash=bias_hash, dataset_key="search_train_n2_seed42"
    )
    cached = extractor.transform(
        pipeline, images, bias_hash=bias_hash, dataset_key="search_train_n2_seed42"
    )
    extractor.transform(
        pipeline, images, bias_hash=bias_hash, dataset_key="validation_seed42"
    )

    assert counter.calls == 4
    np.testing.assert_array_equal(cached, first)
    assert not cached.flags.writeable
    assert len(list(tmp_path.rglob("*.npy"))) == 2


def test_cached_extraction_requires_both_cache_keys(tmp_path) -> None:
    extractor = BatchFeatureExtractor(FeatureCache(tmp_path))
    pipeline = FeaturePipeline((SpatialOperator(),))

    with pytest.raises(ValueError, match="bias_hash and dataset_key"):
        extractor.transform(pipeline, np.zeros((1, 28, 28), dtype=np.float32))


def test_feature_cache_validates_hash_and_matrix(tmp_path) -> None:
    cache = FeatureCache(tmp_path)
    bias_hash = "a" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        cache.get("../unsafe", "split")
    with pytest.raises(ValueError, match="two-dimensional"):
        cache.set(bias_hash, "split", np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        cache.set(bias_hash, "split", np.asarray([[np.inf]], dtype=np.float32))


def test_bias_spec_hash_is_stable_and_changes_with_operator_order() -> None:
    bias = _bias_spec()
    reordered = BiasSpec(
        name=bias.name,
        hypothesis=bias.hypothesis,
        operators=tuple(reversed(bias.operators)),
        prediction=bias.prediction,
        falsification=bias.falsification,
    )

    assert bias_spec_hash(bias) == bias_spec_hash(BiasSpec.from_json(bias.to_json()))
    assert bias_spec_hash(bias) != bias_spec_hash(reordered)
