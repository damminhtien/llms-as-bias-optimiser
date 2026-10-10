from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bias_optimizer.evaluation.protocol import EvaluationDataset
from bias_optimizer.evaluation.representations import RepresentationDefinition
from bias_optimizer.evaluation.robustness import TRANSFORMATIONS, transform_images
from bias_optimizer.evaluation.runner import EvaluationRunner


@dataclass
class _RecordingRepresentation:
    seen_images: list[np.ndarray]
    seen_labels: list[np.ndarray | None]
    fitted: bool = False

    @property
    def name(self) -> str:
        return "recording"

    @property
    def dimension(self) -> int:
        return 1

    @property
    def version(self) -> str:
        return "test-v1"

    @property
    def fitted_state_id(self) -> str:
        return "stateless"

    def fit(self, images, labels=None) -> None:
        self.seen_images.append(images.copy())
        self.seen_labels.append(None if labels is None else labels.copy())
        self.fitted = True

    def transform(self, images) -> np.ndarray:
        assert self.fitted
        return np.asarray(images, dtype=np.float32).mean(axis=(1, 2), keepdims=False)[:, None]


def test_runner_fits_only_the_shared_training_subset_and_never_passes_eval_labels(tmp_path) -> None:
    rng = np.random.default_rng(21)
    train_images = rng.random((180, 28, 28), dtype=np.float32) * 0.2
    train_labels = np.repeat(np.arange(2, dtype=np.int64), 90)
    evaluation_images = 0.8 + rng.random((30, 28, 28), dtype=np.float32) * 0.2
    evaluation_labels = np.repeat(np.arange(2, dtype=np.int64), 15)
    dataset = EvaluationDataset(
        name="leakage-test",
        train_images=train_images,
        train_labels=train_labels,
        evaluation_images=evaluation_images,
        evaluation_labels=evaluation_labels,
        evaluation_indices=np.arange(30, dtype=np.int64),
        fingerprint="leakage-test-fingerprint",
        evaluation_split_id="evaluation-only",
        evaluation_partition="held-out",
        provenance={"source": "synthetic"},
    )
    seen_images: list[np.ndarray] = []
    seen_labels: list[np.ndarray | None] = []
    definition = RepresentationDefinition(
        name="recording",
        dimension=1,
        version="test-v1",
        factory=lambda: _RecordingRepresentation(seen_images, seen_labels),
    )
    runner = EvaluationRunner({"recording": definition}, cache=None)

    model = runner.fit_model(
        dataset=dataset,
        representation="recording",
        probe="linear_logreg",
        train_size=80,
        seed=11,
    )
    expected = dataset.train_images[model.train_indices]
    np.testing.assert_array_equal(seen_images[0], expected)
    np.testing.assert_array_equal(seen_labels[0], dataset.train_labels[model.train_indices])
    assert not np.any(seen_images[0] >= 0.8)
    predictions, _, _ = model.predict(dataset.evaluation_images)
    assert predictions.shape == dataset.evaluation_labels.shape


def test_same_seed_split_is_shared_across_representation_factories(tmp_path) -> None:
    rng = np.random.default_rng(8)
    images = rng.random((180, 28, 28), dtype=np.float32)
    labels = np.repeat(np.arange(2, dtype=np.int64), 90)
    dataset = EvaluationDataset(
        name="shared-split-test",
        train_images=images,
        train_labels=labels,
        evaluation_images=images[:30],
        evaluation_labels=labels[:30],
        evaluation_indices=np.arange(30, dtype=np.int64),
        fingerprint="shared-split-fingerprint",
        evaluation_split_id="shared-eval",
        evaluation_partition="synthetic",
        provenance={"source": "synthetic"},
    )
    events: list[tuple[str, np.ndarray]] = []

    @dataclass
    class NamedRecording:
        representation_name: str

        @property
        def name(self) -> str:
            return self.representation_name

        @property
        def dimension(self) -> int:
            return 1

        @property
        def version(self) -> str:
            return "shared-test-v1"

        @property
        def fitted_state_id(self) -> str:
            return "stateless"

        def fit(self, fit_images, fit_labels=None) -> None:
            events.append((self.name, fit_images.copy()))

        def transform(self, batch) -> np.ndarray:
            return np.asarray(batch, dtype=np.float32).mean(axis=(1, 2))[:, None]

    definitions = {
        name: RepresentationDefinition(
            name=name,
            dimension=1,
            version="shared-test-v1",
            factory=lambda current=name: NamedRecording(current),
        )
        for name in ("first", "second")
    }
    runner = EvaluationRunner(definitions)
    first = runner.fit_model(dataset=dataset, representation="first", probe="linear_logreg", train_size=80, seed=47)
    second = runner.fit_model(dataset=dataset, representation="second", probe="linear_logreg", train_size=80, seed=47)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(events[0][1], events[1][1])
    assert first.representation is not second.representation
    assert first.probe is not second.probe


def test_frozen_robustness_transforms_accept_read_only_images() -> None:
    images = np.zeros((2, 28, 28), dtype=np.float32)
    images.setflags(write=False)
    assert len(TRANSFORMATIONS) == 15
    for name in TRANSFORMATIONS:
        transformed = transform_images(images, name, seed=31)
        assert transformed.shape == images.shape
        assert np.isfinite(transformed).all()
