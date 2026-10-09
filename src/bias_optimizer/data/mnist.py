"""Deterministic MNIST splits with separate search and final-evaluation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ImageArray = NDArray[np.float32]
LabelArray = NDArray[np.int64]

_TRAIN_SIZE = 60_000
_TEST_SIZE = 10_000
_IMAGE_SHAPE = (28, 28)


@dataclass(frozen=True, slots=True)
class MNISTDataConfig:
    """Settings that fix the search split and local OpenML cache location."""

    seed: int = 42
    validation_size: int = 2_000
    data_dir: Path = Path("data")

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= np.iinfo(np.uint32).max:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if self.validation_size <= 0:
            raise ValueError("validation_size must be positive")
        object.__setattr__(self, "data_dir", Path(self.data_dir))


def _readonly_array(values: NDArray, dtype: type) -> NDArray:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _validate_images_and_labels(images: NDArray, labels: NDArray) -> None:
    if images.ndim != 3 or images.shape[1:] != _IMAGE_SHAPE:
        raise ValueError("images must have shape (n, 28, 28)")
    if labels.ndim != 1 or len(images) != len(labels):
        raise ValueError("labels must be a 1D array aligned with images")
    if not np.isfinite(images).all():
        raise ValueError("images must contain only finite values")


def _stratified_sample_indices(
    labels: LabelArray,
    size: int,
    seed: int,
) -> NDArray[np.int64]:
    if size <= 0 or size > len(labels):
        raise ValueError(f"size must be between 1 and {len(labels)}")
    if size == len(labels):
        return np.arange(size, dtype=np.int64)
    indices = np.arange(len(labels))
    try:
        selected, _, _, _ = train_test_split(
            indices,
            labels,
            train_size=size,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"cannot draw a stratified subset of size {size}") from exc
    return np.asarray(selected, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class MNISTSearchData:
    """Training and validation data only; deliberately has no test-set field."""

    train_images: ImageArray
    train_labels: LabelArray
    validation_images: ImageArray
    validation_labels: LabelArray
    seed: int = 42

    def __post_init__(self) -> None:
        train_images = _readonly_array(self.train_images, np.float32)
        train_labels = _readonly_array(self.train_labels, np.int64)
        validation_images = _readonly_array(self.validation_images, np.float32)
        validation_labels = _readonly_array(self.validation_labels, np.int64)
        _validate_images_and_labels(train_images, train_labels)
        _validate_images_and_labels(validation_images, validation_labels)
        object.__setattr__(self, "train_images", train_images)
        object.__setattr__(self, "train_labels", train_labels)
        object.__setattr__(self, "validation_images", validation_images)
        object.__setattr__(self, "validation_labels", validation_labels)

    def sample_training_data(
        self,
        size: int,
        seed: int | None = None,
    ) -> tuple[ImageArray, LabelArray]:
        """Return a reproducible, class-stratified training subset."""
        if size == len(self.train_labels):
            return self.train_images, self.train_labels
        selected = _stratified_sample_indices(
            self.train_labels,
            size,
            self.seed if seed is None else seed,
        )
        return self.train_images[selected], self.train_labels[selected]


@dataclass(frozen=True, slots=True)
class MNISTFinalData:
    """Full official train and test partitions for use after search is frozen."""

    train_images: ImageArray
    train_labels: LabelArray
    test_images: ImageArray
    test_labels: LabelArray

    def __post_init__(self) -> None:
        train_images = _readonly_array(self.train_images, np.float32)
        train_labels = _readonly_array(self.train_labels, np.int64)
        test_images = _readonly_array(self.test_images, np.float32)
        test_labels = _readonly_array(self.test_labels, np.int64)
        _validate_images_and_labels(train_images, train_labels)
        _validate_images_and_labels(test_images, test_labels)
        object.__setattr__(self, "train_images", train_images)
        object.__setattr__(self, "train_labels", train_labels)
        object.__setattr__(self, "test_images", test_images)
        object.__setattr__(self, "test_labels", test_labels)

    def sample_training_data(
        self,
        size: int,
        seed: int = 42,
    ) -> tuple[ImageArray, LabelArray]:
        """Return a reproducible subset from the full train partition."""
        if size == len(self.train_labels):
            return self.train_images, self.train_labels
        selected = self.sample_training_indices(size, seed)
        return self.train_images[selected], self.train_labels[selected]

    def sample_training_indices(self, size: int, seed: int = 42) -> NDArray[np.int64]:
        """Return indices for a reproducible subset of the train partition."""
        return _stratified_sample_indices(self.train_labels, size, seed)


def split_search_data(
    train_images: NDArray,
    train_labels: NDArray,
    *,
    validation_size: int = 2_000,
    seed: int = 42,
) -> MNISTSearchData:
    """Create a deterministic stratified search split from MNIST train data."""
    images = np.asarray(train_images, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.int64)
    _validate_images_and_labels(images, labels)
    if validation_size <= 0 or validation_size >= len(labels):
        raise ValueError("validation_size must be between 1 and train size - 1")

    train_x, validation_x, train_y, validation_y = train_test_split(
        images,
        labels,
        test_size=validation_size,
        random_state=seed,
        stratify=labels,
    )
    return MNISTSearchData(
        train_images=train_x,
        train_labels=train_y,
        validation_images=validation_x,
        validation_labels=validation_y,
        seed=seed,
    )


def _load_openml_arrays(data_dir: Path) -> tuple[ImageArray, LabelArray]:
    dataset = fetch_openml(
        name="mnist_784",
        version=1,
        as_frame=False,
        parser="pandas",
        data_home=str(data_dir.expanduser()),
    )
    pixels = np.asarray(dataset.data, dtype=np.float32)
    labels = np.asarray(dataset.target, dtype=np.int64)
    expected_rows = _TRAIN_SIZE + _TEST_SIZE
    if pixels.shape != (expected_rows, 784) or labels.shape != (expected_rows,):
        raise ValueError(
            "OpenML mnist_784 version 1 must contain 70,000 rows of 784 pixels"
        )
    if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 255)):
        raise ValueError("MNIST pixels must be finite values in [0, 255]")
    images = (pixels / np.float32(255)).reshape(-1, *_IMAGE_SHAPE)
    return images, labels


def load_mnist_search_data(
    config: MNISTDataConfig | None = None,
) -> MNISTSearchData:
    """Load the official 60k train partition and expose only train/validation."""
    if config is None:
        config = MNISTDataConfig()
    if config.validation_size >= _TRAIN_SIZE:
        raise ValueError("validation_size must be smaller than 60,000")
    images, labels = _load_openml_arrays(config.data_dir)
    return split_search_data(
        images[:_TRAIN_SIZE],
        labels[:_TRAIN_SIZE],
        validation_size=config.validation_size,
        seed=config.seed,
    )


def load_mnist_final_data(
    data_dir: Path = Path("data"),
) -> MNISTFinalData:
    """Load the original 60k/10k partitions for post-search final evaluation."""
    images, labels = _load_openml_arrays(data_dir)
    return MNISTFinalData(
        train_images=images[:_TRAIN_SIZE],
        train_labels=labels[:_TRAIN_SIZE],
        test_images=images[_TRAIN_SIZE:],
        test_labels=labels[_TRAIN_SIZE:],
    )
