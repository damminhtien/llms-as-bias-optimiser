"""Run and save the three human-designed MNIST representation baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from bias_optimizer.data import MNISTDataConfig, load_mnist_search_data
from bias_optimizer.features.base import FeatureOperator
from bias_optimizer.features.baselines import (
    DownsampledPixelsOperator,
    HOGOperator,
    RawPixelsOperator,
)
from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.ml.learner import LearnerConfig


def _transform_batch(
    operator: FeatureOperator,
    images: NDArray[np.float32],
) -> tuple[NDArray[np.float32], float]:
    started = perf_counter()
    features = np.stack([operator.transform(image) for image in images])
    return np.asarray(features, dtype=np.float32), (perf_counter() - started) * 1_000


def run_baselines(
    *,
    seed: int = 42,
    validation_size: int = 2_000,
    data_dir: Path = Path("data"),
) -> dict[str, object]:
    """Evaluate raw, downsampled, and HOG features using the fixed learner."""
    dataset = load_mnist_search_data(
        MNISTDataConfig(
            seed=seed,
            validation_size=validation_size,
            data_dir=data_dir,
        )
    )
    evaluator = Evaluator(LearnerConfig(seed=seed))
    operators: tuple[tuple[str, FeatureOperator], ...] = (
        ("raw_pixels", RawPixelsOperator()),
        ("downsampled_pixels_14x14", DownsampledPixelsOperator()),
        ("hog_9bin_4x4", HOGOperator()),
    )

    records: list[dict[str, object]] = []
    validation_cache: dict[str, tuple[NDArray[np.float32], float]] = {}
    for name, operator in operators:
        validation_cache[name] = _transform_batch(
            operator,
            dataset.validation_images,
        )
        validation_features, validation_feature_time_ms = validation_cache[name]

        for train_size in (500, 5_000):
            train_images, train_labels = dataset.sample_training_data(
                train_size,
                seed=seed,
            )
            train_features, train_feature_time_ms = _transform_batch(
                operator,
                train_images,
            )
            evaluation = evaluator.evaluate_features(
                train_features,
                train_labels,
                validation_features,
                dataset.validation_labels,
            )
            records.append(
                {
                    "representation": name,
                    "train_size": train_size,
                    "seed": seed,
                    "feature_dim": int(train_features.shape[1]),
                    "accuracy": evaluation.accuracy,
                    "confusion_matrix": evaluation.confusion_matrix.tolist(),
                    "train_feature_runtime_ms": train_feature_time_ms,
                    "validation_feature_runtime_ms": validation_feature_time_ms,
                    "training_time_ms": evaluation.training_time_ms,
                    "inference_time_ms": evaluation.inference_time_ms,
                }
            )

    return {
        "dataset": "OpenML mnist_784 version 1",
        "split_seed": seed,
        "validation_size": validation_size,
        "validation_is_the_only_search_evaluation_set": True,
        "learner": {
            "scaler": "StandardScaler",
            "classifier": "multinomial LogisticRegression",
            "regularization_c": 1.0,
            "max_iter": 1_000,
            "solver": "lbfgs",
            "seed": seed,
        },
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines.json"))
    args = parser.parse_args()

    report = run_baselines(
        seed=args.seed,
        validation_size=args.validation_size,
        data_dir=args.data_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved baseline results to {args.output}")


if __name__ == "__main__":
    main()
