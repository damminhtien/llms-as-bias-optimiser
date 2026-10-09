"""Post-selection evaluation of frozen finalists on MNIST's official test set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.compiler.bias_compiler import BiasCompiler
from bias_optimizer.data.mnist import MNISTFinalData, load_mnist_final_data
from bias_optimizer.domain.bias import BiasSpec, OperatorSpec, bias_spec_hash
from bias_optimizer.features.baselines import HOGOperator, RawPixelsOperator
from bias_optimizer.features.pipeline import BatchFeatureExtractor, FeaturePipeline
from bias_optimizer.ml.learner import Learner, LearnerConfig

_DIGIT_LABELS = np.arange(10, dtype=np.int64)
_DEFAULT_TRAIN_SIZES = (250, 500, 1_000, 5_000, 60_000)
_DEFAULT_SEEDS = (11, 23, 47)


@dataclass(frozen=True, slots=True)
class _Representation:
    name: str
    kind: str
    pipeline: FeaturePipeline
    operator_specs: tuple[OperatorSpec, ...]
    candidate_id: str | None = None
    selection_roles: tuple[str, ...] = ()


def _archive_path_for(selection_path: Path, stored_path: str) -> Path:
    archive_path = Path(stored_path)
    if archive_path.is_absolute() or archive_path.exists():
        return archive_path
    workspace_relative = selection_path.parent.parent / archive_path
    return workspace_relative if workspace_relative.exists() else archive_path


def _load_frozen_finalists(
    finalists_path: Path,
) -> tuple[dict[str, Any], tuple[tuple[dict[str, Any], BiasSpec], ...]]:
    """Validate the freeze and archive checksum before any test data is loaded."""
    try:
        metadata = json.loads(finalists_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read finalist selection: {exc}") from exc
    if (
        not isinstance(metadata, dict)
        or metadata.get("finalist_selection_frozen") is not True
    ):
        raise ValueError("finalist selection is not frozen")
    if metadata.get("test_set_accessed") is not False:
        raise ValueError("finalist selection must be frozen before test access")

    stored_archive = metadata.get("search_archive")
    expected_hash = metadata.get("search_archive_sha256")
    if not isinstance(stored_archive, str) or not isinstance(expected_hash, str):
        raise TypeError("finalist selection archive path and checksum must be strings")
    archive_path = _archive_path_for(finalists_path, stored_archive)
    try:
        actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read frozen search archive: {exc}") from exc
    if actual_hash != expected_hash:
        raise ValueError("search archive changed after finalist selection")

    finalist_values = metadata.get("finalists")
    if not isinstance(finalist_values, list) or not finalist_values:
        raise ValueError("frozen selection must contain at least one finalist")

    parsed: list[tuple[dict[str, Any], BiasSpec]] = []
    seen_ids: set[str] = set()
    for finalist in finalist_values:
        if not isinstance(finalist, dict):
            raise TypeError("finalist entries must be JSON objects")
        try:
            bias = BiasSpec.from_dict(finalist["bias"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid finalist bias: {exc}") from exc
        candidate_id = finalist.get("candidate_id")
        if candidate_id != bias_spec_hash(bias):
            raise ValueError(f"candidate ID does not match bias {bias.name!r}")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate finalist candidate ID: {candidate_id}")
        roles = finalist.get("selection_roles")
        if not isinstance(roles, list) or not all(
            isinstance(role, str) for role in roles
        ):
            raise ValueError(f"finalist {bias.name!r} has invalid selection roles")
        seen_ids.add(candidate_id)
        parsed.append((finalist, bias))
    return metadata, tuple(parsed)


def _representation_hash(operator_specs: tuple[OperatorSpec, ...]) -> str:
    canonical = json.dumps(
        [operator.to_dict() for operator in operator_specs],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _representations(
    finalist_values: tuple[tuple[dict[str, Any], BiasSpec], ...],
) -> tuple[_Representation, ...]:
    compiler = BiasCompiler()
    finalists = tuple(
        _Representation(
            name=bias.name,
            kind="finalist",
            pipeline=compiler.compile(bias),
            operator_specs=bias.operators,
            candidate_id=str(value["candidate_id"]),
            selection_roles=tuple(sorted(value["selection_roles"])),
        )
        for value, bias in finalist_values
    )
    hog_specs = (
        OperatorSpec(
            "hog",
            {
                "orientations": 9,
                "pixels_per_cell": [4, 4],
                "cells_per_block": [2, 2],
            },
        ),
    )
    baselines = (
        _Representation(
            name="raw_pixels_baseline",
            kind="baseline",
            pipeline=FeaturePipeline((RawPixelsOperator(),)),
            operator_specs=(OperatorSpec("raw_pixels"),),
        ),
        _Representation(
            name="hog_9bin_4x4_baseline",
            kind="baseline",
            pipeline=FeaturePipeline((HOGOperator(),)),
            operator_specs=hog_specs,
        ),
    )
    return (*finalists, *baselines)


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = np.asarray([run["accuracy"] for run in runs], dtype=np.float64)
    training_times = np.asarray(
        [run["training_runtime_ms"] for run in runs], dtype=np.float64
    )
    inference_times = np.asarray(
        [run["inference_runtime_ms"] for run in runs], dtype=np.float64
    )
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies, ddof=1)) if len(runs) > 1 else 0.0,
        "training_runtime_ms_mean": float(np.mean(training_times)),
        "training_runtime_ms_std": (
            float(np.std(training_times, ddof=1)) if len(runs) > 1 else 0.0
        ),
        "inference_runtime_ms_mean": float(np.mean(inference_times)),
        "inference_runtime_ms_std": (
            float(np.std(inference_times, ddof=1)) if len(runs) > 1 else 0.0
        ),
        "confusion_matrix_sum": np.sum(
            [run["confusion_matrix"] for run in runs], axis=0
        )
        .astype(np.int64)
        .tolist(),
    }


def run_final_evaluation(
    *,
    finalists_path: Path = Path("results/finalists.json"),
    output_path: Path = Path("results/final_evaluation.json"),
    data_dir: Path = Path("data"),
    cache_dir: Path = Path("cache/final_evaluation"),
    train_sizes: tuple[int, ...] = _DEFAULT_TRAIN_SIZES,
    seeds: tuple[int, ...] = _DEFAULT_SEEDS,
    final_data: MNISTFinalData | None = None,
) -> dict[str, Any]:
    """Evaluate frozen finalists and raw/HOG baselines on the official test set."""
    finalists_path = Path(finalists_path)
    output_path = Path(output_path)
    selection, finalist_values = _load_frozen_finalists(finalists_path)
    protected_paths = {
        finalists_path.resolve(),
        _archive_path_for(finalists_path, selection["search_archive"]).resolve(),
    }
    if output_path.resolve() in protected_paths:
        raise ValueError("final evaluation output cannot overwrite selection inputs")
    if not train_sizes or len(set(train_sizes)) != len(train_sizes):
        raise ValueError("train_sizes must be a non-empty sequence of unique sizes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty sequence of unique values")
    if any(type(size) is not int or size <= 0 for size in train_sizes):
        raise ValueError("training sizes must be positive integers")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")

    test_evaluation_started_at = datetime.now(UTC).isoformat()
    data = final_data if final_data is not None else load_mnist_final_data(data_dir)
    if any(size > len(data.train_labels) for size in train_sizes):
        raise ValueError("training size cannot exceed the official train partition")

    extractor = BatchFeatureExtractor(FeatureCache(Path(cache_dir)))
    representations = _representations(finalist_values)
    results: list[dict[str, Any]] = []
    evaluation_started = perf_counter()
    for representation in representations:
        representation_hash = _representation_hash(representation.operator_specs)
        feature_started = perf_counter()
        train_features = extractor.transform(
            representation.pipeline,
            data.train_images,
            bias_hash=representation_hash,
            dataset_key="mnist_v1/final_evaluation/train_60000",
        )
        test_features = extractor.transform(
            representation.pipeline,
            data.test_images,
            bias_hash=representation_hash,
            dataset_key="mnist_v1/final_evaluation/test_10000",
        )
        feature_matrix_runtime_ms = (perf_counter() - feature_started) * 1_000

        runs_by_size: dict[str, list[dict[str, Any]]] = {}
        for size in train_sizes:
            size_runs = []
            for seed in seeds:
                indices = data.sample_training_indices(size, seed)
                learner = Learner(LearnerConfig(seed=seed))
                train_started = perf_counter()
                learner.fit(train_features[indices], data.train_labels[indices])
                training_runtime_ms = (perf_counter() - train_started) * 1_000
                inference_started = perf_counter()
                predictions = learner.predict(test_features)
                inference_runtime_ms = (perf_counter() - inference_started) * 1_000
                matrix = confusion_matrix(
                    data.test_labels,
                    predictions,
                    labels=_DIGIT_LABELS,
                )
                size_runs.append(
                    {
                        "seed": seed,
                        "accuracy": float(
                            accuracy_score(data.test_labels, predictions)
                        ),
                        "training_runtime_ms": training_runtime_ms,
                        "inference_runtime_ms": inference_runtime_ms,
                        "confusion_matrix": matrix.astype(np.int64).tolist(),
                    }
                )
            runs_by_size[str(size)] = size_runs

        results.append(
            {
                "name": representation.name,
                "kind": representation.kind,
                "candidate_id": representation.candidate_id,
                "selection_roles": list(representation.selection_roles),
                "operator_specs": [
                    spec.to_dict() for spec in representation.operator_specs
                ],
                "feature_dim": representation.pipeline.feature_dim,
                "feature_matrix_runtime_ms": feature_matrix_runtime_ms,
                "summaries": {
                    size: _summary(runs) for size, runs in runs_by_size.items()
                },
                "runs": runs_by_size,
            }
        )

    report = {
        "dataset": "OpenML mnist_784 version 1; official 60,000/10,000 train/test split",
        "test_set_accessed": True,
        "test_evaluation_started_at_utc": test_evaluation_started_at,
        "finalist_selection": {
            "path": str(finalists_path),
            "sha256": hashlib.sha256(finalists_path.read_bytes()).hexdigest(),
            "search_archive_sha256": selection["search_archive_sha256"],
            "frozen": selection["finalist_selection_frozen"],
            "finalist_count": len(finalist_values),
        },
        "training_sizes": list(train_sizes),
        "seeds": list(seeds),
        "learner": {
            "scaler": "StandardScaler",
            "classifier": "multinomial LogisticRegression",
            "regularization_c": 1.0,
            "max_iter": 1_000,
            "solver": "lbfgs",
        },
        "feature_cache_dir": str(cache_dir),
        "total_evaluation_runtime_ms": (perf_counter() - evaluation_started) * 1_000,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
