"""Evaluate frozen raw-free programs on handwriting transfers and a control set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.compiler.bias_compiler import BiasCompiler
from bias_optimizer.data.transfer import TransferDataset, load_transfer_dataset
from bias_optimizer.domain.bias import BiasSpec, OperatorSpec, bias_spec_hash
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.pipeline import BatchFeatureExtractor
from bias_optimizer.ml.learner import Learner, LearnerConfig

_DATASET_NAMES = ("emnist_digits", "emnist_letters", "kmnist", "fashion_mnist")
_BASELINE_SPECS = (
    BiasSpec(
        "human_topology",
        "Handwritten class identity is partly encoded by connected and loop structure.",
        (OperatorSpec("topology"),),
        "Topology features should beat chance on handwriting transfers.",
        "Near-chance performance across handwriting sets rejects this control.",
    ),
    BiasSpec(
        "human_topology_spatial",
        "Topology combined with coarse ink position captures a compact digit prior.",
        (OperatorSpec("topology"), OperatorSpec("spatial")),
        "Adding spatial distribution should improve low-data handwriting transfer.",
        "No improvement over topology alone rejects this combination.",
    ),
    BiasSpec(
        "human_topology_curvature",
        "Topology and local turning geometry provide complementary stroke cues.",
        (OperatorSpec("topology"), OperatorSpec("curvature")),
        "Curvature should improve transfer on handwriting domains.",
        "No improvement over topology alone rejects this combination.",
    ),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_archive(manifest_path: Path, stored_path: str) -> Path:
    path = Path(stored_path)
    if path.is_absolute() or path.exists():
        return path
    fallback = manifest_path.parent.parent / path
    return fallback if fallback.exists() else path


def _read_frozen_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("transfer evaluation requires frozen finalists")
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("manifest does not attest to pre-test selection")
    archive_value = manifest.get("search_archive")
    expected_hash = manifest.get("search_archive_sha256")
    if not isinstance(archive_value, str) or not isinstance(expected_hash, str):
        raise TypeError("frozen manifest has no archive provenance")
    archive = _resolve_archive(path, archive_value)
    if _file_sha256(archive) != expected_hash:
        raise ValueError("candidate archive changed after finalist freeze")
    finalists = manifest.get("finalists")
    if not isinstance(finalists, list) or not finalists:
        raise ValueError("frozen manifest has no candidates")
    for item in finalists:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        if program_bias_hash(bias) != item.get("candidate_id"):
            raise ValueError("frozen candidate ID does not match its program")
    return manifest, _file_sha256(path)


def _sample_indices(labels: np.ndarray, size: int, *, seed: int) -> np.ndarray:
    if size > len(labels):
        raise ValueError(f"cannot sample {size} examples from {len(labels)} rows")
    indices = np.arange(len(labels))
    if size == len(labels):
        return indices
    selected, _, _, _ = train_test_split(
        indices,
        labels,
        train_size=size,
        random_state=seed,
        stratify=labels,
    )
    return np.asarray(selected, dtype=np.int64)


def _representations(
    finalists: list[dict[str, Any]],
) -> tuple[tuple[str, str, str, Any], ...]:
    program_compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    representations: list[tuple[str, str, str, Any]] = []
    for item in finalists:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        representations.append(
            (
                str(item["candidate_id"]),
                str(item["name"]),
                "discovered_program",
                program_compiler.compile(bias.program),
            )
        )
    legacy_compiler = BiasCompiler()
    for bias in _BASELINE_SPECS:
        representations.append(
            (
                bias_spec_hash(bias),
                bias.name,
                "human_structural_control",
                legacy_compiler.compile(bias),
            )
        )
    return tuple(representations)


def _feature_matrix(
    extractor: BatchFeatureExtractor,
    representation: tuple[str, str, str, Any],
    images: np.ndarray,
    dataset_key: str,
) -> np.ndarray:
    candidate_id, _, _, pipeline = representation
    return extractor.transform(
        pipeline,
        images,
        bias_hash=candidate_id,
        dataset_key=dataset_key,
    )


def _evaluate_representation(
    dataset: TransferDataset,
    representation: tuple[str, str, str, Any],
    *,
    extractor: BatchFeatureExtractor,
    train_sizes: tuple[int, ...],
    train_seeds: tuple[int, ...],
    test_sample_size: int,
    test_sample_seed: int,
    class_count: int,
) -> dict[str, Any]:
    dataset_key = f"transfer/{dataset.name}/{dataset.source_sha256}"
    test_indices = _sample_indices(
        dataset.test_labels, test_sample_size, seed=test_sample_seed
    )
    test_images = dataset.test_images[test_indices]
    test_labels = dataset.test_labels[test_indices]
    test_features = _feature_matrix(
        extractor,
        representation,
        test_images,
        f"{dataset_key}/official_test_sample/{test_sample_seed}/{test_sample_size}",
    )

    runs: list[dict[str, Any]] = []
    for train_size in train_sizes:
        for seed in train_seeds:
            indices = _sample_indices(dataset.train_labels, train_size, seed=seed)
            train_images = dataset.train_images[indices]
            train_labels = dataset.train_labels[indices]
            train_features = _feature_matrix(
                extractor,
                representation,
                train_images,
                f"{dataset_key}/train/{seed}/{train_size}",
            )
            learner = Learner(LearnerConfig(seed=seed))
            fit_start = perf_counter()
            learner.fit(train_features, train_labels)
            training_ms = (perf_counter() - fit_start) * 1_000
            predict_start = perf_counter()
            predictions = learner.predict(test_features)
            inference_ms = (perf_counter() - predict_start) * 1_000
            runs.append(
                {
                    "train_size": train_size,
                    "seed": seed,
                    "accuracy": float(accuracy_score(test_labels, predictions)),
                    "training_runtime_ms": training_ms,
                    "inference_runtime_ms": inference_ms,
                    "confusion_matrix": confusion_matrix(
                        test_labels,
                        predictions,
                        labels=np.arange(class_count),
                    ).tolist(),
                }
            )

    summaries: dict[str, Any] = {}
    for train_size in train_sizes:
        selected_runs = [run for run in runs if run["train_size"] == train_size]
        accuracies = np.asarray(
            [run["accuracy"] for run in selected_runs], dtype=np.float64
        )
        summaries[str(train_size)] = {
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std(ddof=1))
            if len(accuracies) > 1
            else 0.0,
            "test_standard_error_mean": float(
                np.mean(np.sqrt(accuracies * (1 - accuracies) / len(test_labels)))
            ),
        }
    candidate_id, name, kind, pipeline = representation
    return {
        "candidate_id": candidate_id,
        "name": name,
        "kind": kind,
        "feature_dim": pipeline.feature_dim,
        "runs": runs,
        "summaries": summaries,
    }


def evaluate_transfer(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data/transfer"),
    cache_dir: Path = Path("cache/transfer_features"),
    datasets: tuple[str, ...] = _DATASET_NAMES,
    train_sizes: tuple[int, ...] = (500, 5_000),
    train_seeds: tuple[int, ...] = (11, 23, 47),
    test_sample_size: int = 2_000,
    test_sample_seed: int = 31_415,
) -> dict[str, Any]:
    manifest, manifest_sha256 = _read_frozen_manifest(Path(manifest_path))
    finalists = manifest["finalists"]
    representations = _representations(finalists)
    extractor = BatchFeatureExtractor(
        FeatureCache(Path(cache_dir)), SubexpressionCache()
    )
    results: list[dict[str, Any]] = []
    for name in datasets:
        dataset = load_transfer_dataset(name, data_dir=data_dir)
        class_count = int(dataset.train_labels.max()) + 1
        for representation in representations:
            result = _evaluate_representation(
                dataset,
                representation,
                extractor=extractor,
                train_sizes=train_sizes,
                train_seeds=train_seeds,
                test_sample_size=test_sample_size,
                test_sample_seed=test_sample_seed,
                class_count=class_count,
            )
            results.append(
                {
                    "dataset": name,
                    "source": dataset.source,
                    "source_sha256": dataset.source_sha256,
                    "train_count": len(dataset.train_labels),
                    "source_test_count": len(dataset.test_labels),
                    "test_partition": dataset.test_partition,
                    "evaluation_test_count": test_sample_size,
                    "test_sample_seed": test_sample_seed,
                    "preprocessing": dataset.preprocessing,
                    **result,
                }
            )
            print(
                f"{name}: {representation[1]} "
                + ", ".join(
                    f"n={size} {result['summaries'][str(size)]['accuracy_mean']:.4f}"
                    for size in train_sizes
                )
            )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": manifest_sha256,
            "search_archive_sha256": manifest["search_archive_sha256"],
            "mnist_test_accessed_for_selection": False,
            "representation_frozen_before_transfer_test": True,
            "classifier_refit_per_domain": True,
            "learner": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
            "train_sizes": list(train_sizes),
            "train_seeds": list(train_seeds),
            "transfer_test_sample_size": test_sample_size,
            "transfer_test_sample_seed": test_sample_seed,
            "test_sampling": "stratified without replacement from each official test split",
            "datasets": list(datasets),
        },
        "subexpression_cache": extractor.subexpression_cache.metrics,
        "results": results,
        "claim_boundary": "Transfer results are measured on the specified stratified official-test subsets; they are not full-test-set scores or proof of a universal handwriting bias.",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/v2_transfer_finalists.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v2_transfer_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/transfer"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/transfer_features")
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=_DATASET_NAMES, default=_DATASET_NAMES
    )
    parser.add_argument("--test-sample-size", type=int, default=2_000)
    args = parser.parse_args()
    evaluate_transfer(
        args.manifest,
        args.output,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        datasets=tuple(args.datasets),
        test_sample_size=args.test_sample_size,
    )
    print(f"Saved transfer evaluation to {args.output}")


if __name__ == "__main__":
    main()
