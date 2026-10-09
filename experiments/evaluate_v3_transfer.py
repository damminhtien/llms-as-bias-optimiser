"""Compare frozen V3 programs with matched structural and classical baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.compiler.bias_compiler import BiasCompiler
from bias_optimizer.data.transfer import load_transfer_dataset
from bias_optimizer.domain.bias import bias_spec_hash
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.baselines import HOGOperator, RawPixelsOperator
from bias_optimizer.features.pipeline import BatchFeatureExtractor, FeaturePipeline
from experiments.evaluate_transfer import (
    _BASELINE_SPECS,
    _evaluate_representation,
    _read_frozen_manifest,
)
from experiments.evaluate_v3_compact_baseline import zoning_bias

_DATASETS = ("emnist_digits", "emnist_letters", "kmnist", "fashion_mnist")


def _representation_id(name: str) -> str:
    return hashlib.sha256(f"v3-transfer:{name}".encode()).hexdigest()


def _representations(
    finalists: list[dict[str, Any]],
    *,
    v2_manifest_path: Path,
) -> tuple[tuple[str, str, str, Any], ...]:
    program_compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    representations: list[tuple[str, str, str, Any]] = []
    for item in finalists:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        if program_bias_hash(bias) != item.get("candidate_id"):
            raise ValueError("V3 finalist hash does not match its frozen AST")
        representations.append(
            (
                str(item["candidate_id"]),
                str(item["name"]),
                "v3_discovered",
                program_compiler.compile(bias.program),
            )
        )

    legacy_compiler = BiasCompiler()
    for bias in _BASELINE_SPECS[:2]:
        representations.append(
            (
                bias_spec_hash(bias),
                bias.name,
                "human_structural_control",
                legacy_compiler.compile(bias),
            )
        )

    representations.extend(
        (
            (
                _representation_id("raw-pixels-784"),
                "raw_pixels",
                "classical_baseline",
                FeaturePipeline((RawPixelsOperator(),)),
            ),
            (
                _representation_id("hog-9bin-cell4-block2"),
                "hog_9bin_4x4",
                "classical_baseline",
                FeaturePipeline((HOGOperator(),)),
            ),
        )
    )

    for rows, columns in ((4, 4), (5, 6)):
        zoning_spec = zoning_bias(rows, columns)
        zoning_pipeline = program_compiler.compile(zoning_spec.program)
        if zoning_pipeline.feature_dim != rows * columns:
            raise ValueError(f"{zoning_spec.name} baseline has an unexpected width")
        representations.append(
            (
                program_bias_hash(zoning_spec),
                zoning_spec.name,
                "compact_classical_baseline",
                zoning_pipeline,
            )
        )

    v2_manifest, _ = _read_frozen_manifest(v2_manifest_path)
    v2_candidates = v2_manifest.get("finalists")
    if not isinstance(v2_candidates, list):
        raise TypeError("V2 transfer manifest has no finalists list")
    v2_best = next(
        (item for item in v2_candidates if item.get("name") == "spatial_cycle_hist"),
        None,
    )
    if v2_best is None:
        raise ValueError("V2 transfer manifest has no spatial_cycle_hist control")
    v2_bias = ProgramBiasSpec.from_dict(v2_best["bias"])
    representations.append(
        (
            str(v2_best["candidate_id"]),
            "spatial_cycle_hist_v2_best",
            "v2_best",
            program_compiler.compile(v2_bias.program),
        )
    )
    return tuple(representations)


def evaluate_v3_transfer(
    manifest_path: Path,
    v2_manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data/transfer"),
    cache_dir: Path = Path("cache/v3_transfer_features"),
    datasets: tuple[str, ...] = _DATASETS,
    train_sizes: tuple[int, ...] = (500, 5_000),
    train_seeds: tuple[int, ...] = (11, 23, 47),
    test_sample_size: int = 2_000,
    test_sample_seed: int = 31_415,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    v2_manifest_path = Path(v2_manifest_path)
    manifest, manifest_sha256 = _read_frozen_manifest(manifest_path)
    if not v2_manifest_path.exists():
        raise FileNotFoundError(f"V2 comparison manifest not found: {v2_manifest_path}")
    finalists = manifest["finalists"]
    representations = _representations(
        finalists,
        v2_manifest_path=v2_manifest_path,
    )
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

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "v3_selection_manifest": str(manifest_path),
            "v3_selection_manifest_sha256": manifest_sha256,
            "v3_search_archive_sha256": manifest["search_archive_sha256"],
            "v2_selection_manifest": str(v2_manifest_path),
            "v2_selection_manifest_sha256": hashlib.sha256(
                v2_manifest_path.read_bytes()
            ).hexdigest(),
            "mnist_test_accessed_for_selection": False,
            "representations_frozen_before_transfer_test": True,
            "classifier_refit_per_domain": True,
            "learner": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
            "train_sizes": list(train_sizes),
            "train_seeds": list(train_seeds),
            "transfer_test_sample_size": test_sample_size,
            "transfer_test_sample_seed": test_sample_seed,
            "test_sampling": "stratified without replacement from each official test split",
            "datasets": list(datasets),
            "baselines": [
                "raw_pixels",
                "hog_9bin_4x4",
                "human_topology",
                "human_topology_spatial",
                "zoning_4x4",
                "zoning_5x6",
                "spatial_cycle_hist_v2_best",
            ],
            "representation_count": len(representations),
        },
        "results": results,
        "claim_boundary": "Transfer scores use the stated stratified official-test subsets. They are not full-test-set scores and do not establish a universal handwriting prior.",
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
        "--manifest", type=Path, default=Path("results/v3_frozen_finalists.json")
    )
    parser.add_argument(
        "--v2-manifest", type=Path, default=Path("results/v2_transfer_finalists.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v3_transfer_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/transfer"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/v3_transfer_features")
    )
    parser.add_argument("--datasets", nargs="+", choices=_DATASETS, default=_DATASETS)
    parser.add_argument("--test-sample-size", type=int, default=2_000)
    args = parser.parse_args()
    evaluate_v3_transfer(
        args.manifest,
        args.v2_manifest,
        args.output,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        datasets=tuple(args.datasets),
        test_sample_size=args.test_sample_size,
    )
    print(f"Saved V3 transfer evaluation to {args.output}")


if __name__ == "__main__":
    main()
