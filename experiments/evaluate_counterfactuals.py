"""Test ordered-sequence mechanisms by shuffling values before aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.data.mnist import load_mnist_final_data
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.pipeline import BatchFeatureExtractor
from bias_optimizer.ml.learner import Learner, LearnerConfig

_SEQUENCE_OPERATIONS = frozenset(
    {"angles", "edge_lengths", "delta", "delta_angle", "sign", "run_length_encode"}
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_candidates(
    path: Path, count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("counterfactual evaluation requires frozen candidates")
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("candidate manifest was not frozen before test access")
    archive = Path(manifest["search_archive"])
    if not archive.is_absolute() and not archive.exists():
        archive = path.parent.parent / archive
    if _sha256(archive) != manifest.get("search_archive_sha256"):
        raise ValueError("candidate archive changed after finalist freeze")
    candidates = manifest["finalists"][:count]
    for item in candidates:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        if program_bias_hash(bias) != item["candidate_id"]:
            raise ValueError("candidate hash does not match its program")
    return manifest, candidates


def _program_operations(program: Any) -> set[str]:
    result: set[str] = set()
    stack = [program]
    while stack:
        node = stack.pop()
        if node.op in _SEQUENCE_OPERATIONS:
            result.add(node.op)
        stack.extend(node.args)
    return result


def evaluate_counterfactuals(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    cache_dir: Path = Path("cache/counterfactual_features"),
    finalist_count: int = 5,
    train_size: int = 5_000,
    test_sample_size: int = 2_000,
    train_seed: int = 42,
    test_sample_seed: int = 27_182,
    shuffle_seeds: tuple[int, ...] = (11, 23, 47),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest, candidates = _frozen_candidates(manifest_path, finalist_count)
    archive_hash = str(manifest["search_archive_sha256"])

    data = load_mnist_final_data(data_dir)
    train_images, train_labels = data.sample_training_data(train_size, train_seed)
    test_indices = np.arange(len(data.test_labels))
    if test_sample_size < len(test_indices):
        from sklearn.model_selection import train_test_split

        test_indices, _, _, _ = train_test_split(
            test_indices,
            data.test_labels,
            train_size=test_sample_size,
            random_state=test_sample_seed,
            stratify=data.test_labels,
        )
    test_images = data.test_images[test_indices]
    test_labels = data.test_labels[test_indices]

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    extractor = BatchFeatureExtractor(
        FeatureCache(Path(cache_dir)), SubexpressionCache()
    )
    results: list[dict[str, Any]] = []
    for item in candidates:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        pipeline = compiler.compile(bias.program)
        operations = sorted(_program_operations(bias.program))
        if not operations:
            results.append(
                {
                    "candidate_id": item["candidate_id"],
                    "name": bias.name,
                    "status": "no_sequence_valued_node",
                    "interventions": [],
                }
            )
            continue

        train_features = extractor.transform(
            pipeline,
            train_images,
            bias_hash=program_bias_hash(bias),
            dataset_key=f"mnist_counterfactual/train/{train_seed}/{train_size}",
        )
        test_features = extractor.transform(
            pipeline,
            test_images,
            bias_hash=program_bias_hash(bias),
            dataset_key=f"mnist_counterfactual/test/{test_sample_seed}/{test_sample_size}",
        )
        learner = Learner(LearnerConfig(seed=train_seed))
        learner.fit(train_features, train_labels)
        baseline_start = perf_counter()
        original_predictions = learner.predict(test_features)
        baseline_ms = (perf_counter() - baseline_start) * 1_000
        original_accuracy = float(accuracy_score(test_labels, original_predictions))

        interventions: list[dict[str, Any]] = []
        for target_op in operations:
            for shuffle_seed in shuffle_seeds:
                start = perf_counter()
                counterfactual_features = np.stack(
                    [
                        pipeline.transform_with_sequence_shuffle(
                            image,
                            target_op=target_op,
                            seed=shuffle_seed + index,
                        )
                        for index, image in enumerate(test_images)
                    ]
                )
                predictions = learner.predict(counterfactual_features)
                elapsed_ms = (perf_counter() - start) * 1_000
                interventions.append(
                    {
                        "target_op": target_op,
                        "shuffle_seed": shuffle_seed,
                        "accuracy": float(accuracy_score(test_labels, predictions)),
                        "accuracy_delta": float(
                            accuracy_score(test_labels, predictions) - original_accuracy
                        ),
                        "prediction_agreement": float(
                            np.mean(predictions == original_predictions)
                        ),
                        "runtime_ms": elapsed_ms,
                    }
                )
        results.append(
            {
                "candidate_id": item["candidate_id"],
                "name": bias.name,
                "hypothesis": bias.hypothesis,
                "mechanism": bias.mechanism,
                "feature_dim": pipeline.feature_dim,
                "sequence_operations": operations,
                "baseline_accuracy": original_accuracy,
                "baseline_inference_runtime_ms": baseline_ms,
                "intervention_semantics": "shuffle values independently within each path sequence at the selected node; the sequence multiset is preserved and downstream nodes are recomputed",
                "interventions": interventions,
            }
        )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": _sha256(manifest_path),
            "search_archive_sha256": archive_hash,
            "mnist_official_test_accessed_after_freeze": True,
            "train_size": train_size,
            "train_seed": train_seed,
            "test_sample_size": len(test_labels),
            "test_sample_seed": test_sample_seed,
            "test_sampling": "stratified without replacement from MNIST official test split",
            "shuffle_seeds": list(shuffle_seeds),
            "learner": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
        },
        "results": results,
        "claim_boundary": "A lower score after an order-destroying intervention supports sequence dependence in the tested representation and sample; it does not establish a general causal mechanism by itself.",
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
        "--output", type=Path, default=Path("results/v2_counterfactual_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/counterfactual_features")
    )
    parser.add_argument("--finalist-count", type=int, default=5)
    parser.add_argument("--test-sample-size", type=int, default=2_000)
    args = parser.parse_args()
    report = evaluate_counterfactuals(
        args.manifest,
        args.output,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        finalist_count=args.finalist_count,
        test_sample_size=args.test_sample_size,
    )
    print(f"Saved counterfactual analysis to {args.output}")
    for result in report["results"]:
        print(
            f"{result['name']}: baseline accuracy={result.get('baseline_accuracy', float('nan')):.4f}; "
            f"interventions={len(result['interventions'])}"
        )


if __name__ == "__main__":
    main()
