"""Test frozen V3 finalists with mechanism-matched counterfactuals."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bias_optimizer.cache.feature_cache import FeatureCache
from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.data.mnist import load_mnist_final_data
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.pipeline import BatchFeatureExtractor
from bias_optimizer.ml.learner import Learner, LearnerConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(program: Expr):
    yield program
    for child in program.args:
        yield from _walk(child)


def _intervention_plan(program: Expr, mechanism: str) -> dict[str, str]:
    nodes = tuple(_walk(program))
    operations = {node.op for node in nodes}
    if (
        mechanism
        in {
            "graph and location interaction",
            "cross-variable relation",
        }
        and "cross_histogram" in operations
    ):
        centroid_node = next(
            (
                node
                for node in nodes
                if node.op == "path_summary"
                and node.params.get("measure") in {"centroid_x", "centroid_y"}
            ),
            None,
        )
        if centroid_node is not None:
            measure = str(centroid_node.params["measure"])
            return {
                "kind": "centroid_perturbation",
                "target_op": "path_summary",
                "target_measure": measure,
                "preserved": "path count, loop-flag multiset, and centroid marginal when multiple paths exist",
                "destroyed": "assignment between path centroid and graph summary",
            }
        return {
            "kind": "cross_pair_shuffle",
            "target_op": "cross_histogram",
            "preserved": "both input value marginals and event locations",
            "destroyed": "pairing between the two relational variables",
        }
    if mechanism == "spatial conditioning":
        node = next((node for node in nodes if node.op == "spatial_condition"), None)
        if node is not None and node.args:
            return {
                "kind": "spatial_reassignment",
                "target_op": node.args[0].op,
                "preserved": "event-value multiset and location multiset",
                "destroyed": "event-to-location assignment",
            }
    if mechanism == "ordered stroke or turn persistence":
        for op in ("autocorrelation", "run_length_encode"):
            if op not in operations:
                continue
            target = next(
                (node.args[0].op for node in nodes if node.op == op and node.args),
                None,
            )
            if target is not None:
                return {
                    "kind": "sequence_order",
                    "target_op": target,
                    "preserved": "per-path event-value multisets and event locations",
                    "destroyed": "within-path event order",
                }
    if mechanism == "topological structure":
        target = "skeletonize" if "skeletonize" in operations else "threshold"
        if target in operations:
            return {
                "kind": "topology_rewire",
                "target_op": target,
                "preserved": "foreground pixel count",
                "destroyed": "connectivity or cycle rank",
            }
    return {
        "kind": "feature_identity_permutation",
        "target_op": "representation_output",
        "preserved": "exact marginal distribution of whole feature vectors",
        "destroyed": "assignment of the proposed representation to each image",
    }


def _apply_intervention(
    pipeline: Any,
    image: np.ndarray,
    plan: dict[str, str],
    *,
    seed: int,
) -> np.ndarray:
    kind = plan["kind"]
    if kind == "sequence_order":
        return pipeline.transform_with_sequence_shuffle(
            image, target_op=plan["target_op"], seed=seed
        )
    if kind == "spatial_reassignment":
        return pipeline.transform_with_spatial_reassignment(
            image, target_op=plan["target_op"], seed=seed
        )
    if kind == "cross_pair_shuffle":
        return pipeline.transform_with_cross_pair_shuffle(
            image, target_op=plan["target_op"], seed=seed
        )
    if kind == "centroid_perturbation":
        return pipeline.transform_with_centroid_perturbation(
            image,
            target_measure=plan["target_measure"],
            seed=seed,
        )
    if kind == "topology_rewire":
        return pipeline.transform_with_topology_rewire(
            image, target_op=plan["target_op"], seed=seed
        )
    raise ValueError(f"unsupported per-image intervention {kind!r}")


def evaluate_v3_counterfactuals(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    cache_dir: Path = Path("cache/v3_counterfactual_features"),
    finalist_count: int = 5,
    train_size: int = 5_000,
    test_sample_size: int = 2_000,
    train_seed: int = 42,
    test_sample_seed: int = 27_182,
    intervention_seeds: tuple[int, ...] = (11, 23, 47),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("counterfactual evaluation requires frozen finalists")
    archive_path = Path(manifest["search_archive"])
    if not archive_path.is_absolute() and not archive_path.exists():
        archive_path = manifest_path.parent.parent / archive_path
    archive_hash = _sha256(archive_path)
    if archive_hash != manifest.get("search_archive_sha256"):
        raise ValueError("search archive changed after finalists were frozen")
    finalists = manifest.get("finalists")
    if not isinstance(finalists, list) or not finalists:
        raise ValueError("frozen manifest has no finalists")
    finalists = finalists[:finalist_count]

    data = load_mnist_final_data(data_dir)
    train_images, train_labels = data.sample_training_data(train_size, train_seed)
    test_indices = np.arange(len(data.test_labels))
    if test_sample_size < len(test_indices):
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
    results = []
    for finalist in finalists:
        bias = ProgramBiasSpec.from_dict(finalist["bias"])
        if program_bias_hash(bias) != finalist.get("candidate_id"):
            raise ValueError("finalist hash does not match its frozen AST")
        pipeline = compiler.compile(bias.program)
        plan = _intervention_plan(
            bias.program, finalist["expected_counterfactual"]["mechanism"]
        )
        train_features = extractor.transform(
            pipeline,
            train_images,
            bias_hash=program_bias_hash(bias),
            dataset_key=f"mnist_v3_counterfactual/train/{train_seed}/{train_size}",
        )
        test_features = extractor.transform(
            pipeline,
            test_images,
            bias_hash=program_bias_hash(bias),
            dataset_key=f"mnist_v3_counterfactual/test/{test_sample_seed}/{test_sample_size}",
        )
        learner = Learner(LearnerConfig(seed=train_seed))
        learner.fit(train_features, train_labels)
        original_predictions = learner.predict(test_features)
        baseline_accuracy = float(accuracy_score(test_labels, original_predictions))

        intervention_results = []
        for seed in intervention_seeds:
            failures = 0
            changed = 0
            if plan["kind"] == "feature_identity_permutation":
                permutation = np.random.default_rng(seed).permutation(
                    len(test_features)
                )
                intervened_features = test_features[permutation]
                applied = len(test_features)
                changed = int(
                    np.count_nonzero(
                        np.any(intervened_features != test_features, axis=1)
                    )
                )
            else:
                rows = []
                for index, (image, original) in enumerate(
                    zip(test_images, test_features, strict=True)
                ):
                    try:
                        transformed = _apply_intervention(
                            pipeline,
                            image,
                            plan,
                            seed=seed + index,
                        )
                        rows.append(transformed)
                        changed += int(not np.array_equal(transformed, original))
                    except ValueError:
                        rows.append(original)
                        failures += 1
                intervened_features = np.stack(rows)
                applied = len(test_features) - failures

            predictions = learner.predict(intervened_features)
            accuracy = float(accuracy_score(test_labels, predictions))
            intervention_results.append(
                {
                    "seed": seed,
                    "accuracy": accuracy,
                    "accuracy_delta": accuracy - baseline_accuracy,
                    "prediction_agreement": float(
                        np.mean(predictions == original_predictions)
                    ),
                    "applied_sample_count": applied,
                    "changed_feature_sample_count": changed,
                    "failed_sample_count": failures,
                }
            )
        results.append(
            {
                "candidate_id": finalist["candidate_id"],
                "name": bias.name,
                "descriptor": finalist["descriptor"],
                "mechanism": bias.mechanism,
                "feature_dim": pipeline.feature_dim,
                "baseline_accuracy": baseline_accuracy,
                "intervention": plan,
                "intervention_results": intervention_results,
            }
        )

    report: dict[str, Any] = {
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
            "intervention_seeds": list(intervention_seeds),
            "learner": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
        },
        "results": results,
        "claim_boundary": "Interventions preserve the stated nuisance marginals and test the frozen representation on the sampled official test set; they support, but do not alone prove, the proposed mechanism.",
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
        "--output", type=Path, default=Path("results/v3_counterfactual_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/v3_counterfactual_features")
    )
    parser.add_argument("--finalist-count", type=int, default=5)
    parser.add_argument("--test-sample-size", type=int, default=2_000)
    args = parser.parse_args()
    report = evaluate_v3_counterfactuals(
        args.manifest,
        args.output,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        finalist_count=args.finalist_count,
        test_sample_size=args.test_sample_size,
    )
    print(
        json.dumps(
            {"results": len(report["results"]), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
