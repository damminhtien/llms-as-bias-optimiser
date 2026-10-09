"""Search bounded image-transform programs and test orbit-mean pooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from bias_optimizer.data.mnist import MNISTDataConfig, load_mnist_search_data
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.invariance import OrbitPooledPipeline
from bias_optimizer.invariance.proposer import InvarianceProposer
from bias_optimizer.invariance.transform import TransformProgram
from bias_optimizer.ml.learner import Learner, LearnerConfig


def _read_frozen_candidate(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("invariance search requires a frozen representation")
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("representation must be frozen before test-set access")
    archive = Path(manifest["search_archive"])
    if not archive.is_absolute() and not archive.exists():
        archive = path.parent.parent / archive
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != manifest.get("search_archive_sha256"):
        raise ValueError("discovery archive changed after representation freeze")
    finalists = manifest.get("finalists")
    if not isinstance(finalists, list) or not finalists:
        raise ValueError("frozen manifest has no candidates")
    first = finalists[0]
    bias = ProgramBiasSpec.from_dict(first["bias"])
    if program_bias_hash(bias) != first.get("candidate_id"):
        raise ValueError("frozen candidate ID does not match its program")
    return manifest, first


def _features(pipeline: object, images: np.ndarray) -> np.ndarray:
    transform = pipeline.transform
    return np.stack([transform(image) for image in images])


def search_invariances(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    proposal_count: int = 4,
    evaluation_size: int = 500,
    train_size: int = 5_000,
    seed: int = 42,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    manifest, finalist = _read_frozen_candidate(manifest_path)
    bias = ProgramBiasSpec.from_dict(finalist["bias"])
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    representation = compiler.compile(bias.program)
    data = load_mnist_search_data(MNISTDataConfig(data_dir=data_dir))
    train_images, train_labels = data.sample_training_data(train_size, seed=seed)
    selection_indices, holdout_indices, _, _ = train_test_split(
        np.arange(len(data.validation_labels)),
        data.validation_labels,
        test_size=0.5,
        random_state=seed,
        stratify=data.validation_labels,
    )
    if evaluation_size > len(selection_indices):
        raise ValueError("evaluation_size exceeds the invariance-selection split")
    selected_indices, _, _, _ = train_test_split(
        selection_indices,
        data.validation_labels[selection_indices],
        train_size=evaluation_size,
        random_state=seed + 1,
        stratify=data.validation_labels[selection_indices],
    )
    selection_images = data.validation_images[selected_indices]
    selection_labels = data.validation_labels[selected_indices]
    holdout_images = data.validation_images[holdout_indices]
    holdout_labels = data.validation_labels[holdout_indices]

    train_features = _features(representation, train_images)
    selection_original = _features(representation, selection_images)
    learner = Learner(LearnerConfig(seed=seed))
    learner.fit(train_features, train_labels)
    original_predictions = learner.predict(selection_original)
    original_accuracy = float(accuracy_score(selection_labels, original_predictions))

    proposer = InvarianceProposer()
    proposals = proposer.propose(
        count=proposal_count,
        representation_name=bias.name,
    )
    candidate_results: list[dict[str, object]] = []
    selected_programs: list[tuple[float, float, int, TransformProgram]] = []
    for proposal in proposals:
        transformed = np.stack(
            [
                proposal.spec.transform.apply(image, seed=seed + index)
                for index, image in enumerate(selection_images)
            ]
        )
        transformed_features = _features(representation, transformed)
        predictions = learner.predict(transformed_features)
        accuracy = float(accuracy_score(selection_labels, predictions))
        agreement = float(np.mean(predictions == original_predictions))
        change = np.abs(transformed - selection_images)
        changed_fraction = float(np.mean(change > 0.01))
        accuracy_drop = original_accuracy - accuracy
        passes = accuracy_drop <= 0.02 and agreement >= 0.95 and changed_fraction > 0
        candidate_results.append(
            {
                "name": proposal.spec.name,
                "hypothesis": proposal.spec.hypothesis,
                "mechanism": proposal.spec.mechanism,
                "transform": proposal.spec.transform.to_dict(),
                "transform_sha256": hashlib.sha256(
                    proposal.spec.transform.to_json().encode("utf-8")
                ).hexdigest(),
                "prediction": proposal.spec.prediction,
                "falsification": proposal.spec.falsification,
                "label_accuracy": accuracy,
                "label_accuracy_delta": accuracy - original_accuracy,
                "prediction_agreement": agreement,
                "changed_pixel_fraction": changed_fraction,
                "passes_selection_rule": passes,
                "model": proposal.model,
            }
        )
        if passes:
            selected_programs.append(
                (
                    accuracy,
                    agreement,
                    -len(proposal.spec.transform.steps),
                    proposal.spec.transform,
                )
            )

    orbit_result: dict[str, object] | None = None
    if selected_programs:
        _, _, _, chosen_transform = max(selected_programs, key=lambda item: item[:3])
        orbit_pipeline = OrbitPooledPipeline(
            representation,
            chosen_transform,
            seeds=(11, 23, 47),
        )
        baseline_holdout = _features(representation, holdout_images)
        pooled_train = _features(orbit_pipeline, train_images)
        pooled_holdout = _features(orbit_pipeline, holdout_images)
        baseline_learner = Learner(LearnerConfig(seed=seed))
        baseline_learner.fit(train_features, train_labels)
        pooled_learner = Learner(LearnerConfig(seed=seed))
        pooled_learner.fit(pooled_train, train_labels)
        baseline_accuracy = float(
            accuracy_score(holdout_labels, baseline_learner.predict(baseline_holdout))
        )
        pooled_accuracy = float(
            accuracy_score(holdout_labels, pooled_learner.predict(pooled_holdout))
        )
        orbit_result = {
            "transform": chosen_transform.to_dict(),
            "holdout_size": len(holdout_labels),
            "baseline_accuracy": baseline_accuracy,
            "orbit_mean_accuracy": pooled_accuracy,
            "accuracy_delta": pooled_accuracy - baseline_accuracy,
            "aggregation": "mean feature vector over the identity view and three deterministic transformed views",
            "selection_and_holdout_share_mnist_validation_source": True,
        }

    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            "search_archive_sha256": manifest["search_archive_sha256"],
            "representation_candidate_id": finalist["candidate_id"],
            "representation_frozen_before_invariance_search": True,
            "official_test_accessed": False,
            "train_size": train_size,
            "train_seed": seed,
            "selection_size": len(selection_labels),
            "holdout_size": len(holdout_labels),
            "classifier": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
            "selection_rule": "label accuracy drop <= 0.02 versus the untransformed validation view, prediction agreement >= 0.95, and at least one changed pixel",
        },
        "baseline_selection_accuracy": original_accuracy,
        "proposals": candidate_results,
        "orbit_pooling": orbit_result,
        "claim_boundary": "This bounded validation experiment measures candidate identity preservation and one derived orbit-mean representation; it does not establish an invariance for every writing style or domain.",
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
        "--output", type=Path, default=Path("results/v2_invariance_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument("--evaluation-size", type=int, default=500)
    args = parser.parse_args()
    report = search_invariances(
        args.manifest,
        args.output,
        data_dir=args.data_dir,
        proposal_count=args.proposals,
        evaluation_size=args.evaluation_size,
    )
    print(f"Saved invariance search to {args.output}")
    for proposal in report["proposals"]:
        print(
            f"{proposal['name']}: accuracy delta {proposal['label_accuracy_delta']:+.4f}; "
            f"agreement {proposal['prediction_agreement']:.4f}; "
            f"passes={proposal['passes_selection_rule']}"
        )


if __name__ == "__main__":
    main()
