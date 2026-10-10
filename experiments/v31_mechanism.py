"""Run Phase C mechanism destroyers against a training-calibrated nuisance control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from bias_optimizer.data.mnist import (
    MNISTDataConfig,
    load_mnist_final_data,
    load_mnist_search_data,
)
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.evaluation.artifacts import append_jsonl, read_jsonl, write_json
from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.protocol import EvaluationDataset, load_frozen_protocol
from bias_optimizer.evaluation.representations import (
    ProgramRepresentation,
    build_frozen_registry,
)
from bias_optimizer.evaluation.robustness import (
    calibrate_noise_sigma,
    gaussian_noise_images,
    spatial_reassignment_features,
)
from bias_optimizer.evaluation.runner import EvaluationRunner
from bias_optimizer.evaluation.statistics import paired_bootstrap_delta


def run_mechanism(*, root: Path = Path(".")) -> dict:
    root = Path(root)
    protocol, _ = load_frozen_protocol(root)
    phase = protocol["phases"]["C_mechanism_robustness"]
    search = load_mnist_search_data(MNISTDataConfig(data_dir=root / "data/mnist"))
    final = load_mnist_final_data(root / "data/mnist")
    dataset = EvaluationDataset.from_mnist_official_test(
        search,
        final,
        test_sample_size=int(phase["test_sample_size"]),
        test_sample_seed=int(phase["test_sample_seed"]),
    )
    registry = build_frozen_registry(root)
    runner = EvaluationRunner(registry, cache=FeatureMatrixCache(root / "cache/v31_features"))
    output_path = root / "results/v31/mechanism_runs.jsonl"
    rows = read_jsonl(output_path)
    completed = {(row["representation"], int(row["seed"])) for row in rows}
    for name in phase["v3_representations"]:
        for seed in phase["seeds"]:
            if (name, int(seed)) in completed:
                continue
            model = runner.fit_model(
                dataset=dataset,
                representation=name,
                probe=phase["probe"],
                train_size=int(phase["train_size"]),
                seed=int(seed),
            )
            if not isinstance(model.representation, ProgramRepresentation):
                raise TypeError("mechanism intervention requires a frozen V3 program")
            target_op = _spatial_target(model.representation.pipeline.program)
            training_images = dataset.train_images[model.train_indices]
            target_train = spatial_reassignment_features(
                model.representation,
                training_images,
                target_op=target_op,
                seed=int(seed) * 100_000,
            )
            calibration = calibrate_noise_sigma(
                model.representation,
                training_images,
                target_features=target_train,
                seed=int(seed),
            )

            clean_predictions, _, _ = model.predict(dataset.evaluation_images)
            target_features = spatial_reassignment_features(
                model.representation,
                dataset.evaluation_images,
                target_op=target_op,
                seed=int(phase["test_sample_seed"]),
            )
            target_predictions, _ = model.predict_features(target_features)
            nuisance_images = gaussian_noise_images(
                dataset.evaluation_images,
                calibration["sigma"],
                seed=int(phase["test_sample_seed"]),
            )
            nuisance_predictions, _, _ = model.predict(nuisance_images)
            targets = dataset.evaluation_labels
            clean_accuracy = float(accuracy_score(targets, clean_predictions))
            target_accuracy = float(accuracy_score(targets, target_predictions))
            nuisance_accuracy = float(accuracy_score(targets, nuisance_predictions))
            target_rms = _rms_displacement(
                model.representation.transform(dataset.evaluation_images), target_features
            )
            nuisance_features = model.representation.transform(nuisance_images)
            nuisance_rms = _rms_displacement(
                model.representation.transform(dataset.evaluation_images), nuisance_features
            )
            row = {
                "representation": name,
                "probe": phase["probe"],
                "train_size": phase["train_size"],
                "seed": seed,
                "target_op": target_op,
                "noise_sigma": calibration["sigma"],
                "noise_calibration_example_count": len(training_images),
                "calibration_target_feature_rms": calibration["target_feature_rms"],
                "test_target_feature_rms": target_rms,
                "test_nuisance_feature_rms": nuisance_rms,
                "clean_accuracy": clean_accuracy,
                "target_accuracy": target_accuracy,
                "nuisance_accuracy": nuisance_accuracy,
                "target_drop": clean_accuracy - target_accuracy,
                "nuisance_drop": clean_accuracy - nuisance_accuracy,
                "target_prediction_agreement": float(np.mean(clean_predictions == target_predictions)),
                "nuisance_prediction_agreement": float(np.mean(clean_predictions == nuisance_predictions)),
                "evaluation_partition": dataset.evaluation_partition,
                "evaluation_split_id": dataset.evaluation_split_id,
                "evaluation_indices": dataset.evaluation_indices.tolist(),
                "targets": targets.tolist(),
                "clean_predictions": clean_predictions.tolist(),
                "target_predictions": target_predictions.tolist(),
                "nuisance_predictions": nuisance_predictions.tolist(),
            }
            row["target_bootstrap"] = paired_bootstrap_delta(
                targets, clean_predictions, target_predictions
            )
            row["nuisance_bootstrap"] = paired_bootstrap_delta(
                targets, clean_predictions, nuisance_predictions
            )
            append_jsonl(output_path, row)
            rows.append(row)
            completed.add((name, int(seed)))
            print(
                f"Phase C mechanism: {name} seed={seed} "
                f"target drop={row['target_drop']:.4f}, nuisance drop={row['nuisance_drop']:.4f}, "
                f"sigma={calibration['sigma']:.3f}"
            )
    summaries = []
    for name in phase["v3_representations"]:
        per_seed = [row for row in rows if row["representation"] == name]
        target_drops = [float(row["target_drop"]) for row in per_seed]
        nuisance_drops = [float(row["nuisance_drop"]) for row in per_seed]
        if len(per_seed) != len(phase["seeds"]):
            raise ValueError(f"mechanism results are incomplete for {name}")
        summaries.append({
            "representation": name,
            "target_drop_mean": float(np.mean(target_drops)),
            "target_drop_sd": float(np.std(target_drops, ddof=1)),
            "nuisance_drop_mean": float(np.mean(nuisance_drops)),
            "nuisance_drop_sd": float(np.std(nuisance_drops, ddof=1)),
            "target_minus_nuisance_drop_mean": float(np.mean(np.asarray(target_drops) - np.asarray(nuisance_drops))),
        })

    output = {
        "schema_version": 1,
        "phase": "C_mechanism",
        "protocol_sha256": load_frozen_protocol(root)[1]["protocol"]["sha256"],
        "dataset_provenance": dataset.provenance,
        "test_sample_seed": phase["test_sample_seed"],
        "runs": rows,
        "summaries": summaries,
    }
    write_json(root / "results/v31/mechanism.json", output)
    return output


def _spatial_target(program: Expr) -> str:
    stack = [program]
    while stack:
        node = stack.pop()
        if node.op == "spatial_condition" and node.args:
            return node.args[0].op
        stack.extend(reversed(node.args))
    raise ValueError("frozen V3 program has no spatial_condition operation")


def _rms_displacement(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left, dtype=np.float64) - right))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_mechanism(root=args.root)
    print(f"completed {len(result['runs'])} mechanism runs")


if __name__ == "__main__":
    main()
