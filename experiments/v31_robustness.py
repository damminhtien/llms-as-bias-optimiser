"""Run clean-trained robustness checks over the frozen V3 transformation set."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from bias_optimizer.data.mnist import (
    MNISTDataConfig,
    load_mnist_final_data,
    load_mnist_search_data,
)
from bias_optimizer.evaluation.artifacts import append_jsonl, read_jsonl, write_json
from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.protocol import EvaluationDataset, load_frozen_protocol
from bias_optimizer.evaluation.representations import build_frozen_registry
from bias_optimizer.evaluation.robustness import TRANSFORMATIONS, transform_images
from bias_optimizer.evaluation.runner import EvaluationRunner


def run_robustness(*, root: Path = Path(".")) -> int:
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
    runner = EvaluationRunner(
        build_frozen_registry(root),
        cache=FeatureMatrixCache(root / "cache/v31_features"),
    )
    output = root / "results/v31/robustness_runs.jsonl"
    completed = {
        (row["representation"], int(row["seed"]), row["transformation"])
        for row in read_jsonl(output)
    }
    new_runs = 0
    for name in phase["v3_representations"]:
        for seed in phase["seeds"]:
            model = runner.fit_model(
                dataset=dataset,
                representation=name,
                probe=phase["probe"],
                train_size=int(phase["train_size"]),
                seed=int(seed),
            )
            clean_predictions, _, _ = model.predict(dataset.evaluation_images)
            clean_accuracy = float(accuracy_score(dataset.evaluation_labels, clean_predictions))
            for transform_index, transformation in enumerate(TRANSFORMATIONS):
                key = (name, int(seed), transformation)
                if key in completed:
                    continue
                transformed_images = transform_images(
                    dataset.evaluation_images,
                    transformation,
                    seed=20_261_010 + transform_index,
                )
                predictions, _, _ = model.predict(transformed_images)
                accuracy = float(accuracy_score(dataset.evaluation_labels, predictions))
                append_jsonl(
                    output,
                    {
                        "representation": name,
                        "probe": phase["probe"],
                        "train_size": phase["train_size"],
                        "seed": seed,
                        "transformation": transformation,
                        "clean_accuracy": clean_accuracy,
                        "transformed_accuracy": accuracy,
                        "accuracy_drop": clean_accuracy - accuracy,
                        "prediction_agreement": float(np.mean(clean_predictions == predictions)),
                        "evaluation_split_id": dataset.evaluation_split_id,
                        "evaluation_indices": dataset.evaluation_indices.tolist(),
                        "targets": dataset.evaluation_labels.tolist(),
                        "clean_predictions": clean_predictions.tolist(),
                        "predictions": predictions.tolist(),
                    },
                )
                completed.add(key)
                new_runs += 1
                print(
                    f"Phase C robustness: {name} seed={seed} {transformation} "
                    f"accuracy={accuracy:.4f} agreement={np.mean(clean_predictions == predictions):.4f}"
                )

    summaries = _summaries(read_jsonl(output))
    write_json(
        root / "results/v31/robustness.json",
        {
            "schema_version": 1,
            "phase": "C_robustness",
            "evaluation_partition": dataset.evaluation_partition,
            "evaluation_split_id": dataset.evaluation_split_id,
            "runs": read_jsonl(output),
            "summaries": summaries,
        },
    )
    return new_runs


def _summaries(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["representation"], row["transformation"])].append(row)
    result = []
    for (representation, transformation), values in sorted(groups.items()):
        drops = [float(value["accuracy_drop"]) for value in values]
        agreements = [float(value["prediction_agreement"]) for value in values]
        result.append({
            "representation": representation,
            "transformation": transformation,
            "accuracy_drop_mean": float(np.mean(drops)),
            "accuracy_drop_sd": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            "prediction_agreement_mean": float(np.mean(agreements)),
            "seed_count": len(values),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(f"completed {run_robustness(root=args.root)} new robustness runs")


if __name__ == "__main__":
    main()
