"""Run Phase D transfer for representations selected by the frozen A/B rule."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bias_optimizer.data.transfer import load_transfer_dataset
from bias_optimizer.evaluation.artifacts import append_jsonl, read_jsonl, write_json
from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.protocol import (
    EvaluationDataset,
    load_frozen_protocol,
)
from bias_optimizer.evaluation.representations import build_frozen_registry
from bias_optimizer.evaluation.runner import EvaluationRunner


def select_transfer_representations(root: Path) -> tuple[list[str], dict[str, object]]:
    """Apply the predeclared validation-only gate and return exactly two V3 names."""
    protocol, _ = load_frozen_protocol(root)
    core = read_jsonl(root / "results/v31/core_matrix.jsonl")
    complementarity_path = root / "results/v31/complementarity.json"
    if not complementarity_path.is_file():
        raise FileNotFoundError("Phase D requires the Phase B complementarity artifact")
    complementarity = json.loads(complementarity_path.read_text(encoding="utf-8"))
    names = protocol["phases"]["B_complementarity"]["v3_representations"]
    linear_5k: dict[str, list[float]] = defaultdict(list)
    linear_500: dict[str, list[float]] = defaultdict(list)
    zoning_500: list[float] = []
    for row in core:
        key = row["key"]
        if key["probe"] != "linear_logreg":
            continue
        if int(key["train_size"]) == 5000 and key["representation"] in names:
            linear_5k[key["representation"]].append(float(row["accuracy"]))
        if int(key["train_size"]) == 500 and key["representation"] in names:
            linear_500[key["representation"]].append(float(row["accuracy"]))
        if key["representation"] == "zoning_30d" and int(key["train_size"]) == 500:
            zoning_500.append(float(row["accuracy"]))
    if set(linear_5k) != set(names) or set(linear_500) != set(names) or len(zoning_500) != 3:
        raise ValueError("Phase A artifacts are incomplete for the frozen transfer gate")
    mean_v3 = {name: float(np.mean(scores)) for name, scores in linear_5k.items()}
    mean_v3_500 = {name: float(np.mean(scores)) for name, scores in linear_500.items()}
    mean_zoning = float(np.mean(zoning_500))

    comp_rows = complementarity["paired_runs"]
    selected_eligible = []
    gate = {}
    for name in names:
        v3_beats_zoning = mean_v3_500[name] > mean_zoning
        hog_positive_seeds = {
            int(row["seed"])
            for row in comp_rows
            if row["anchor"] == "hog"
            and row["augmented_representation"] == f"hog_plus_{name}"
            and row["probe"] == "linear_logreg"
            and int(row["train_size"]) in (500, 5000)
            and float(row["delta_complementarity"]) > 0
        }
        complementarity_pass = len(hog_positive_seeds) >= 2
        eligible = v3_beats_zoning or complementarity_pass
        gate[name] = {
            "mean_linear_accuracy_5000": mean_v3[name],
            "mean_linear_accuracy_500": mean_v3_500[name],
            "v3_beats_zoning30_at_500_mean": v3_beats_zoning,
            "positive_hog_complementarity_seeds": sorted(hog_positive_seeds),
            "eligible": eligible,
        }
        if eligible:
            selected_eligible.append(name)
    if len(selected_eligible) < 2:
        raise ValueError("fewer than two V3 representations pass the frozen Phase A/B gate")
    selected_v3 = sorted(selected_eligible, key=lambda name: (-mean_v3[name], name))[:2]
    controls = ["hog", "raw_pixels", "zoning_30d", "hog_pca_30d", "hog_pca_60d"]
    selection = {
        "rule": protocol["phases"]["D_transfer"]["selection_rule"],
        "zoning30_linear_accuracy_500_mean": mean_zoning,
        "v3_gate": gate,
        "selected_v3": selected_v3,
        "controls": controls,
    }
    return [*selected_v3, *controls], selection


def run_transfer(*, root: Path = Path("."), data_dir: Path | None = None) -> int:
    root = Path(root)
    protocol, manifest = load_frozen_protocol(root)
    phase = protocol["phases"]["D_transfer"]
    representation_names, selection = select_transfer_representations(root)
    registry = build_frozen_registry(root)
    runner = EvaluationRunner(registry, cache=FeatureMatrixCache(root / "cache/v31_features"))
    output = root / "results/v31/transfer_runs.jsonl"
    completed = {
        _run_key(row)
        for row in read_jsonl(output)
    }
    new_runs = 0
    data_root = Path(data_dir) if data_dir is not None else root / "data/transfer"
    provenance: dict[str, object] = {}
    for dataset_name in phase["datasets"]:
        transfer = load_transfer_dataset(dataset_name, data_dir=data_root)
        dataset = EvaluationDataset.from_transfer(
            transfer,
            test_sample_size=int(protocol["data"]["transfer_test_sample_size"]),
            test_sample_seed=int(protocol["data"]["transfer_test_sample_seed"]),
        )
        provenance[dataset_name] = {
            **dataset.provenance,
            "training_count": len(dataset.train_labels),
            "source_test_count": len(transfer.test_labels),
            "evaluation_count": len(dataset.evaluation_labels),
            "evaluation_partition": dataset.evaluation_partition,
            "evaluation_split_id": dataset.evaluation_split_id,
        }
        for representation in representation_names:
            for probe in phase["probes"]:
                for size in phase["train_sizes"]:
                    for seed in phase["seeds"]:
                        key = (dataset_name, representation, probe, int(size), int(seed))
                        if key in completed:
                            continue
                        result = runner.evaluate(
                            dataset=dataset,
                            representation=representation,
                            probe=probe,
                            train_size=int(size),
                            seed=int(seed),
                        )
                        row = result.to_dict()
                        row["phase"] = "D_transfer"
                        row["dataset_provenance"] = dataset.provenance
                        row["evaluation_partition"] = dataset.evaluation_partition
                        row["evaluation_split_id"] = dataset.evaluation_split_id
                        append_jsonl(output, row)
                        completed.add(key)
                        new_runs += 1
                        print(
                            f"Phase D: {dataset_name} {representation} × {probe} "
                            f"n={size} seed={seed} accuracy={result.accuracy:.4f}"
                        )

    rows = read_jsonl(output)
    write_json(
        root / "results/v31/transfer.json",
        {
            "schema_version": 1,
            "phase": "D_transfer",
            "protocol_sha256": manifest["protocol"]["sha256"],
            "selection": selection,
            "dataset_provenance": provenance,
            "runs": rows,
            "summaries": _summaries(rows),
        },
    )
    return new_runs


def _summaries(rows: list[dict]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = row["key"]
        groups[(key["dataset"], key["representation"], key["probe"], int(key["train_size"]))].append(row)
    result = []
    for (dataset, representation, probe, size), values in sorted(groups.items()):
        scores = [float(value["accuracy"]) for value in values]
        result.append({
            "dataset": dataset,
            "representation": representation,
            "probe": probe,
            "train_size": size,
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "seed_count": len(scores),
        })
    return result


def _run_key(row: dict) -> tuple[str, str, str, int, int]:
    key = row["key"]
    return key["dataset"], key["representation"], key["probe"], int(key["train_size"]), int(key["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    print(f"completed {run_transfer(root=args.root, data_dir=args.data_dir)} new transfer runs")


if __name__ == "__main__":
    main()
