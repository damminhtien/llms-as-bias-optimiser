"""Run Phase B: paired HOG/raw complementarity for the top frozen V3 biases."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from bias_optimizer.data.mnist import MNISTDataConfig, load_mnist_search_data
from bias_optimizer.evaluation.artifacts import append_jsonl, read_jsonl, write_json
from bias_optimizer.evaluation.complementarity import paired_complementarity
from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.protocol import EvaluationDataset, load_frozen_protocol
from bias_optimizer.evaluation.representations import (
    build_concatenated_representation,
    build_frozen_registry,
)
from bias_optimizer.evaluation.runner import EvaluationRunner
from bias_optimizer.evaluation.schema import EvaluationResult
from bias_optimizer.evaluation.statistics import mcnemar_exact, paired_bootstrap_delta


def run_complementarity(*, root: Path = Path(".")) -> int:
    root = Path(root)
    protocol, _ = load_frozen_protocol(root)
    data = load_mnist_search_data(MNISTDataConfig(data_dir=root / "data/mnist"))
    dataset = EvaluationDataset.from_mnist_search(data)
    registry = build_frozen_registry(root)
    runner = EvaluationRunner(registry, cache=FeatureMatrixCache(root / "cache/v31_features"))
    core_path = root / "results/v31/core_matrix.jsonl"
    core_rows = read_jsonl(core_path)
    core_results = [EvaluationResult.from_dict(row) for row in core_rows]

    output = root / "results/v31/complementarity_runs.jsonl"
    completed = {
        _run_key(row)
        for row in read_jsonl(output)
    }
    added_runs = 0
    for anchor in ("hog", "raw_pixels"):
        for v3_name in protocol["phases"]["B_complementarity"]["v3_representations"]:
            definition = build_concatenated_representation(registry, anchor, v3_name)
            runner.representations[definition.name] = definition
            for probe in protocol["phases"]["B_complementarity"]["probes"]:
                for size in protocol["phases"]["B_complementarity"]["train_sizes"]:
                    for seed in protocol["data"]["training_seeds"]:
                        run_key = (definition.name, probe, int(size), int(seed))
                        if run_key in completed:
                            continue
                        result = runner.evaluate(
                            dataset=dataset,
                            representation=definition.name,
                            probe=probe,
                            train_size=int(size),
                            seed=int(seed),
                        )
                        row = result.to_dict()
                        row["phase"] = "B_complementarity"
                        append_jsonl(output, row)
                        completed.add(run_key)
                        added_runs += 1
                        print(
                            f"Phase B: {definition.name} × {probe} "
                            f"n={size} seed={seed} accuracy={result.accuracy:.4f}"
                        )

    augmented_results = [
        EvaluationResult.from_dict(row) for row in read_jsonl(output)
    ]
    pair_rows: list[dict[str, object]] = []
    paired_statistics: dict[str, object] = {}
    for anchor in ("hog", "raw_pixels"):
        for v3_name in protocol["phases"]["B_complementarity"]["v3_representations"]:
            combo_name = f"{anchor}_plus_{v3_name}"
            for probe in protocol["phases"]["B_complementarity"]["probes"]:
                for size in protocol["phases"]["B_complementarity"]["train_sizes"]:
                    baseline = [r for r in core_results if r.key.representation == anchor and r.key.probe == probe and r.key.train_size == size]
                    augmented = [r for r in augmented_results if r.key.representation == combo_name and r.key.probe == probe and r.key.train_size == size]
                    rows = paired_complementarity(baseline, augmented)
                    pair_rows.extend(rows)
                    group = f"{anchor}+{v3_name}/{probe}/n={size}"
                    base_by_seed = {r.key.seed: r for r in baseline}
                    combo_by_seed = {r.key.seed: r for r in augmented}
                    ordered = sorted(set(base_by_seed) & set(combo_by_seed))
                    if len(ordered) != 3:
                        raise ValueError(f"expected all three paired seeds for {group}")
                    first = base_by_seed[ordered[0]]
                    base_predictions = np.stack([base_by_seed[seed].predictions for seed in ordered])
                    combo_predictions = np.stack([combo_by_seed[seed].predictions for seed in ordered])
                    if not all(np.array_equal(first.targets, base_by_seed[seed].targets) for seed in ordered):
                        raise ValueError(f"baseline targets differ across seeds for {group}")
                    paired_statistics[group] = {
                        "bootstrap": paired_bootstrap_delta(
                            first.targets, base_predictions, combo_predictions
                        ),
                        "mcnemar_by_seed": {
                            str(seed): mcnemar_exact(
                                first.targets,
                                base_by_seed[seed].predictions,
                                combo_by_seed[seed].predictions,
                            )
                            for seed in ordered
                        },
                    }

    summaries = _summaries(pair_rows)
    write_json(
        root / "results/v31/complementarity.json",
        {
            "schema_version": 1,
            "phase": "B_complementarity",
            "runs_added_this_call": added_runs,
            "paired_runs": pair_rows,
            "summaries": summaries,
            "paired_statistics": paired_statistics,
        },
    )
    return added_runs


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (
            row["anchor"],
            str(row["augmented_representation"]).removeprefix(f"{row['anchor']}_plus_"),
            row["probe"],
            row["train_size"],
        )
        groups[key].append(float(row["delta_complementarity"]))
    result = []
    for (anchor, added, probe, size), deltas in sorted(groups.items()):
        result.append({
            "anchor": anchor,
            "v3_representation": added,
            "train_size": size,
            "probe": probe,
            "delta_mean": float(np.mean(deltas)),
            "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "seed_count": len(deltas),
        })
    return result


def _run_key(row: dict) -> tuple[str, str, int, int]:
    key = row["key"]
    return key["representation"], key["probe"], int(key["train_size"]), int(key["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(f"completed {run_complementarity(root=args.root)} new runs")


if __name__ == "__main__":
    main()
