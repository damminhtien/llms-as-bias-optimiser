"""Run the frozen V3.1 core matrix or its predeclared learning-curve extension."""

from __future__ import annotations

import argparse
from pathlib import Path

from bias_optimizer.data.mnist import MNISTDataConfig, load_mnist_search_data
from bias_optimizer.evaluation.artifacts import append_jsonl, read_jsonl
from bias_optimizer.evaluation.feature_cache import FeatureMatrixCache
from bias_optimizer.evaluation.protocol import (
    EvaluationDataset,
    load_frozen_protocol,
)
from bias_optimizer.evaluation.representations import build_frozen_registry
from bias_optimizer.evaluation.runner import EvaluationRunner


def run_core_matrix(
    *,
    root: Path = Path("."),
    smoke: bool = False,
    curve_extension: bool = False,
) -> int:
    root = Path(root)
    protocol, _ = load_frozen_protocol(root)
    data = load_mnist_search_data(MNISTDataConfig(data_dir=root / "data/mnist"))
    dataset = EvaluationDataset.from_mnist_search(data)
    registry = build_frozen_registry(root, include_all_v3=False)
    runner = EvaluationRunner(
        registry,
        cache=FeatureMatrixCache(root / "cache/v31_features"),
    )

    phase = protocol["phases"]["A_core"]
    if smoke:
        representation_names = ["angle_centroid_pairwise"]
        probe_names = ["linear_logreg", "rbf_svm"]
        sizes = [250, 500]
        seeds = [11]
        output = root / "results/v31/smoke_matrix.jsonl"
        phase_name = "A_smoke"
    elif curve_extension:
        representation_names = phase["representations"]
        probe_names = phase["probes"]
        sizes = protocol["phases"]["A_learning_curve_extension"]["additional_sizes"]
        seeds = phase["seeds"]
        output = root / "results/v31/core_matrix.jsonl"
        phase_name = "A_learning_curve_extension"
    else:
        representation_names = phase["representations"]
        probe_names = phase["probes"]
        sizes = phase["train_sizes"]
        seeds = phase["seeds"]
        output = root / "results/v31/core_matrix.jsonl"
        phase_name = "A_core"

    completed = {
        _run_key(row)
        for row in read_jsonl(output)
    }
    count = 0
    for representation in representation_names:
        for probe in probe_names:
            for size in sizes:
                for seed in seeds:
                    run_key = (representation, probe, size, seed)
                    if run_key in completed:
                        continue
                    result = runner.evaluate(
                        dataset=dataset,
                        representation=representation,
                        probe=probe,
                        train_size=size,
                        seed=seed,
                    )
                    row = result.to_dict()
                    row["phase"] = phase_name
                    row["evaluation_partition"] = dataset.evaluation_partition
                    row["dataset_fingerprint"] = dataset.fingerprint
                    append_jsonl(output, row)
                    completed.add(run_key)
                    count += 1
                    print(
                        f"{phase_name}: {representation} × {probe} "
                        f"n={size} seed={seed} accuracy={result.accuracy:.4f}"
                    )
    return count


def _run_key(row: dict) -> tuple[str, str, int, int]:
    key = row["key"]
    return key["representation"], key["probe"], int(key["train_size"]), int(key["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke", action="store_true", help="one representation, two probes, two sizes")
    group.add_argument("--curve-extension", action="store_true", help="run the frozen 100 and 2,500 sample points")
    args = parser.parse_args()
    count = run_core_matrix(root=args.root, smoke=args.smoke, curve_extension=args.curve_extension)
    print(f"completed {count} new runs")


if __name__ == "__main__":
    main()
