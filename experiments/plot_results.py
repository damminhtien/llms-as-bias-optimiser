"""Create publication-ready plots from the frozen search and test reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from bias_optimizer.domain.search import SearchRecord
from bias_optimizer.search.archive import SearchArchive


def _operator_key(result: dict[str, Any]) -> str:
    return json.dumps(result["operator_specs"], sort_keys=True, separators=(",", ":"))


def _unique_representations(report: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in report["results"]:
        groups[_operator_key(result)].append(result)

    unique = []
    for values in groups.values():
        representative = dict(values[0])
        if len(values) > 1:
            representative["name"] = "raw pixels (candidate and baseline)"
        unique.append(representative)
    return unique


def _plot_learning_curve(report: dict[str, Any], output: Path) -> None:
    train_sizes = report["training_sizes"]
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for result in _unique_representations(report):
        means = [
            result["summaries"][str(size)]["accuracy_mean"] for size in train_sizes
        ]
        deviations = [
            result["summaries"][str(size)]["accuracy_std"] for size in train_sizes
        ]
        baseline = result["kind"] == "baseline"
        axis.errorbar(
            train_sizes,
            means,
            yerr=deviations,
            label=result["name"],
            marker="s" if baseline else "o",
            linestyle="--" if baseline else "-",
            linewidth=1.6,
            markersize=4,
            capsize=2,
        )
    axis.set_xscale("log")
    axis.set_xticks(train_sizes, labels=[f"{size:,}" for size in train_sizes])
    axis.set_ylim(0.1, 1.0)
    axis.set_xlabel("Training examples (log scale)")
    axis.set_ylabel("Official test accuracy (mean ± SD, 3 seeds)")
    axis.set_title("MNIST learning curves after finalist selection")
    axis.grid(True, which="both", alpha=0.25)
    figure.subplots_adjust(right=0.72)
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout(rect=(0, 0, 0.72, 1))
    figure.savefig(output / "learning_curve.png", dpi=180)
    plt.close(figure)


def _plot_bias_evolution(records: tuple[SearchRecord, ...], output: Path) -> None:
    by_generation: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record.record_type == "candidate":
            by_generation[record.generation].append(record.evaluation.accuracy_500)
    generations = sorted(by_generation)
    if not generations:
        raise ValueError("search archive contains no candidate generations")
    maxima = [max(by_generation[generation]) for generation in generations]
    best_so_far = np.maximum.accumulate(maxima)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for generation in generations:
        values = by_generation[generation]
        axis.scatter(
            [generation] * len(values),
            values,
            alpha=0.45,
            color="#829ab1",
            s=28,
            label="Candidate validation accuracy"
            if generation == generations[0]
            else None,
        )
    axis.plot(
        generations,
        maxima,
        marker="o",
        color="#e07a32",
        label="Best in generation",
    )
    axis.plot(
        generations,
        best_so_far,
        marker="s",
        linestyle="--",
        color="#256d5b",
        label="Best so far",
    )
    axis.set_xlabel("Search generation")
    axis.set_ylabel("Validation accuracy at 500 training examples")
    axis.set_title("Bias search evolution (validation only)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "bias_evolution.png", dpi=180)
    plt.close(figure)


def _plot_feature_dimension(report: dict[str, Any], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for result in _unique_representations(report):
        summary = result["summaries"]["500"]
        baseline = result["kind"] == "baseline"
        axis.errorbar(
            result["feature_dim"],
            summary["accuracy_mean"],
            yerr=summary["accuracy_std"],
            fmt="s" if baseline else "o",
            capsize=3,
            markersize=7,
            label=result["name"],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Feature dimension (log scale)")
    axis.set_ylabel("Official test accuracy at 500 examples (mean ± SD)")
    axis.set_title("Representation size versus low-data accuracy")
    axis.grid(True, which="both", alpha=0.25)
    figure.subplots_adjust(right=0.72)
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout(rect=(0, 0, 0.72, 1))
    figure.savefig(output / "feature_dimension_vs_accuracy.png", dpi=180)
    plt.close(figure)


def _plot_runtime(report: dict[str, Any], output: Path) -> None:
    # A repeated representation reuses the same full-dataset feature matrix;
    # assign its first-run transform cost to each equivalent report entry.
    costs_by_representation: dict[str, float] = {}
    for result in report["results"]:
        key = _operator_key(result)
        costs_by_representation[key] = max(
            costs_by_representation.get(key, 0.0),
            result["feature_matrix_runtime_ms"],
        )

    figure, axis = plt.subplots(figsize=(10, 6))
    unique_results = _unique_representations(report)
    for result in unique_results:
        summary = result["summaries"]["500"]
        runtime_ms = (
            costs_by_representation[_operator_key(result)]
            + summary["training_runtime_ms_mean"]
            + summary["inference_runtime_ms_mean"]
        )
        baseline = result["kind"] == "baseline"
        axis.scatter(
            runtime_ms / 1_000,
            summary["accuracy_mean"],
            marker="s" if baseline else "o",
            s=56,
            label=result["name"],
        )
    axis.set_xscale("log")
    axis.set_xlabel("Feature extraction + fit + inference time (seconds, log scale)")
    axis.set_ylabel("Official test accuracy at 500 examples")
    axis.set_title("Runtime versus low-data test accuracy")
    axis.grid(True, which="both", alpha=0.25)
    figure.subplots_adjust(right=0.72)
    axis.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.tight_layout(rect=(0, 0, 0.72, 1))
    figure.savefig(output / "runtime_vs_accuracy.png", dpi=180)
    plt.close(figure)


def _plot_confusion_matrices(report: dict[str, Any], output: Path) -> None:
    finalists = [result for result in report["results"] if result["kind"] == "finalist"]
    columns = 4
    rows = (len(finalists) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(15, 7.5), squeeze=False)
    for axis, result in zip(axes.flat, finalists, strict=False):
        matrix = np.asarray(
            result["summaries"]["500"]["confusion_matrix_sum"], dtype=np.float64
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0
        )
        image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        axis.set_title(result["name"], fontsize=8)
        axis.set_xticks(range(10), labels=range(10), fontsize=6)
        axis.set_yticks(range(10), labels=range(10), fontsize=6)
        axis.set_xlabel("Predicted", fontsize=7)
        axis.set_ylabel("Actual", fontsize=7)
        for row in range(10):
            for column in range(10):
                axis.text(
                    column,
                    row,
                    f"{normalized[row, column]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if normalized[row, column] > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(finalists) :]:
        axis.remove()
    figure.suptitle("Finalist confusion matrices at 500 examples (3 seeds combined)")
    figure.tight_layout()
    figure.savefig(output / "finalist_confusion_matrices.png", dpi=200)
    plt.close(figure)


def _plot_ablations(records: tuple[SearchRecord, ...], output: Path) -> None:
    records_by_id = {record.candidate_id: record for record in records}
    comparisons = []
    for record in records:
        if record.record_type != "ablation" or record.base_candidate_id is None:
            continue
        parent = records_by_id.get(record.base_candidate_id)
        if parent is None:
            continue
        delta = record.evaluation.accuracy_500 - parent.evaluation.accuracy_500
        label = (
            f"G{record.generation} −{record.removed_operator} · "
            f"{record.candidate_id[:6]}"
        )
        comparisons.append((label, delta))
    if not comparisons:
        raise ValueError("search archive contains no parent-linked ablations")
    comparisons.sort(key=lambda item: item[1])

    labels = [label for label, _ in comparisons]
    deltas = [delta * 100 for _, delta in comparisons]
    colors = ["#3f8f70" if delta > 0 else "#bd5b57" for delta in deltas]
    figure, axis = plt.subplots(figsize=(12, max(6, len(labels) * 0.32)))
    positions = np.arange(len(labels))
    axis.barh(positions, deltas, color=colors)
    axis.set_yticks(positions, labels=labels, fontsize=7)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Δ validation accuracy@500 (ablated − parent, percentage points)")
    axis.set_title("Ablation contribution across searched candidates")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "ablation_contributions.png", dpi=180)
    plt.close(figure)


def generate_plots(
    *,
    search_path: Path = Path("results/search.jsonl"),
    final_evaluation_path: Path = Path("results/final_evaluation.json"),
    output_dir: Path = Path("results/figures"),
) -> tuple[Path, ...]:
    """Validate report provenance and save the six requested experiment plots."""
    search_path = Path(search_path)
    final_evaluation_path = Path(final_evaluation_path)
    report = json.loads(final_evaluation_path.read_text(encoding="utf-8"))
    if report.get("test_set_accessed") is not True:
        raise ValueError("final evaluation report does not record test-set access")
    if report.get("finalist_selection", {}).get("frozen") is not True:
        raise ValueError("final evaluation report is not linked to a frozen selection")
    expected_search_hash = report["finalist_selection"].get("search_archive_sha256")
    actual_search_hash = hashlib.sha256(search_path.read_bytes()).hexdigest()
    if actual_search_hash != expected_search_hash:
        raise ValueError("search archive differs from the frozen final evaluation")

    archive = SearchArchive(search_path)
    records = archive.records
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_learning_curve(report, output_dir)
    _plot_bias_evolution(records, output_dir)
    _plot_feature_dimension(report, output_dir)
    _plot_runtime(report, output_dir)
    _plot_confusion_matrices(report, output_dir)
    _plot_ablations(records, output_dir)
    return tuple(
        output_dir / filename
        for filename in (
            "learning_curve.png",
            "bias_evolution.png",
            "feature_dimension_vs_accuracy.png",
            "runtime_vs_accuracy.png",
            "finalist_confusion_matrices.png",
            "ablation_contributions.png",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, default=Path("results/search.jsonl"))
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("results/final_evaluation.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    for path in generate_plots(
        search_path=args.search,
        final_evaluation_path=args.evaluation,
        output_dir=args.output_dir,
    ):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
