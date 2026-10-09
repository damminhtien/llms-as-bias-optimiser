"""Build a compact, provenance-linked V2 report from archived run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.novelty.descriptors import IMPLEMENTED_NICHES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _search_section(track: str, *, results_dir: Path) -> str:
    archive = results_dir / f"program_search_{track}.jsonl"
    if not archive.exists():
        raise FileNotFoundError(f"missing {track} archive: {archive}")
    records = [
        ProgramSearchRecord.from_json(line)
        for line in archive.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generations = Counter(record.generation for record in records)
    expected = {0: 20, **{generation: 18 for generation in range(1, 11)}}
    if any(generations[generation] != count for generation, count in expected.items()):
        raise ValueError(
            f"{track} search is incomplete: "
            + ", ".join(
                f"g{generation}={generations[generation]}/{count}"
                for generation, count in expected.items()
                if generations[generation] != count
            )
        )
    responses_path = results_dir / f"program_llm_responses_{track}.jsonl"
    responses = (
        [
            json.loads(line)
            for line in responses_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if responses_path.exists()
        else []
    )
    accepted = sum(bool(item.get("accepted")) for item in responses)
    request_sizes = Counter()
    for item in responses:
        match = re.search(
            r"Propose exactly (\d+) distinct representation programs",
            item.get("prompt", ""),
        )
        if match:
            request_sizes[int(match.group(1))] += 1
    best = sorted(
        records,
        key=lambda record: (
            -record.evaluation.accuracy_500,
            -record.evaluation.ranking_score,
            record.candidate_id,
        ),
    )[:5]
    niche_counts = Counter(record.niche for record in records)
    target_niches = {
        niche
        for niche in IMPLEMENTED_NICHES
        if track == "augmentation" or niche != "raw_pixels"
    }
    underexplored_niches = sorted(target_niches - set(niche_counts))

    lines = [
        f"### {track.title()} track",
        "",
        f"- Candidate archive: `{archive}` (SHA-256 `{_sha256(archive)}`).",
        f"- Records: {len(records)} (20 seeds + 10 × 18 proposals).",
        f"- LLM response records: {len(responses)}; accepted response batches: {accepted}.",
        "- Requested programs per LLM batch: "
        + ", ".join(f"{size}: {count}" for size, count in sorted(request_sizes.items()))
        + ".",
        "- Generation counts: "
        + ", ".join(
            f"{generation}: {generations[generation]}" for generation in range(11)
        )
        + ".",
        "- Implemented niches with no candidates: "
        + (", ".join(underexplored_niches) if underexplored_niches else "none")
        + ".",
        "- MAP-Elites occupied niches: "
        + ", ".join(
            f"{niche} ({count})" for niche, count in sorted(niche_counts.items())
        )
        + ".",
        "",
        "| Candidate | Niche | Features | Validation accuracy @500 | @5,000 | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for record in best:
        lines.append(
            f"| `{record.bias.name}` (`{record.candidate_id}`) | {record.niche} | "
            f"{record.evaluation.feature_dim} | {record.evaluation.accuracy_500:.4f} | "
            f"{record.evaluation.accuracy_5000:.4f} | {record.evaluation.ranking_score:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _transfer_section(report: dict[str, Any] | None) -> str:
    if report is None:
        return "Transfer evaluation has not been run."
    rows = sorted(
        report["results"],
        key=lambda item: (item["dataset"], item["kind"], item["name"]),
    )
    lines = [
        f"Manifest SHA-256: `{report['protocol']['selection_manifest_sha256']}`.",
        f"Candidate archive SHA-256: `{report['protocol']['search_archive_sha256']}`.",
        (
            f"Training sizes {report['protocol']['train_sizes']}, train seeds "
            f"{report['protocol']['train_seeds']}, stratified test sample "
            f"{report['protocol']['transfer_test_sample_size']} per domain."
        ),
        "",
        "| Dataset | Representation | Type | Dim | Train size | Test accuracy (mean ± SD) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        for size in report["protocol"]["train_sizes"]:
            summary = row["summaries"][str(size)]
            lines.append(
                f"| {row['dataset']} | {row['name']} | {row['kind']} | "
                f"{row['feature_dim']} | {size} | "
                f"{summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f} |"
            )
    lines.extend(
        [
            "",
            ("Test split provenance (partition, source URL, and checksum):"),
        ]
    )
    for dataset in report["protocol"]["datasets"]:
        row = next(item for item in rows if item["dataset"] == dataset)
        lines.append(
            f"- `{dataset}`: {row['test_partition']}; source `{row['source']}` "
            f"(SHA-256 `{row['source_sha256']}`)."
        )
    return "\n".join(lines)


def _pixel_augmentation_section(report: dict[str, Any] | None) -> str:
    if report is None:
        return "Raw-pixel augmentation controls have not been run."
    rows = report["results"]
    lines = [
        f"Selection manifest SHA-256: `{report['protocol']['selection_manifest_sha256']}`.",
        f"Official test accessed: `{report['protocol']['official_test_accessed']}`.",
        "These are post-search compositions, not new LLM proposals; they use the same MNIST validation split that selected the finalists.",
        "",
        "| Representation | Type | Dim | Accuracy @500 | Δ vs raw | Accuracy @5,000 | Δ vs raw |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['kind']} | {row['feature_dim']} | "
            f"{row['accuracy_500']:.4f} | "
            f"{row['delta_accuracy_500_vs_raw_pixels']:+.4f} | "
            f"{row['accuracy_5000']:.4f} | "
            f"{row['delta_accuracy_5000_vs_raw_pixels']:+.4f} |"
        )
    lines.append("")
    lines.append(report["protocol"]["claim_boundary"])
    return "\n".join(lines)


def _counterfactual_section(report: dict[str, Any] | None) -> str:
    if report is None:
        return "Counterfactual evaluation has not been run."
    lines = [
        (
            f"Manifest SHA-256: `{report['protocol']['selection_manifest_sha256']}`; "
            f"test sample n={report['protocol']['test_sample_size']}."
        ),
        "",
        "| Candidate | Sequence node | Mean accuracy change | Mean prediction agreement |",
        "| --- | --- | ---: | ---: |",
    ]
    for result in report["results"]:
        for op in sorted(
            {item["target_op"] for item in result.get("interventions", [])}
        ):
            values = [
                item for item in result["interventions"] if item["target_op"] == op
            ]
            delta = sum(item["accuracy_delta"] for item in values) / len(values)
            agreement = sum(item["prediction_agreement"] for item in values) / len(
                values
            )
            lines.append(
                f"| {result['name']} | {op} | {delta:+.4f} | {agreement:.4f} |"
            )
        if result.get("status") == "no_sequence_valued_node":
            lines.append(f"| {result['name']} | none | — | — |")
    interventions = [
        intervention
        for result in report["results"]
        for intervention in result.get("interventions", [])
    ]
    if interventions:
        mean_delta = sum(item["accuracy_delta"] for item in interventions) / len(
            interventions
        )
        mean_agreement = sum(
            item["prediction_agreement"] for item in interventions
        ) / len(interventions)
        if mean_delta == 0.0 and mean_agreement == 1.0:
            lines.append(
                "\nEvery tested sequence-shuffle intervention left predictions unchanged "
                f"(mean accuracy change {mean_delta:+.4f}; agreement {mean_agreement:.4f}). "
                "These finalists provide no evidence that sequence order itself matters; "
                "their distributional features do not support an order-sensitive mechanism."
            )
    return "\n".join(lines)


def _invariance_section(report: dict[str, Any] | None) -> str:
    if report is None:
        return "Invariance search has not been run."
    lines = [
        f"Official test accessed: `{report['protocol']['official_test_accessed']}`.",
        f"Baseline validation accuracy: {report['baseline_selection_accuracy']:.4f}.",
        "",
        "| Proposal | Transform program | Accuracy delta | Agreement | Passed rule |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for proposal in report["proposals"]:
        compact = json.dumps(
            proposal["transform"], sort_keys=True, separators=(",", ":")
        )
        lines.append(
            f"| {proposal['name']} | `{compact}` | "
            f"{proposal['label_accuracy_delta']:+.4f} | "
            f"{proposal['prediction_agreement']:.4f} | "
            f"{proposal['passes_selection_rule']} |"
        )
    orbit = report.get("orbit_pooling")
    if orbit is None:
        lines.append(
            "\nNo proposal met the predeclared rule; orbit pooling was not evaluated."
        )
    else:
        lines.append(
            "\nOrbit mean on validation holdout: "
            f"{orbit['baseline_accuracy']:.4f} → {orbit['orbit_mean_accuracy']:.4f} "
            f"(Δ {orbit['accuracy_delta']:+.4f})."
        )
    return "\n".join(lines)


def build_report(results_dir: Path, output_path: Path) -> str:
    results_dir = Path(results_dir)
    pixel_augmentation = _read_json(results_dir / "v2_pixel_augmentation_controls.json")
    transfer = _read_json(results_dir / "v2_transfer_report.json")
    counterfactual = _read_json(results_dir / "v2_counterfactual_report.json")
    invariance = _read_json(results_dir / "v2_invariance_report.json")
    profile = _read_json(results_dir / "subexpression_cache_profile.json")
    sections = [
        "# V2 Representation Program Synthesis — Experimental Report",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} from local archives and reports.",
        "",
        "## Search results",
        "",
        _search_section("discovery", results_dir=results_dir),
        "",
        _search_section("augmentation", results_dir=results_dir),
        "",
        "## Raw-pixel augmentation controls",
        "",
        _pixel_augmentation_section(pixel_augmentation),
        "",
        "## Frozen transfer evaluation",
        "",
        _transfer_section(transfer),
        "",
        "## Sequence counterfactuals",
        "",
        _counterfactual_section(counterfactual),
        "",
        "## Invariance search and orbit pooling",
        "",
        _invariance_section(invariance),
        "",
        "## Cross-candidate cache profile",
        "",
    ]
    if profile is None:
        sections.append("Cache profiling has not been run.")
    else:
        sections.append(
            f"Compared {profile['candidate_count']} programs × {profile['image_count']} images. "
            f"Feature matrices exactly equal: `{profile['feature_matrices_exactly_equal']}`. "
            f"Uncached {profile['baseline_seconds']:.3f}s; cached {profile['cached_seconds']:.3f}s; "
            f"speedup {profile['speedup']:.3f}×; metrics `{profile['cache_metrics']}`."
        )
    sections.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "The discovery track excludes raw pixels and the augmentation track permits them. "
                "Transfer candidates are selected on MNIST validation and frozen before transfer "
                "test access. The transfer report uses stratified test samples of the stated size; "
                "it does not claim full-test accuracy. The invariance search uses MNIST train/validation "
                "only. Counterfactual accuracy changes support or weaken sequence-mechanism claims for "
                "the tested sample and representation. Both searches evaluated the same number of "
                "candidates, but augmentation request batches were capped at four after output "
                "truncation; cross-track maxima are descriptive, not a controlled comparison. None "
                "of these results alone proves a general or universal handwriting inductive bias."
            ),
            "",
            (
                "Dataset sources: [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), "
                "[CODH KMNIST](https://github.com/rois-codh/kmnist), "
                "[OpenML KMNIST mirror](https://www.openml.org/d/41982), and "
                "[Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)."
            ),
            "",
        ]
    )
    report = "\n".join(sections)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/V2_PROGRAM_SYNTHESIS.md")
    )
    args = parser.parse_args()
    report = build_report(args.results_dir, args.output)
    print(f"Saved report to {args.output} ({len(report)} characters)")


if __name__ == "__main__":
    main()
