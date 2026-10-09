"""Build a compact V3 discovery report from the staged search artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.novelty.descriptors import mechanism_family


def _archive_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_search(
    archive_path: Path,
    finalists_path: Path | None = None,
    behavioral_path: Path | None = None,
    responses_path: Path | None = None,
    failures_path: Path | None = None,
    compact_baseline_path: Path | None = None,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    records = tuple(
        ProgramSearchRecord.from_json(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("V3 search archive is empty")
    generations = Counter(record.generation for record in records)
    best_by_cell: dict[tuple[str, ...], ProgramSearchRecord] = {}
    for record in records:
        current = best_by_cell.get(record.descriptor_cell)
        if (
            current is None
            or record.evaluation.ranking_score > current.evaluation.ranking_score
        ):
            best_by_cell[record.descriptor_cell] = record
    finalists = None
    if finalists_path is not None and Path(finalists_path).exists():
        finalists = json.loads(Path(finalists_path).read_text(encoding="utf-8"))
        if finalists.get("search_archive_sha256") != _archive_sha256(archive_path):
            raise ValueError("finalist evaluations refer to a different search archive")
    behavioral = None
    if behavioral_path is not None and Path(behavioral_path).exists():
        behavioral = json.loads(Path(behavioral_path).read_text(encoding="utf-8"))
        if behavioral.get("search_archive_sha256") != _archive_sha256(archive_path):
            raise ValueError("behavioral novelty refers to a different search archive")
    behavior_by_id = (
        {item["candidate_id"]: item for item in behavioral["results"]}
        if behavioral is not None
        else {}
    )
    responses_path = Path(responses_path) if responses_path is not None else None
    failures_path = Path(failures_path) if failures_path is not None else None
    failures = []
    if failures_path is not None and failures_path.exists():
        failures = [
            json.loads(line)
            for line in failures_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    compact_baseline = None
    if compact_baseline_path is not None and Path(compact_baseline_path).exists():
        compact_baseline = json.loads(
            Path(compact_baseline_path).read_text(encoding="utf-8")
        )
    return {
        "archive_sha256": _archive_sha256(archive_path),
        "candidate_count": len(records),
        "generation_counts": dict(sorted(generations.items())),
        "proposal_family_counts": dict(
            sorted(
                Counter(
                    mechanism_family(record.bias.program) for record in records
                ).items()
            )
        ),
        "model_counts": dict(
            sorted(Counter(record.model or "unknown" for record in records).items())
        ),
        "llm_response_count": (
            sum(
                bool(line.strip())
                for line in responses_path.read_text(encoding="utf-8").splitlines()
            )
            if responses_path is not None and responses_path.exists()
            else None
        ),
        "failure_counts": dict(
            sorted(Counter(item["failure_type"] for item in failures).items())
        ),
        "descriptor_cell_count": len(best_by_cell),
        "best_per_cell": [
            {
                "descriptor": dict(
                    zip(
                        ("source", "order", "spatial", "composition", "complexity"),
                        cell,
                        strict=True,
                    )
                ),
                "candidate_id": record.candidate_id,
                "name": record.bias.name,
                "mechanism": record.bias.mechanism,
                "accuracy_500": record.evaluation.accuracy_500,
                "accuracy_5000": record.evaluation.accuracy_5000,
                "ranking_score": record.evaluation.ranking_score,
                "feature_dim": record.evaluation.feature_dim,
                "runtime_ms": record.evaluation.runtime_ms,
                "novelty": record.novelty,
                "behavioral_novelty": behavior_by_id.get(record.candidate_id, {}).get(
                    "behavioral_novelty"
                ),
                "program": record.bias.program.to_dict(),
            }
            for cell, record in sorted(best_by_cell.items())
        ],
        "finalist_evaluations": finalists,
        "behavioral_novelty": behavioral,
        "compact_baseline": compact_baseline,
    }


def _markdown(summary: dict[str, Any]) -> str:
    rows = summary["best_per_cell"]
    cells = [
        "| Descriptor cell | Candidate | A@500 | A@5000 | Dim | Runtime ms | Structural novelty | Behavioral novelty |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        descriptor = row["descriptor"]
        cell = " / ".join(descriptor.values())
        accuracy_5000 = row["accuracy_5000"]
        cells.append(
            f"| `{cell}` | `{row['name']}` | {row['accuracy_500']:.4f} | "
            f"{accuracy_5000 if accuracy_5000 is not None else 'not evaluated'} | "
            f"{row['feature_dim']} | {row['runtime_ms']:.1f} | {row['novelty']:.3f} | "
            f"{row['behavioral_novelty'] if row['behavioral_novelty'] is not None else 'not evaluated'} |"
        )
    finalists = summary["finalist_evaluations"]
    finalist_section = "Top-20 stage-two evaluation has not been run."
    if finalists is not None:
        finalist_rows = [
            "| Rank | Candidate | A@500 | A@5000 | Dim | Full score |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for item in finalists["evaluations"]:
            result = item["full_validation"]
            finalist_rows.append(
                f"| {item['rank']} | `{item['bias']['name']}` | "
                f"{result['accuracy_500']:.4f} | {result['accuracy_5000']:.4f} | "
                f"{result['feature_dim']} | {result['ranking_score']:.4f} |"
            )
        finalist_section = "\n".join(finalist_rows)
    compact = summary["compact_baseline"]
    compact_section = "The 16-dimensional zoning result is pending."
    if compact is not None:
        baselines = compact.get("baselines")
        if isinstance(baselines, list):
            compact_section = "; ".join(
                f"`{item['name']}` ({item['feature_dim']}D) scored "
                f"{item['mnist_validation_accuracy_500']:.4f} at 500 and "
                f"{item['mnist_validation_accuracy_5000']:.4f} at 5,000 "
                "MNIST validation examples"
                for item in baselines
            )
        else:
            compact_section = (
                f"`{compact['name']}` ({compact['feature_dim']}D) scored "
                f"{compact['mnist_validation_accuracy_500']:.4f} at 500 and "
                f"{compact['mnist_validation_accuracy_5000']:.4f} at 5,000 "
                "MNIST validation examples."
            )
    generations = summary["generation_counts"]
    return f"""# V3 Program Search

## Protocol

- Candidate records: {summary["candidate_count"]}.
- Generation counts: `{generations}`.
- Accepted mechanism families: `{summary["proposal_family_counts"]}`.
- LLM response batches: `{summary["llm_response_count"]}`; recorded exclusions: `{summary["failure_counts"]}`.
- Proposal model(s): `{summary["model_counts"]}`.
- MAP-Elites cells: {summary["descriptor_cell_count"]}.
- Stage one: all proposals scored on MNIST validation with 500 training examples.
- Stage two: only the top 20 are re-evaluated at 500 and 5,000 examples.
- Search archive SHA-256: `{summary["archive_sha256"]}`.
- Behavioral novelty uses validation probes and pairwise-distance correlation; it is shown when `evaluate_v3_behavioral_novelty.py` has completed.

## Compact classical baseline

{compact_section}

The frozen V2 references are `cycle_angle_hist` at 0.7180 validation accuracy
with 500 examples and the post-search `raw + angle_hist` composition at 0.8525.
See [`V2_PROGRAM_SYNTHESIS.md`](V2_PROGRAM_SYNTHESIS.md) for their protocols and
limitations.

## Best candidate in each descriptor cell

{chr(10).join(cells)}

Each table row's full AST and mechanism statement are available in the companion
JSON summary generated by `experiments/summarize_v3_search.py`.

## Top-20 full validation evaluation

{finalist_section}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("results/v3_program_search_discovery.jsonl"),
    )
    parser.add_argument(
        "--finalists", type=Path, default=Path("results/v3_finalist_evaluations.json")
    )
    parser.add_argument(
        "--behavioral-novelty",
        type=Path,
        default=Path("results/v3_behavioral_novelty.json"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("results/v3_program_llm_responses_discovery.jsonl"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/v3_program_search_discovery_failures.jsonl"),
    )
    parser.add_argument(
        "--compact-baseline",
        type=Path,
        default=Path("results/v3_compact_baseline.json"),
    )
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/v3_search_summary.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/V3_SEARCH.md")
    )
    args = parser.parse_args()
    finalists_path = args.finalists if args.finalists.exists() else None
    behavioral_path = (
        args.behavioral_novelty if args.behavioral_novelty.exists() else None
    )
    summary = summarize_search(
        args.archive,
        finalists_path,
        behavioral_path,
        args.responses,
        args.failures,
        args.compact_baseline,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": summary["candidate_count"],
                "descriptor_cell_count": summary["descriptor_cell_count"],
                "archive_sha256": summary["archive_sha256"],
                "markdown_output": str(args.markdown_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
