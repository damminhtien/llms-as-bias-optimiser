"""Audit V3 pilot diversity and apply its predeclared go/no-go rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.novelty.descriptors import mechanism_family

_REQUEST_COUNT = re.compile(r"Propose exactly (\d+) distinct representation programs")
_MECHANISM_FOCUS = re.compile(r"Requested mechanism family for this batch: ([a-z_]+)")
_RELATIONAL_OPERATIONS = {
    "spatial_condition",
    "pairwise_difference",
    "cross_histogram",
    "path_summary",
}


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _is_global_histogram(record: ProgramSearchRecord) -> bool:
    descriptor = record.descriptor
    operations = {node.op for node in _walk(record.bias.program)}
    return (
        descriptor["order"] == "orderless"
        and descriptor["spatial"] == "global"
        and descriptor["composition"] == "single"
        and "histogram" in operations
    )


def _walk(expr):
    yield expr
    for child in expr.args:
        yield from _walk(child)


def _proposal_rates(
    path: Path,
    *,
    seed_programs: tuple[str, ...],
) -> dict[str, Any]:
    if not path.exists():
        return {
            "attempts": 0,
            "invalid_ast_count": 0,
            "duplicate_count": 0,
            "unique_valid_count": 0,
            "invalid_ast_rate": None,
            "duplicate_rate": None,
            "requested_mechanism_counts": {},
        }
    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.DISCOVERY,
            max_feature_dim=128,
        )
    )
    seen = set(seed_programs)
    attempts = 0
    invalid = 0
    duplicate = 0
    unique_valid = 0
    requested_families: Counter[str] = Counter()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        prompt = entry.get("prompt", "")
        count_match = _REQUEST_COUNT.search(prompt)
        requested = int(count_match.group(1)) if count_match else 0
        family_match = _MECHANISM_FOCUS.search(prompt)
        if family_match:
            requested_families[family_match.group(1)] += requested
        try:
            payload = json.loads(entry.get("response", ""))
            proposals = payload.get("proposals")
            if not isinstance(proposals, list):
                raise TypeError("response has no proposals array")
        except json.JSONDecodeError, ValueError, AttributeError:
            attempts += max(requested, 1)
            invalid += max(requested, 1)
            continue

        if requested > len(proposals):
            missing = requested - len(proposals)
            attempts += missing
            invalid += missing
        for proposal in proposals:
            attempts += 1
            try:
                bias = ProgramBiasSpec.from_dict(proposal)
                compiler.compile(bias.program)
            except KeyError, TypeError, ValueError:
                invalid += 1
                continue
            signature = bias.program.to_json()
            if signature in seen:
                duplicate += 1
                continue
            seen.add(signature)
            unique_valid += 1

    denominator = attempts or 1
    return {
        "attempts": attempts,
        "invalid_ast_count": invalid,
        "duplicate_count": duplicate,
        "unique_valid_count": unique_valid,
        "invalid_ast_rate": invalid / denominator,
        "duplicate_rate": duplicate / denominator,
        "requested_mechanism_counts": dict(sorted(requested_families.items())),
    }


def summarize_pilot(
    archive_path: Path,
    responses_path: Path,
    failures_path: Path | None = None,
    *,
    expected_seeds: int = 10,
    expected_generations: int = 2,
    candidates_per_generation: int = 15,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    responses_path = Path(responses_path)
    failures_path = (
        Path(failures_path)
        if failures_path is not None
        else archive_path.with_name(f"{archive_path.stem}_failures.jsonl")
    )
    records = tuple(
        ProgramSearchRecord.from_json(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    by_generation = Counter(record.generation for record in records)
    expected = {
        0: expected_seeds,
        **{
            generation: candidates_per_generation
            for generation in range(1, expected_generations + 1)
        },
    }
    complete = all(
        by_generation[generation] == count for generation, count in expected.items()
    )
    valid = tuple(record for record in records if record.generation > 0)
    outside_histogram = sum(not _is_global_histogram(record) for record in valid)
    outside_fraction = outside_histogram / len(valid) if valid else 0.0
    dimensions = [record.evaluation.feature_dim for record in valid]
    families = Counter(mechanism_family(record.bias.program) for record in valid)
    relational_count = sum(
        bool({node.op for node in _walk(record.bias.program)} & _RELATIONAL_OPERATIONS)
        for record in valid
    )
    rates = _proposal_rates(
        responses_path,
        seed_programs=tuple(
            record.bias.program.to_json()
            for record in records
            if record.generation == 0
        ),
    )
    failures = (
        [
            json.loads(line)
            for line in failures_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if failures_path.exists()
        else []
    )
    failure_types = Counter(item["failure_type"] for item in failures)
    go = complete and outside_fraction >= 0.30
    return {
        "archive_sha256": _sha256(archive_path),
        "response_archive_sha256": _sha256(responses_path),
        "failure_archive_sha256": _sha256(failures_path),
        "candidate_count": len(records),
        "expected_candidate_count": expected_seeds
        + expected_generations * candidates_per_generation,
        "generation_counts": {str(key): by_generation[key] for key in sorted(expected)},
        "complete": complete,
        "descriptor_cell_count": len({record.descriptor_cell for record in records}),
        "occupied_descriptor_cells": [
            list(cell)
            for cell in sorted({record.descriptor_cell for record in records})
        ],
        "outside_global_histogram_count": outside_histogram,
        "outside_global_histogram_fraction": outside_fraction,
        "valid_proposal_candidate_count": len(valid),
        "invalid_ast_count": rates["invalid_ast_count"],
        "invalid_ast_rate": rates["invalid_ast_rate"],
        "proposal_attempt_count": rates["attempts"],
        "duplicate_proposal_count": rates["duplicate_count"],
        "duplicate_rate": rates["duplicate_rate"],
        "feature_dimensions": {
            "min": min(dimensions) if dimensions else None,
            "median": float(sorted(dimensions)[len(dimensions) // 2])
            if dimensions
            else None,
            "max": max(dimensions) if dimensions else None,
        },
        "order_sensitive_candidate_count": families["order_sensitive"],
        "mechanism_family_counts": dict(sorted(families.items())),
        "relational_candidate_count": relational_count,
        "requested_mechanism_counts": rates["requested_mechanism_counts"],
        "llm_models": sorted({record.model for record in valid if record.model}),
        "evaluated_train_sizes": [500, 5_000],
        "mechanism_mismatch_count": failure_types["mechanism_mismatch"],
        "evaluation_failure_count": failure_types["evaluation_error"],
        "decision": "GO" if go else "NO-GO",
        "go_no_go_rule": "40 complete records and at least 30% of valid LLM proposals, excluding hand-built seeds, outside the global-histogram family",
    }


def _markdown(summary: dict[str, Any]) -> str:
    dims = summary["feature_dimensions"]
    families = summary["mechanism_family_counts"] or {"none": 0}
    requested = summary["requested_mechanism_counts"] or {"none": 0}
    cells = (
        "\n".join(f"- `{tuple(cell)}`" for cell in summary["occupied_descriptor_cells"])
        or "- No cells"
    )
    return f"""# V3 Pilot Report

## Gate

- Decision: **{summary["decision"]}**.
- Candidates: {summary["candidate_count"]}/{summary["expected_candidate_count"]}.
- Complete generations: {summary["generation_counts"]}.
- LLM proposals outside the global-histogram family: {summary["outside_global_histogram_count"]}/{summary["valid_proposal_candidate_count"]} ({summary["outside_global_histogram_fraction"]:.1%}).
- Predeclared rule: {summary["go_no_go_rule"]}.
- Evaluated train sizes: {summary["evaluated_train_sizes"]}.
- LLM model: {", ".join(summary["llm_models"]) or "not recorded"}.
- Archive SHA-256: `{summary["archive_sha256"]}`.
- Response archive SHA-256: `{summary["response_archive_sha256"]}`.
- Failure archive SHA-256: `{summary["failure_archive_sha256"]}`.

## Search health

- Occupied descriptor cells: {summary["descriptor_cell_count"]}.
- Invalid AST rate: {summary["invalid_ast_rate"]:.1%} ({summary["invalid_ast_count"]}/{summary["proposal_attempt_count"]} attempts).
- Duplicate proposal rate: {summary["duplicate_rate"]:.1%} ({summary["duplicate_proposal_count"]}/{summary["proposal_attempt_count"]} attempts).
- Feature width min / median / max: {dims["min"]} / {dims["median"]} / {dims["max"]}.
- Order-sensitive candidates: {summary["order_sensitive_candidate_count"]}.
- Relational candidates: {summary["relational_candidate_count"]}.
- Rejected for mechanism mismatch: {summary["mechanism_mismatch_count"]}.
- Rejected for evaluation errors: {summary["evaluation_failure_count"]}.
- Accepted mechanism families: {families}.
- Requested proposal slots, including refills: {requested}.

## Descriptor cells

{cells}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("results/v3_pilot_program_search_discovery.jsonl"),
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=Path("results/v3_pilot_llm_responses_discovery.jsonl"),
    )
    parser.add_argument("--failures", type=Path)
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/v3_pilot_summary.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/V3_PILOT.md")
    )
    args = parser.parse_args()
    summary = summarize_pilot(args.archive, args.responses, args.failures)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
