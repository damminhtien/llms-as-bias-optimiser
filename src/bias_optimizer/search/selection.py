"""Freeze a low-data finalist set without loading the test partition."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bias_optimizer.domain.bias import bias_spec_hash
from bias_optimizer.domain.search import SearchRecord
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.search.archive import SearchArchive


def _candidate_summary(record: SearchRecord, roles: set[str]) -> dict[str, Any]:
    evaluation = record.evaluation
    return {
        "candidate_id": record.candidate_id,
        "bias": record.bias.to_dict(),
        "selection_roles": sorted(roles),
        "accuracy_500": evaluation.accuracy_500,
        "accuracy_5000": evaluation.accuracy_5000,
        "feature_dim": evaluation.feature_dim,
        "runtime_ms": evaluation.runtime_ms,
        "ranking_score": evaluation.ranking_score,
    }


def select_finalists(
    *,
    archive_path: Path = Path("results/search.jsonl"),
    output_path: Path = Path("results/finalists.json"),
) -> dict[str, Any]:
    """Select the top five plus smallest, fastest, and original stroke-flow bias."""
    archive_path = Path(archive_path)
    archive = SearchArchive(archive_path)
    eligible = archive.top_by_accuracy(max(1, len(archive.records)))
    if not eligible:
        raise ValueError("search archive contains no eligible candidates")

    roles: dict[str, set[str]] = {}
    records_by_id: dict[str, SearchRecord] = {}

    def include(record: SearchRecord, role: str) -> None:
        records_by_id[record.candidate_id] = record
        roles.setdefault(record.candidate_id, set()).add(role)

    for rank, record in enumerate(archive.top_by_accuracy(5), start=1):
        include(record, f"top_5_low_data_accuracy_rank_{rank}")

    smallest = min(
        eligible,
        key=lambda record: (
            record.evaluation.feature_dim,
            -record.evaluation.accuracy_500,
            record.evaluation.runtime_ms,
            record.candidate_id,
        ),
    )
    include(smallest, "smallest_representation")

    fastest = min(
        eligible,
        key=lambda record: (
            record.evaluation.runtime_ms,
            -record.evaluation.accuracy_500,
            record.evaluation.feature_dim,
            record.candidate_id,
        ),
    )
    include(fastest, "fastest_representation")

    stroke_flow = archive.get(bias_spec_hash(initial_human_biases()[-1]))
    if stroke_flow is None or stroke_flow.record_type != "candidate":
        raise ValueError("original stroke-flow seed is missing from the archive")
    include(stroke_flow, "original_stroke_flow_hypothesis")

    candidates = [
        _candidate_summary(records_by_id[candidate_id], candidate_roles)
        for candidate_id, candidate_roles in roles.items()
    ]
    metadata: dict[str, Any] = {
        "selected_at_utc": datetime.now(UTC).isoformat(),
        "finalist_selection_frozen": True,
        "test_set_accessed": False,
        "search_archive": str(archive_path),
        "search_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "criteria": [
            "top five by validation accuracy at 500 training examples",
            "one smallest representation",
            "one fastest representation",
            "original stroke-flow seed regardless of rank",
        ],
        "finalists": candidates,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata
