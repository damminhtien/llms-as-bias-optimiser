"""Freeze MNIST-validation finalists before loading any transfer test data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(path: Path) -> tuple[ProgramSearchRecord, ...]:
    records = tuple(
        ProgramSearchRecord.from_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("discovery archive is empty")
    if any(record.track != SearchTrack.DISCOVERY.value for record in records):
        raise ValueError("candidate archive contains a non-discovery record")
    return records


def freeze_candidates(
    archive_path: Path,
    output_path: Path,
    *,
    count: int = 5,
) -> dict[str, object]:
    """Rank by MNIST validation A@500, verify raw-free ASTs, and freeze IDs."""
    archive_path = Path(archive_path)
    records = _load_records(archive_path)
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    unique: dict[str, ProgramSearchRecord] = {}
    for record in records:
        compiler.compile(record.bias.program)
        unique.setdefault(record.candidate_id, record)
    finalists = sorted(
        unique.values(),
        key=lambda item: (
            -item.evaluation.accuracy_500,
            -item.evaluation.ranking_score,
            item.candidate_id,
        ),
    )[:count]
    if len(finalists) < count:
        raise ValueError(f"need {count} unique candidates; found {len(finalists)}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "finalist_selection_frozen": True,
        "test_set_accessed": False,
        "selection_rule": "top discovery records by MNIST validation accuracy_500, then ranking_score, then candidate_id",
        "search_archive": str(archive_path),
        "search_archive_sha256": _sha256(archive_path),
        "candidate_count": len(unique),
        "finalist_count": len(finalists),
        "finalists": [
            {
                "candidate_id": record.candidate_id,
                "name": record.bias.name,
                "mnist_validation_accuracy_500": record.evaluation.accuracy_500,
                "mnist_validation_accuracy_5000": record.evaluation.accuracy_5000,
                "feature_dim": record.evaluation.feature_dim,
                "niche": record.niche,
                "complexity_bin": record.complexity_bin,
                "bias": record.bias.to_dict(),
            }
            for record in finalists
        ],
        "transfer_protocol": {
            "datasets": [
                "emnist_digits",
                "emnist_letters",
                "kmnist",
                "fashion_mnist",
            ],
            "representation_frozen": True,
            "classifier_refit_per_domain": True,
            "train_sizes": [500, 5000],
            "transfer_test_sample_size": 2000,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("results/program_search_discovery.jsonl")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v2_transfer_finalists.json")
    )
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    manifest = freeze_candidates(args.archive, args.output, count=args.count)
    print(
        json.dumps(
            {
                "finalist_count": manifest["finalist_count"],
                "candidate_ids": [
                    finalist["candidate_id"]
                    for finalist in manifest["finalists"]  # type: ignore[index]
                ],
                "archive_sha256": manifest["search_archive_sha256"],
                "test_set_accessed": manifest["test_set_accessed"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
