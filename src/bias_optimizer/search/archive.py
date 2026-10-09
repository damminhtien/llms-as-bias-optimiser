"""JSONL-backed search history with deterministic candidate identities."""

from __future__ import annotations

import os
from pathlib import Path

from bias_optimizer.domain.search import SearchRecord


class SearchArchive:
    """Persist evaluated candidates and provide stable ranking queries."""

    def __init__(self, path: Path = Path("results/search.jsonl")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[SearchRecord] = []
        self._records_by_id: dict[str, SearchRecord] = {}
        if self.path.exists():
            self._load()

    @property
    def records(self) -> tuple[SearchRecord, ...]:
        """Return records in archive insertion order."""
        return tuple(self._records)

    def add(self, record: SearchRecord) -> None:
        """Append a record once; equal retries are idempotent."""
        if not isinstance(record, SearchRecord):
            raise TypeError("search archive requires a SearchRecord")
        existing = self._records_by_id.get(record.candidate_id)
        if existing is not None:
            if existing.to_json() == record.to_json():
                return
            raise ValueError(
                f"candidate {record.candidate_id} already has a different record"
            )

        with self.path.open("a", encoding="utf-8") as archive:
            archive.write(record.to_json() + "\n")
            archive.flush()
            os.fsync(archive.fileno())
        self._records.append(record)
        self._records_by_id[record.candidate_id] = record

    def top(self, k: int) -> tuple[SearchRecord, ...]:
        """Rank candidates and ablations that improve low-data accuracy."""
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        ranked = sorted(
            self._rankable_records(),
            key=lambda record: (
                -record.evaluation.ranking_score,
                -record.evaluation.accuracy_500,
                -record.evaluation.accuracy_5000,
                record.evaluation.feature_dim,
                record.candidate_id,
            ),
        )
        return tuple(ranked[:k])

    def top_by_accuracy(self, k: int) -> tuple[SearchRecord, ...]:
        """Rank eligible representations by validation accuracy at 500 samples."""
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        ranked = sorted(
            self._rankable_records(),
            key=lambda record: (
                -record.evaluation.accuracy_500,
                -record.evaluation.accuracy_5000,
                record.evaluation.feature_dim,
                record.evaluation.runtime_ms,
                record.candidate_id,
            ),
        )
        return tuple(ranked[:k])

    def _rankable_records(self) -> tuple[SearchRecord, ...]:
        return tuple(record for record in self._records if self._is_rankable(record))

    def _is_rankable(self, record: SearchRecord) -> bool:
        if record.record_type == "candidate":
            return True
        if record.record_type != "ablation" or record.base_candidate_id is None:
            return False
        baseline = self._records_by_id.get(record.base_candidate_id)
        return (
            baseline is not None
            and record.evaluation.accuracy_500 > baseline.evaluation.accuracy_500
        )

    def contains(self, candidate_id: str) -> bool:
        """Return whether this candidate ID has already been evaluated."""
        return candidate_id in self._records_by_id

    def get(self, candidate_id: str) -> SearchRecord | None:
        """Return a record by candidate ID, if it exists."""
        return self._records_by_id.get(candidate_id)

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as archive:
            for line_number, line in enumerate(archive, start=1):
                if not line.strip():
                    continue
                try:
                    record = SearchRecord.from_json(line)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid search archive record at line {line_number}: {exc}"
                    ) from exc
                existing = self._records_by_id.get(record.candidate_id)
                if existing is not None:
                    if existing.to_json() != record.to_json():
                        raise ValueError(
                            "conflicting records for candidate "
                            f"{record.candidate_id} at line {line_number}"
                        )
                    continue
                self._records.append(record)
                self._records_by_id[record.candidate_id] = record
