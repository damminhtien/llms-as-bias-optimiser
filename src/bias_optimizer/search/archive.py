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
        """Return up to k records ranked by score with stable tie breaks."""
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        ranked = sorted(
            self._records,
            key=lambda record: (
                -record.evaluation.ranking_score,
                -record.evaluation.accuracy_500,
                -record.evaluation.accuracy_5000,
                record.evaluation.feature_dim,
                record.candidate_id,
            ),
        )
        return tuple(ranked[:k])

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
