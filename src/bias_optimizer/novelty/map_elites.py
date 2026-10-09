"""Quality-diversity archive retaining the best candidate in each descriptor cell."""

from __future__ import annotations

import json
import os
from pathlib import Path

from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.novelty.descriptors import DescriptorCell


class MapElitesArchive:
    """Persist candidates and keep one quality elite per five-axis descriptor cell."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[ProgramSearchRecord] = []
        self._by_id: dict[str, ProgramSearchRecord] = {}
        self._cells: dict[DescriptorCell, ProgramSearchRecord] = {}
        if self.path is not None and self.path.exists():
            for line_number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                try:
                    self._insert(ProgramSearchRecord.from_json(line))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid program archive record at line {line_number}"
                    ) from exc

    @property
    def records(self) -> tuple[ProgramSearchRecord, ...]:
        return tuple(self._records)

    @property
    def elites(self) -> tuple[ProgramSearchRecord, ...]:
        return tuple(
            sorted(
                self._cells.values(),
                key=lambda item: (-item.evaluation.ranking_score, item.candidate_id),
            )
        )

    @property
    def occupied_cells(self) -> tuple[DescriptorCell, ...]:
        return tuple(sorted(self._cells))

    def contains(self, candidate_id: str) -> bool:
        return candidate_id in self._by_id

    def elite(self, cell: DescriptorCell) -> ProgramSearchRecord | None:
        if not isinstance(cell, tuple) or len(cell) != 5:
            raise ValueError("descriptor cell must contain five axis values")
        return self._cells.get(cell)

    def add(self, record: ProgramSearchRecord) -> bool:
        if not isinstance(record, ProgramSearchRecord):
            raise TypeError("archive accepts ProgramSearchRecord values")
        if record.candidate_id in self._by_id:
            return False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(record.to_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self._insert(record)
        return True

    def top(self, count: int) -> tuple[ProgramSearchRecord, ...]:
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        return self.elites[:count]

    def underexplored_cells(
        self, cells: tuple[DescriptorCell, ...]
    ) -> tuple[DescriptorCell, ...]:
        return tuple(cell for cell in cells if cell not in self._cells)

    def underexplored_niches(self, niches: tuple[str, ...]) -> tuple[str, ...]:
        """Return V2 niche gaps for the frozen V2 report helper."""
        occupied = {record.niche for record in self._records}
        return tuple(niche for niche in niches if niche not in occupied)

    def _insert(self, record: ProgramSearchRecord) -> None:
        candidate_id = record.candidate_id
        if candidate_id in self._by_id:
            return
        self._records.append(record)
        self._by_id[candidate_id] = record
        cell = record.descriptor_cell
        current = self._cells.get(cell)
        if (
            current is None
            or record.evaluation.ranking_score > current.evaluation.ranking_score
        ):
            self._cells[cell] = record
