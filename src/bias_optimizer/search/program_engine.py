"""LLM-guided typed-program search with raw-free and augmentation tracks."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import (
    ProgramSearchFailure,
    ProgramSearchRecord,
)
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.mutations import add_raw_pixel_anchor
from bias_optimizer.dsl.seeds import generate_seed_programs
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.llm.program_proposer import (
    ProgramProposalError,
    ProgramProposer,
    ProgramSearchContext,
    ProposedProgram,
)
from bias_optimizer.ml.evaluator import ProgramEvaluator
from bias_optimizer.novelty.descriptors import (
    DESCRIPTOR_TARGET_CELLS,
    MECHANISM_FAMILIES,
    describe_program,
    mechanism_family,
)
from bias_optimizer.novelty.map_elites import MapElitesArchive
from bias_optimizer.novelty.tree_distance import structural_novelty

_ELITE_PARENTS = 5
_PROPOSAL_BATCH_SIZE = 4


class _ProgramEvaluator(Protocol):
    def evaluate(self, bias: ProgramBiasSpec): ...


class _ProgramProposer(Protocol):
    def propose(
        self,
        context: ProgramSearchContext,
        *,
        previous_biases: Iterable[ProgramBiasSpec] = (),
        count: int = 4,
    ) -> tuple[ProposedProgram, ...]: ...


class ProgramSearchEngine:
    """Run a constrained expression search with balanced mechanism proposals."""

    def __init__(
        self,
        *,
        track: SearchTrack = SearchTrack.DISCOVERY,
        max_feature_dim: int | None = None,
        evaluator: _ProgramEvaluator | None = None,
        proposer: _ProgramProposer | None = None,
        archive: MapElitesArchive | None = None,
        failure_path: Path | None = None,
        seed_programs: Iterable[ProgramBiasSpec] | None = None,
        seed_count: int = 20,
        seed: int = 42,
        max_proposal_rounds: int = 20,
        require_complete: bool = False,
    ) -> None:
        self.track = SearchTrack(track)
        self.max_feature_dim = max_feature_dim if max_feature_dim is not None else 128
        self.constraints = ProgramConstraints(
            track=self.track,
            max_feature_dim=self.max_feature_dim,
            forbid_raw_pixels=self.track is SearchTrack.AUGMENTATION,
            require_vector_root=self.track is SearchTrack.AUGMENTATION,
        )
        self._compiler = ProgramCompiler(self.constraints)
        self._evaluator = (
            evaluator
            if evaluator is not None
            else ProgramEvaluator(
                compiler=self._compiler,
                raw_pixel_anchor=self.track is SearchTrack.AUGMENTATION,
            )
        )
        self._proposer = proposer if proposer is not None else ProgramProposer()
        self._archive = archive if archive is not None else MapElitesArchive()
        self._failure_path = (
            Path(failure_path)
            if failure_path is not None
            else (
                self._archive.path.with_name(
                    f"{self._archive.path.stem}_failures.jsonl"
                )
                if self._archive.path is not None
                else None
            )
        )
        self._failure_count = (
            sum(
                1
                for line in self._failure_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if self._failure_path is not None and self._failure_path.exists()
            else 0
        )
        if type(max_proposal_rounds) is not int or max_proposal_rounds <= 0:
            raise ValueError("max_proposal_rounds must be positive")
        if type(require_complete) is not bool:
            raise TypeError("require_complete must be a boolean")
        self._max_proposal_rounds = max_proposal_rounds
        self._require_complete = require_complete
        if any(record.track != self.track.value for record in self._archive.records):
            raise ValueError("archive contains records from a different search track")
        if seed_programs is None:
            seeds = generate_seed_programs(
                seed_count,
                seed=seed,
                track=self.track,
                max_feature_dim=self.max_feature_dim,
            )
        else:
            seeds = tuple(seed_programs)
        self._seed_programs = tuple(seeds)
        if not all(isinstance(item, ProgramBiasSpec) for item in self._seed_programs):
            raise TypeError("seed_programs must contain ProgramBiasSpec values")

    @property
    def archive(self) -> MapElitesArchive:
        return self._archive

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def run(
        self,
        *,
        generations: int = 10,
        candidates_per_generation: int = 18,
    ) -> MapElitesArchive:
        if type(generations) is not int or generations < 0:
            raise ValueError("generations must be a non-negative integer")
        if type(candidates_per_generation) is not int or candidates_per_generation <= 0:
            raise ValueError("candidates_per_generation must be a positive integer")

        self._evaluate_seeds()
        for generation in range(1, generations + 1):
            existing = sum(
                record.generation == generation for record in self._archive.records
            )
            remaining = candidates_per_generation - existing
            if remaining > 0:
                self._run_generation(generation, remaining)
        if self._require_complete:
            underfilled = {
                generation: sum(
                    record.generation == generation for record in self._archive.records
                )
                for generation in range(1, generations + 1)
            }
            underfilled = {
                generation: count
                for generation, count in underfilled.items()
                if count < candidates_per_generation
            }
            if underfilled:
                detail = ", ".join(
                    f"generation {generation}: {count}/{candidates_per_generation}"
                    for generation, count in underfilled.items()
                )
                raise RuntimeError(
                    "search archive is incomplete and can be resumed: " + detail
                )
        return self._archive

    def _evaluate_seeds(self) -> None:
        for bias in self._seed_programs:
            signature = bias.program.to_json()
            if any(
                record.bias.program.to_json() == signature
                for record in self._archive.records
            ):
                continue
            self._evaluate_and_archive(bias, generation=0, parent_ids=())

    def _run_generation(self, generation: int, target_count: int) -> None:
        quotas = [target_count // len(MECHANISM_FAMILIES)] * len(MECHANISM_FAMILIES)
        for index in range(target_count % len(MECHANISM_FAMILIES)):
            quotas[index] += 1
        rounds_per_family = max(1, self._max_proposal_rounds // len(MECHANISM_FAMILIES))
        previous_biases = tuple(record.bias for record in self._archive.records)
        seen_programs = {
            record.bias.program.to_json() for record in self._archive.records
        }

        for family, quota in zip(MECHANISM_FAMILIES, quotas, strict=True):
            family_accepted = 0
            for _ in range(rounds_per_family):
                remaining = quota - family_accepted
                if remaining <= 0:
                    break
                parents = self._archive.top(_ELITE_PARENTS)
                target_cells = tuple(
                    cell
                    for cell in DESCRIPTOR_TARGET_CELLS
                    if self.track is SearchTrack.AUGMENTATION or cell[0] != "pixel"
                )
                context = ProgramSearchContext(
                    generation=generation,
                    track=self.track,
                    max_feature_dim=self.max_feature_dim,
                    top_candidates=parents,
                    underexplored_cells=self._archive.underexplored_cells(target_cells),
                    explored_programs=tuple(sorted(seen_programs)),
                    mechanism_focus=family,
                )
                try:
                    proposals = self._proposer.propose(
                        context,
                        previous_biases=previous_biases,
                        count=min(remaining, _PROPOSAL_BATCH_SIZE),
                    )
                except ProgramProposalError:
                    # Failed exchanges are durable; retry this family only.
                    continue
                parent_ids = tuple(record.candidate_id for record in parents)
                for proposal in proposals:
                    if not isinstance(proposal, ProposedProgram):
                        raise TypeError("proposer must return ProposedProgram values")
                    signature = proposal.bias.program.to_json()
                    if signature in seen_programs:
                        continue
                    seen_programs.add(signature)
                    actual_family = mechanism_family(proposal.bias.program)
                    if actual_family != family:
                        self._record_failure(
                            proposal.bias,
                            generation=generation,
                            mechanism_focus=family,
                            failure_type="mechanism_mismatch",
                            message=f"proposal classified as {actual_family}",
                        )
                        continue
                    try:
                        self._evaluate_and_archive(
                            proposal.bias,
                            generation=generation,
                            parent_ids=parent_ids,
                            prompt=proposal.prompt,
                            model=proposal.model,
                        )
                    except (
                        TypeError,
                        ValueError,
                        FloatingPointError,
                        OverflowError,
                    ) as exc:
                        self._record_failure(
                            proposal.bias,
                            generation=generation,
                            mechanism_focus=family,
                            failure_type="evaluation_error",
                            message=f"{type(exc).__name__}: {exc}",
                        )
                        continue
                    family_accepted += 1
                    if family_accepted >= quota:
                        break

    def _record_failure(
        self,
        bias: ProgramBiasSpec,
        *,
        generation: int,
        mechanism_focus: str,
        failure_type: str,
        message: str,
    ) -> None:
        record = ProgramSearchFailure(
            generation=generation,
            track=self.track.value,
            mechanism_focus=mechanism_focus,
            bias=bias,
            failure_type=failure_type,
            message=message[:1_000],
        )
        if self._failure_path is not None:
            self._failure_path.parent.mkdir(parents=True, exist_ok=True)
            with self._failure_path.open("a", encoding="utf-8") as stream:
                stream.write(record.to_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        self._failure_count += 1

    def _evaluate_and_archive(
        self,
        bias: ProgramBiasSpec,
        *,
        generation: int,
        parent_ids: tuple[str, ...],
        prompt: str | None = None,
        model: str | None = None,
    ) -> None:
        self._compiler.compile(bias.program)
        evaluation = self._evaluator.evaluate(bias)
        described_program = (
            add_raw_pixel_anchor(bias.program)
            if self.track is SearchTrack.AUGMENTATION
            else bias.program
        )
        descriptor = describe_program(described_program)
        novelty = structural_novelty(
            bias.program,
            tuple(record.bias.program for record in self._archive.records),
        )
        self._archive.add(
            ProgramSearchRecord(
                generation=generation,
                track=self.track.value,
                bias=bias,
                evaluation=evaluation,
                source=descriptor.source,
                order=descriptor.order,
                spatial=descriptor.spatial,
                composition=descriptor.composition,
                complexity=descriptor.complexity,
                novelty=novelty,
                parent_ids=parent_ids,
                prompt=prompt,
                model=model,
            )
        )
