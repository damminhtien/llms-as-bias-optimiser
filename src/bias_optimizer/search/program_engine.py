"""LLM-guided typed-program search with raw-free and augmentation tracks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.seeds import generate_seed_programs
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.llm.program_proposer import (
    ProgramProposer,
    ProgramSearchContext,
    ProposedProgram,
)
from bias_optimizer.ml.evaluator import ProgramEvaluator
from bias_optimizer.novelty.descriptors import IMPLEMENTED_NICHES, describe_program
from bias_optimizer.novelty.map_elites import MapElitesArchive
from bias_optimizer.novelty.tree_distance import structural_novelty

_MAX_PROPOSAL_ROUNDS = 5
_ELITE_PARENTS = 5


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
    """Run a constrained expression search and maintain one elite per QD cell."""

    def __init__(
        self,
        *,
        track: SearchTrack = SearchTrack.DISCOVERY,
        max_feature_dim: int | None = None,
        evaluator: _ProgramEvaluator | None = None,
        proposer: _ProgramProposer | None = None,
        archive: MapElitesArchive | None = None,
        seed_programs: Iterable[ProgramBiasSpec] | None = None,
        seed_count: int = 20,
        seed: int = 42,
    ) -> None:
        self.track = SearchTrack(track)
        self.max_feature_dim = (
            max_feature_dim
            if max_feature_dim is not None
            else (128 if self.track is SearchTrack.DISCOVERY else 1_024)
        )
        self.constraints = ProgramConstraints(
            track=self.track,
            max_feature_dim=self.max_feature_dim,
        )
        self._compiler = ProgramCompiler(self.constraints)
        self._evaluator = (
            evaluator
            if evaluator is not None
            else ProgramEvaluator(compiler=self._compiler)
        )
        self._proposer = proposer if proposer is not None else ProgramProposer()
        self._archive = archive if archive is not None else MapElitesArchive()
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
        accepted = 0
        for _ in range(_MAX_PROPOSAL_ROUNDS):
            remaining = target_count - accepted
            if remaining <= 0:
                break
            parents = self._archive.top(_ELITE_PARENTS)
            target_niches = tuple(
                niche
                for niche in IMPLEMENTED_NICHES
                if self.track is SearchTrack.AUGMENTATION or niche != "raw_pixels"
            )
            context = ProgramSearchContext(
                generation=generation,
                track=self.track,
                max_feature_dim=self.max_feature_dim,
                top_candidates=parents,
                underexplored_niches=self._archive.underexplored_niches(target_niches),
                explored_programs=tuple(
                    record.bias.program.to_json()
                    for record in self._archive.records[-20:]
                ),
            )
            previous_biases = tuple(record.bias for record in self._archive.records)
            proposals = self._proposer.propose(
                context,
                previous_biases=previous_biases,
                count=remaining,
            )
            if not proposals:
                continue
            added_before = len(self._archive.records)
            parent_ids = tuple(record.candidate_id for record in parents)
            seen_programs = {
                record.bias.program.to_json() for record in self._archive.records
            }
            for proposal in proposals:
                if not isinstance(proposal, ProposedProgram):
                    raise TypeError("proposer must return ProposedProgram values")
                signature = proposal.bias.program.to_json()
                if signature in seen_programs:
                    continue
                seen_programs.add(signature)
                self._evaluate_and_archive(
                    proposal.bias,
                    generation=generation,
                    parent_ids=parent_ids,
                    prompt=proposal.prompt,
                    model=proposal.model,
                )
                accepted += 1
                if accepted >= target_count:
                    break
            if len(self._archive.records) == added_before:
                continue

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
        descriptor = describe_program(bias.program)
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
                niche=descriptor.primary_niche,
                complexity_bin=descriptor.complexity_bin,
                novelty=novelty,
                parent_ids=parent_ids,
                prompt=prompt,
                model=model,
            )
        )
