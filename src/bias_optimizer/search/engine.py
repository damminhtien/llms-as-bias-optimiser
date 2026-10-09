"""Sequential search loop for evaluating LLM-proposed bias specifications."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from bias_optimizer.domain.bias import BiasSpec, bias_spec_hash
from bias_optimizer.domain.search import SearchRecord
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.llm.prompts import CandidateEvidence, PromptEvidence
from bias_optimizer.llm.proposer import LLMProposer, ProposedBias
from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.search.archive import SearchArchive

_ELITE_SIZE = 5
_MAX_PROPOSAL_ROUNDS = 5


class _Proposer(Protocol):
    def propose(
        self,
        evidence: PromptEvidence,
        *,
        previous_biases: Iterable[BiasSpec] = (),
        count: int = 4,
    ) -> tuple[ProposedBias, ...]: ...


class SearchEngine:
    """Evaluate the five fixed seeds, then evolve the best candidates in rounds."""

    def __init__(
        self,
        *,
        evaluator: Evaluator | None = None,
        proposer: _Proposer | None = None,
        archive: SearchArchive | None = None,
        seed_biases: Iterable[BiasSpec] | None = None,
    ) -> None:
        self._evaluator = evaluator if evaluator is not None else Evaluator()
        self._proposer = proposer if proposer is not None else LLMProposer()
        self._archive = archive if archive is not None else SearchArchive()
        seeds = initial_human_biases() if seed_biases is None else tuple(seed_biases)
        self._seed_biases = tuple(seeds)
        if not self._seed_biases or not all(
            isinstance(bias, BiasSpec) for bias in self._seed_biases
        ):
            raise ValueError("seed_biases must contain at least one BiasSpec")

    @property
    def archive(self) -> SearchArchive:
        return self._archive

    def run(
        self,
        *,
        generations: int = 5,
        candidates_per_generation: int = 5,
    ) -> SearchArchive:
        """Run sequential search rounds and return the durable archive."""
        if type(generations) is not int or generations < 0:
            raise ValueError("generations must be a non-negative integer")
        if type(candidates_per_generation) is not int or candidates_per_generation <= 0:
            raise ValueError("candidates_per_generation must be a positive integer")

        self._evaluate_seeds()
        for generation in range(1, generations + 1):
            existing_count = sum(
                record.generation == generation for record in self._archive.records
            )
            remaining = candidates_per_generation - existing_count
            if remaining > 0:
                self._run_generation(generation, remaining)
        return self._archive

    def _evaluate_seeds(self) -> None:
        for bias in self._seed_biases:
            if self._archive.contains(bias_spec_hash(bias)):
                continue
            evaluation = self._evaluator.evaluate(bias)
            self._archive.add(
                SearchRecord(generation=0, bias=bias, evaluation=evaluation)
            )

    def _run_generation(self, generation: int, candidate_count: int) -> None:
        accepted = 0
        for _ in range(_MAX_PROPOSAL_ROUNDS):
            remaining = candidate_count - accepted
            parents = self._archive.top(_ELITE_SIZE)
            if remaining <= 0 or not parents:
                break
            evidence = self._build_evidence(generation, parents)
            proposals = self._proposer.propose(
                evidence,
                previous_biases=(record.bias for record in self._archive.records),
                count=remaining,
            )
            if not proposals:
                break

            added_before = len(self._archive.records)
            parent_ids = tuple(record.candidate_id for record in parents)
            for proposal in proposals:
                if not isinstance(proposal, ProposedBias):
                    raise TypeError("proposer must return ProposedBias values")
                candidate_id = bias_spec_hash(proposal.bias)
                if self._archive.contains(candidate_id):
                    continue
                evaluation = self._evaluator.evaluate(proposal.bias)
                self._archive.add(
                    SearchRecord(
                        generation=generation,
                        bias=proposal.bias,
                        evaluation=evaluation,
                        parent_ids=parent_ids,
                        prompt=proposal.prompt,
                        model=proposal.model,
                    )
                )
                accepted += 1
                if accepted >= candidate_count:
                    break
            if len(self._archive.records) == added_before:
                break

    @staticmethod
    def _build_evidence(
        generation: int,
        parents: tuple[SearchRecord, ...],
    ) -> PromptEvidence:
        candidates = tuple(
            CandidateEvidence(
                name=record.bias.name,
                hypothesis=record.bias.hypothesis,
                operators=tuple(operator.name for operator in record.bias.operators),
                accuracy_500=record.evaluation.accuracy_500,
                accuracy_5000=record.evaluation.accuracy_5000,
                feature_dim=record.evaluation.feature_dim,
            )
            for record in parents
        )
        return PromptEvidence(generation=generation, top_candidates=candidates)
