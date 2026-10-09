"""Sequential search loop for evaluating LLM-proposed bias specifications."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Protocol

from numpy.typing import NDArray

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec, bias_spec_hash
from bias_optimizer.domain.search import FailureFeedback, SearchRecord
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.llm.prompts import (
    AblationEvidence,
    CandidateEvidence,
    ConfusionEvidence,
    PromptEvidence,
)
from bias_optimizer.llm.proposer import LLMProposer, ProposedBias
from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.search.archive import SearchArchive

_ELITE_SIZE = 5
_MAX_PROPOSAL_ROUNDS = 5
_MAX_FAILURE_PAIRS = 5


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
                record.generation == generation and record.record_type == "candidate"
                for record in self._archive.records
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
            self._evaluate_ablations(generation, parents)
            evidence = self._build_evidence(
                generation,
                parents,
                ablation_records=self._archive.records,
            )
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
                failure_feedback = self._failure_feedback(
                    proposal,
                    evidence.confusion_pairs,
                    evaluation.confusion_matrix,
                )
                self._archive.add(
                    SearchRecord(
                        generation=generation,
                        bias=proposal.bias,
                        evaluation=evaluation,
                        parent_ids=parent_ids,
                        prompt=proposal.prompt,
                        model=proposal.model,
                        proposal_category=proposal.category,
                        failure_feedback=failure_feedback,
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
        *,
        ablation_records: tuple[SearchRecord, ...] = (),
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
        best_confusions = parents[0].evaluation.confusion_matrix
        pairs = [
            (int(best_confusions[actual, predicted]), actual, predicted)
            for actual in range(10)
            for predicted in range(10)
            if actual != predicted and best_confusions[actual, predicted] > 0
        ]
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        confusion_pairs = tuple(
            ConfusionEvidence(actual, predicted, count)
            for count, actual, predicted in pairs[:_MAX_FAILURE_PAIRS]
        )
        parent_by_id = {record.candidate_id: record for record in parents}
        ablations = tuple(
            AblationEvidence(
                candidate_name=parent_by_id[record.base_candidate_id].bias.name,
                removed_operator=record.removed_operator,
                accuracy_with_operator=parent_by_id[
                    record.base_candidate_id
                ].evaluation.accuracy_500,
                accuracy_without_operator=record.evaluation.accuracy_500,
            )
            for record in ablation_records
            if record.record_type == "ablation"
            and record.base_candidate_id in parent_by_id
        )
        return PromptEvidence(
            generation=generation,
            top_candidates=candidates,
            confusion_pairs=confusion_pairs,
            ablations=ablations,
        )

    def _evaluate_ablations(
        self,
        generation: int,
        parents: tuple[SearchRecord, ...],
    ) -> None:
        """Evaluate each distinct one-operator removal for elite candidates."""
        for parent in parents:
            seen_operators: set[str] = set()
            for index, removed in enumerate(parent.bias.operators):
                signature = removed.to_json()
                if signature in seen_operators:
                    continue
                seen_operators.add(signature)
                remaining = tuple(
                    operator
                    for operator_index, operator in enumerate(parent.bias.operators)
                    if operator_index != index
                )
                if not remaining:
                    continue
                ablated_bias = self._make_ablation_bias(
                    parent.bias,
                    removed,
                    remaining,
                )
                if self._archive.contains(bias_spec_hash(ablated_bias)):
                    continue
                evaluation = self._evaluator.evaluate(ablated_bias)
                self._archive.add(
                    SearchRecord(
                        generation=generation,
                        bias=ablated_bias,
                        evaluation=evaluation,
                        parent_ids=(parent.candidate_id,),
                        record_type="ablation",
                        base_candidate_id=parent.candidate_id,
                        removed_operator=removed.name,
                    )
                )

    @staticmethod
    def _make_ablation_bias(
        bias: BiasSpec,
        removed: OperatorSpec,
        remaining: tuple[OperatorSpec, ...],
    ) -> BiasSpec:
        operator_suffix = hashlib.sha256(removed.to_json().encode("utf-8")).hexdigest()[
            :8
        ]
        return BiasSpec(
            name=f"{bias.name}_without_{removed.name}_{operator_suffix}",
            hypothesis=f"Ablation of {bias.name} without the {removed.name} operator.",
            operators=remaining,
            prediction="Measure whether removing this operator preserves accuracy.",
            falsification="Reject removal if validation accuracy falls materially.",
        )

    @staticmethod
    def _failure_feedback(
        proposal: ProposedBias,
        targets: tuple[ConfusionEvidence, ...],
        confusion_matrix: NDArray,
    ) -> tuple[FailureFeedback, ...]:
        if proposal.category != "failure-driven":
            return ()
        return tuple(
            FailureFeedback(
                actual_digit=pair.actual_digit,
                predicted_digit=pair.predicted_digit,
                baseline_count=pair.count,
                candidate_count=int(
                    confusion_matrix[pair.actual_digit, pair.predicted_digit]
                ),
            )
            for pair in targets
        )
