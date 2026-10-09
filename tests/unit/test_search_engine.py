"""Tests for the sequential MVP search engine."""

from collections.abc import Iterable

import numpy as np
import pytest

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.llm.prompts import PROPOSAL_CATEGORIES, PromptEvidence
from bias_optimizer.llm.proposer import ProposedBias
from bias_optimizer.search.archive import SearchArchive
from bias_optimizer.search.engine import SearchEngine


def _bias(name: str, operators: tuple[str, ...]) -> BiasSpec:
    return BiasSpec(
        name=name,
        hypothesis=f"Hypothesis for {name}.",
        operators=tuple(OperatorSpec(operator) for operator in operators),
        prediction="Validation accuracy should improve.",
        falsification="Reject if validation accuracy does not improve.",
    )


def _proposals(*biases: BiasSpec) -> tuple[ProposedBias, ...]:
    return tuple(
        ProposedBias(
            category=PROPOSAL_CATEGORIES[index % len(PROPOSAL_CATEGORIES)],
            bias=bias,
            model="mock-model",
            prompt="bounded validation evidence",
            raw_response="{}",
        )
        for index, bias in enumerate(biases)
    )


class _FakeEvaluator:
    def __init__(self) -> None:
        self.evaluated: list[BiasSpec] = []

    def evaluate(self, bias: BiasSpec) -> Evaluation:
        self.evaluated.append(bias)
        is_ablation = "_without_" in bias.name
        accuracy = 0.6 if is_ablation else min(0.99, 0.7 + len(self.evaluated) / 100)
        confusion_count = 9 if is_ablation else max(0, 10 - len(self.evaluated))
        confusion_matrix = np.eye(10, dtype=np.int64) * 20
        confusion_matrix[3, 3] -= confusion_count
        confusion_matrix[3, 5] = confusion_count
        return Evaluation(
            accuracy_500=accuracy,
            accuracy_5000=min(0.99, accuracy + 0.05),
            feature_dim=4,
            feature_runtime_ms=1.0,
            training_runtime_ms=2.0,
            inference_runtime_ms=1.0,
            confusion_matrix=confusion_matrix,
        )


class _FakeProposer:
    def __init__(self, *batches: tuple[ProposedBias, ...]) -> None:
        self._batches = list(batches)
        self.calls: list[tuple[PromptEvidence, tuple[BiasSpec, ...], int]] = []

    def propose(
        self,
        evidence: PromptEvidence,
        *,
        previous_biases: Iterable[BiasSpec] = (),
        count: int = 4,
    ) -> tuple[ProposedBias, ...]:
        self.calls.append((evidence, tuple(previous_biases), count))
        if not self._batches:
            return ()
        return self._batches.pop(0)


def _first_generation() -> tuple[BiasSpec, ...]:
    return (
        _bias("candidate_symmetry", ("symmetry",)),
        _bias("candidate_curvature_symmetry", ("curvature", "symmetry")),
        _bias("candidate_direction", ("stroke_direction",)),
        _bias("candidate_spatial_symmetry", ("spatial", "symmetry")),
        _bias("candidate_raw_topology", ("raw_pixels", "topology")),
    )


def _second_generation() -> tuple[BiasSpec, ...]:
    return (
        _bias("candidate_skeleton", ("skeleton_graph",)),
        _bias("candidate_curvature_direction", ("curvature", "stroke_direction")),
        _bias("candidate_curvature_spatial", ("curvature", "spatial")),
        _bias("candidate_direction_spatial", ("stroke_direction", "spatial")),
        _bias("candidate_symmetry_direction", ("symmetry", "stroke_direction")),
    )


def test_search_engine_evaluates_five_seeds_and_five_candidates_per_round(
    tmp_path,
) -> None:
    evaluator = _FakeEvaluator()
    proposer = _FakeProposer(
        _proposals(*_first_generation()),
        _proposals(*_second_generation()),
    )
    archive = SearchArchive(tmp_path / "search.jsonl")
    engine = SearchEngine(evaluator=evaluator, proposer=proposer, archive=archive)

    result = engine.run(generations=2, candidates_per_generation=5)

    assert result is archive
    candidate_records = [
        record for record in archive.records if record.record_type == "candidate"
    ]
    assert len(candidate_records) == 15
    assert len(evaluator.evaluated) > len(candidate_records)
    assert [
        sum(
            record.generation == generation and record.record_type == "candidate"
            for record in archive.records
        )
        for generation in range(3)
    ] == [5, 5, 5]
    assert [call[2] for call in proposer.calls] == [5, 5]
    assert len(proposer.calls[0][1]) == 12
    assert len(proposer.calls[1][1]) > len(proposer.calls[0][1])
    assert all(len(call[0].top_candidates) == 5 for call in proposer.calls)
    assert proposer.calls[0][0].ablations
    assert (3, 5, 5) == (
        proposer.calls[0][0].confusion_pairs[0].actual_digit,
        proposer.calls[0][0].confusion_pairs[0].predicted_digit,
        proposer.calls[0][0].confusion_pairs[0].count,
    )
    generated = [
        record
        for record in archive.records
        if record.generation > 0 and record.record_type == "candidate"
    ]
    assert all(len(record.parent_ids) == 5 for record in generated)
    assert all(record.model == "mock-model" for record in generated)
    failure_candidates = [
        record for record in generated if record.proposal_category == "failure-driven"
    ]
    assert failure_candidates
    assert failure_candidates[0].failure_feedback[0].actual_digit == 3
    assert failure_candidates[0].failure_feedback[0].predicted_digit == 5
    assert failure_candidates[0].failure_feedback[0].improvement > 0
    assert all(
        record.failure_feedback == ()
        for record in generated
        if record.proposal_category != "failure-driven"
    )
    assert len(archive.top(5)) == 5


def test_search_engine_skips_seed_and_duplicate_candidates(tmp_path) -> None:
    seeds = initial_human_biases()
    unique_biases = _first_generation()[:4]
    proposer = _FakeProposer(
        _proposals(
            seeds[1],
            unique_biases[0],
            unique_biases[1],
            unique_biases[2],
            unique_biases[3],
        ),
        _proposals(_bias("candidate_transition", ("direction_transition",))),
    )
    evaluator = _FakeEvaluator()
    archive = SearchArchive(tmp_path / "search.jsonl")

    result = SearchEngine(
        evaluator=evaluator,
        proposer=proposer,
        archive=archive,
        seed_biases=seeds,
    ).run(generations=1, candidates_per_generation=5)

    candidate_records = [
        record for record in result.records if record.record_type == "candidate"
    ]
    assert len(candidate_records) == 10
    assert len(evaluator.evaluated) > 10
    assert sum(record.bias == seeds[1] for record in result.records) == 1
    assert (
        sum(
            record.generation == 1 and record.record_type == "candidate"
            for record in result.records
        )
        == 5
    )


def test_search_engine_resume_is_idempotent_for_completed_generations(tmp_path) -> None:
    evaluator = _FakeEvaluator()
    proposer = _FakeProposer(_proposals(*_first_generation()))
    engine = SearchEngine(
        evaluator=evaluator,
        proposer=proposer,
        archive=SearchArchive(tmp_path / "search.jsonl"),
    )

    archive = engine.run(generations=1, candidates_per_generation=5)
    initial_ids = tuple(record.candidate_id for record in archive.records)
    evaluated_count = len(evaluator.evaluated)
    resumed = engine.run(generations=1, candidates_per_generation=5)

    assert tuple(record.candidate_id for record in resumed.records) == initial_ids
    assert len(evaluator.evaluated) == evaluated_count
    assert len(proposer.calls) == 1


@pytest.mark.parametrize(
    ("generations", "count", "message"),
    [(-1, 5, "generations"), (1, 0, "candidates_per_generation")],
)
def test_search_engine_rejects_invalid_run_sizes(
    tmp_path, generations: int, count: int, message: str
) -> None:
    engine = SearchEngine(
        evaluator=_FakeEvaluator(),
        proposer=_FakeProposer(),
        archive=SearchArchive(tmp_path / "search.jsonl"),
    )

    with pytest.raises(ValueError, match=message):
        engine.run(generations=generations, candidates_per_generation=count)
