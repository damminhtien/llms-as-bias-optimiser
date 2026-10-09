"""Balanced mechanism request scheduling for the V3 search controller."""

from collections import Counter

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.validator import SearchTrack
from bias_optimizer.llm.program_proposer import (
    ProgramSearchContext,
    ProposedProgram,
    _program_output_schema,
    build_program_prompt,
)
from bias_optimizer.novelty.descriptors import (
    MECHANISM_FAMILIES,
    describe_program,
    mechanism_family,
)
from bias_optimizer.novelty.map_elites import MapElitesArchive
from bias_optimizer.search.program_engine import ProgramSearchEngine


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _paths() -> Expr:
    return _node("paths", _node("graph", _node("skeletonize", _node("image"))))


def _program_for(family: str) -> Expr:
    if family == "order_sensitive":
        return _node(
            "autocorrelation",
            _node(
                "degree_sequence", _node("graph", _node("skeletonize", _node("image")))
            ),
        )
    if family == "spatial_relational":
        return _node(
            "spatial_condition",
            _node(
                "degree_sequence", _node("graph", _node("skeletonize", _node("image")))
            ),
        )
    if family == "graph_relational":
        return _node(
            "cross_histogram",
            _node("path_summary", _paths(), measure="branch_endpoints"),
            _node("path_summary", _paths(), measure="centroid_y"),
        )
    return _node("histogram", _node("angles", _paths()), bins=6)


class _Evaluator:
    def evaluate(self, _bias: ProgramBiasSpec) -> Evaluation:
        return Evaluation(
            accuracy_500=0.7,
            accuracy_5000=0.8,
            feature_dim=4,
            feature_runtime_ms=1,
            training_runtime_ms=1,
            inference_runtime_ms=1,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        )


class _Proposer:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def propose(
        self,
        context: ProgramSearchContext,
        *,
        previous_biases=(),
        count: int = 4,
    ) -> tuple[ProposedProgram, ...]:
        assert count == 1
        family = context.mechanism_focus
        self.requested.append(family)
        bias = ProgramBiasSpec(
            name=f"candidate_{family}",
            hypothesis="A bounded structural program may distinguish visual categories.",
            mechanism=f"The representation tests the {family} mechanism.",
            program=_program_for(family),
            prediction="The mechanism should improve held-out validation accuracy.",
            falsification="Reject it if the matched mechanism intervention has no effect.",
        )
        return (ProposedProgram(bias, "test-model", "test prompt", "{}"),)


class _OneMismatchProposer(_Proposer):
    def __init__(self) -> None:
        super().__init__()
        self._sent_mismatch = False

    def propose(
        self,
        context: ProgramSearchContext,
        *,
        previous_biases=(),
        count: int = 4,
    ) -> tuple[ProposedProgram, ...]:
        if context.mechanism_focus == "order_sensitive" and not self._sent_mismatch:
            self._sent_mismatch = True
            self.requested.append(context.mechanism_focus)
            bias = ProgramBiasSpec(
                name="off_family_spatial_program",
                hypothesis="A location-conditioned shape cue may distinguish classes.",
                mechanism="The event distribution changes with image height.",
                program=_node("spatial_condition", _node("angles", _paths())),
                prediction="A location-aware feature may improve validation accuracy.",
                falsification="Reject it if spatial relocation does not alter the output.",
            )
            return (ProposedProgram(bias, "test-model", "test prompt", "{}"),)
        return super().propose(context, previous_biases=previous_biases, count=count)


def test_generation_requests_all_four_mechanism_families() -> None:
    proposer = _Proposer()
    engine = ProgramSearchEngine(
        track=SearchTrack.DISCOVERY,
        evaluator=_Evaluator(),
        proposer=proposer,
        archive=MapElitesArchive(),
        seed_programs=(),
        max_proposal_rounds=4,
        require_complete=True,
    )

    archive = engine.run(generations=1, candidates_per_generation=4)
    generated = tuple(record for record in archive.records if record.generation == 1)

    assert Counter(proposer.requested) == Counter(MECHANISM_FAMILIES)
    assert Counter(
        mechanism_family(record.bias.program) for record in generated
    ) == Counter(MECHANISM_FAMILIES)


def test_search_record_round_trips_500_only_screening_evaluation() -> None:
    program = _node("cycle_rank", _node("graph", _node("skeletonize", _node("image"))))
    descriptor = describe_program(program)
    bias = ProgramBiasSpec(
        name="screened_candidate",
        hypothesis="A compact graph signal may separate handwritten classes.",
        mechanism="Cycle structure can be represented without image intensities.",
        program=program,
        prediction="The topology statistic should exceed a matched compact baseline.",
        falsification="Reject it if the held-out validation score does not improve.",
    )
    record = ProgramSearchRecord(
        generation=1,
        track="discovery",
        bias=bias,
        evaluation=Evaluation(
            accuracy_500=0.8,
            accuracy_5000=None,
            feature_dim=1,
            feature_runtime_ms=1,
            training_runtime_ms=1,
            inference_runtime_ms=1,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        ),
        source=descriptor.source,
        order=descriptor.order,
        spatial=descriptor.spatial,
        composition=descriptor.composition,
        complexity=descriptor.complexity,
        novelty=1.0,
    )

    restored = ProgramSearchRecord.from_json(record.to_json())

    assert restored.evaluation.accuracy_5000 is None
    assert restored.evaluation.accuracy_500 == 0.8


def test_search_rejects_off_family_candidates_and_persists_failure(tmp_path) -> None:
    failure_path = tmp_path / "failures.jsonl"
    engine = ProgramSearchEngine(
        track=SearchTrack.DISCOVERY,
        evaluator=_Evaluator(),
        proposer=_OneMismatchProposer(),
        archive=MapElitesArchive(),
        failure_path=failure_path,
        seed_programs=(),
        max_proposal_rounds=8,
        require_complete=True,
    )

    archive = engine.run(generations=1, candidates_per_generation=4)

    assert len(archive.records) == 4
    assert engine.failure_count == 1
    assert '"failure_type":"mechanism_mismatch"' in failure_path.read_text()


def test_proposer_prompt_receives_descriptor_cells_and_mechanism_focus() -> None:
    cell = ("path_geometry", "higher_order", "localized", "single", "deep")
    context = ProgramSearchContext(
        generation=2,
        track=SearchTrack.DISCOVERY,
        max_feature_dim=128,
        top_candidates=(),
        underexplored_cells=(cell,),
        mechanism_focus="spatial_relational",
    )

    prompt = build_program_prompt(context, 1)

    assert "Requested mechanism family for this batch: spatial_relational" in prompt
    assert "Every proposal must use spatial_condition" in prompt
    assert "underexplored_cells" in prompt
    assert '"path_geometry","higher_order","localized","single","deep"' in prompt

    free_context = ProgramSearchContext(
        generation=2,
        track=SearchTrack.DISCOVERY,
        max_feature_dim=128,
        top_candidates=(),
        underexplored_cells=(cell,),
        mechanism_focus="free_exploration",
    )
    free_prompt = build_program_prompt(free_context, 1)

    assert "Use a global, order-insensitive mechanism" in free_prompt
    assert "Do not use delta" in free_prompt


def test_augmentation_proposal_schema_only_accepts_structural_vectors() -> None:
    schema = _program_output_schema(1, track=SearchTrack.AUGMENTATION)
    serialized = str(schema)

    assert "flatten_pixels" not in serialized
    assert schema["$defs"]["program"]["anyOf"] == [{"$ref": "#/$defs/expr_vector_6"}]
