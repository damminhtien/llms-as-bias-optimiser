"""Tests for validated, categorized, deduplicated LLM proposals."""

import json

import pytest

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.llm import JsonlResponseArchive, LLMProposer, MockLLMClient
from bias_optimizer.llm.prompts import (
    PROPOSAL_CATEGORIES,
    CandidateEvidence,
    PromptEvidence,
)


def _bias(name: str, operators: tuple[str, ...]) -> BiasSpec:
    return BiasSpec(
        name=name,
        hypothesis=f"Hypothesis for {name}.",
        operators=tuple(OperatorSpec(operator) for operator in operators),
        prediction="Validation accuracy should improve.",
        falsification="Reject if validation accuracy does not improve.",
    )


def _evidence() -> PromptEvidence:
    return PromptEvidence(
        generation=1,
        top_candidates=(
            CandidateEvidence(
                name="topology_seed",
                hypothesis="Topology captures loops and connected parts.",
                operators=("topology",),
                accuracy_500=0.42,
                accuracy_5000=0.41,
                feature_dim=4,
            ),
        ),
    )


def _response(*biases: BiasSpec) -> str:
    return json.dumps({"proposals": [bias.to_dict() for bias in biases]})


def test_proposer_returns_four_validated_categories_and_archives_response(
    tmp_path,
) -> None:
    biases = (
        _bias("exploit", ("topology", "spatial")),
        _bias("failure", ("topology", "stroke_direction")),
        _bias("simplify", ("spatial",)),
        _bias("explore", ("raw_pixels", "curvature")),
    )
    raw = _response(*biases)
    client = MockLLMClient(raw)
    archive = JsonlResponseArchive(tmp_path / "responses.jsonl")
    proposer = LLMProposer(client, response_archive=archive)

    proposals = proposer.propose(_evidence())

    assert tuple(proposal.category for proposal in proposals) == PROPOSAL_CATEGORIES
    assert tuple(proposal.bias for proposal in proposals) == biases
    assert all(proposal.model == "unspecified" for proposal in proposals)
    assert all(
        "Validation evidence (search/validation split only)" in proposal.prompt
        for proposal in proposals
    )
    assert all(proposal.raw_response == raw for proposal in proposals)
    assert len(client.prompts) == 1
    assert len(archive.path.read_text(encoding="utf-8").splitlines()) == 1


def test_proposer_deduplicates_by_representation_against_history_and_batch() -> None:
    existing = _bias("existing_name", ("topology", "spatial"))
    reversed_operators = _bias("same_representation", ("spatial", "topology"))
    biases = (
        reversed_operators,
        _bias("unique_failure", ("topology", "stroke_direction")),
        _bias("duplicate_in_batch", ("stroke_direction", "topology")),
        _bias("unique_exploration", ("raw_pixels", "curvature")),
    )
    proposer = LLMProposer(MockLLMClient(_response(*biases)))

    proposals = proposer.propose(_evidence(), previous_biases=(existing,))

    assert tuple(proposal.category for proposal in proposals) == (
        "failure-driven",
        "exploration",
    )
    assert tuple(proposal.bias.name for proposal in proposals) == (
        "unique_failure",
        "unique_exploration",
    )
    prompt_payload = json.loads(
        proposals[0].prompt.split(
            "Validation evidence (search/validation split only):\n", 1
        )[1]
    )
    assert prompt_payload["explored_representations"] == [
        '[{"name":"spatial","params":{}},{"name":"topology","params":{}}]'
    ]


def test_proposer_supports_five_candidates_with_cycled_categories() -> None:
    biases = (
        _bias("exploit", ("topology", "spatial")),
        _bias("failure", ("topology", "stroke_direction")),
        _bias("simplify", ("spatial",)),
        _bias("explore", ("raw_pixels", "curvature")),
        _bias("second_exploitation", ("symmetry",)),
    )
    client = MockLLMClient(_response(*biases))
    proposer = LLMProposer(client)

    proposals = proposer.propose(_evidence(), count=5)

    assert tuple(proposal.category for proposal in proposals) == (
        "exploitation",
        "failure-driven",
        "simplification",
        "exploration",
        "exploitation",
    )
    assert "exactly 5 candidates" in client.prompts[0]


def test_proposer_rejects_invalid_count() -> None:
    proposer = LLMProposer(MockLLMClient("unused"))

    with pytest.raises(ValueError, match="count"):
        proposer.propose(_evidence(), count=0)


def test_proposer_rejects_non_compact_evidence_and_invalid_history() -> None:
    proposer = LLMProposer(
        MockLLMClient(
            _response(
                *(
                    _bias(f"bias_{index}", (operator,))
                    for index, operator in enumerate(
                        ("topology", "spatial", "curvature", "raw_pixels")
                    )
                )
            )
        )
    )

    with pytest.raises(TypeError, match="PromptEvidence"):
        proposer.propose(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="BiasSpec"):
        proposer.propose(_evidence(), previous_biases=(object(),))  # type: ignore[arg-type]
