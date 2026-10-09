"""Tests for compact, constrained proposer prompts."""

import json

import pytest

from bias_optimizer.features.registry import OPERATOR_NAMES
from bias_optimizer.llm.prompts import (
    PROPOSAL_CATEGORIES,
    AblationEvidence,
    CandidateEvidence,
    ConfusionEvidence,
    PromptEvidence,
    build_proposer_prompt,
)


def _evidence() -> PromptEvidence:
    return PromptEvidence(
        generation=2,
        top_candidates=(
            CandidateEvidence(
                name="topology_spatial",
                hypothesis="Vertical location complements topology.",
                operators=("topology", "spatial"),
                accuracy_500=0.74,
                accuracy_5000=0.89,
                feature_dim=7,
            ),
        ),
        confusion_pairs=(ConfusionEvidence(3, 5, 17), ConfusionEvidence(4, 9, 12)),
        ablations=(AblationEvidence("topology_spatial", "spatial", 0.74, 0.70),),
    )


def test_prompt_covers_objective_constraints_evidence_and_schema() -> None:
    prompt = build_proposer_prompt(_evidence())
    payload = json.loads(
        prompt.split("Validation evidence (search/validation split only):\n", 1)[1]
    )

    assert "low-data MNIST" in prompt
    assert "StandardScaler" in prompt and "LogisticRegression(C=1.0" in prompt
    assert ", ".join(OPERATOR_NAMES) in prompt
    assert "only topology accepts a parameter" in prompt
    assert "all other operators require an empty params object" in prompt
    assert payload["confusion_pairs"][0] == {
        "actual_digit": 3,
        "predicted_digit": 5,
        "count": 17,
    }
    assert payload["ablations"][0]["accuracy_delta"] == pytest.approx(0.04)
    assert "Prefer conceptual changes over simply adding more features" in prompt
    assert "Return JSON only" in prompt
    assert '"proposals"' in prompt


def test_prompt_requests_the_four_candidate_categories_in_order() -> None:
    prompt = build_proposer_prompt(_evidence())
    positions = [prompt.index(category) for category in PROPOSAL_CATEGORIES]

    assert positions == sorted(positions)
    assert len(PROPOSAL_CATEGORIES) == 4


def test_prompt_cycles_categories_to_support_five_candidate_generations() -> None:
    prompt = build_proposer_prompt(_evidence(), proposal_count=5)

    assert "Produce exactly 5 candidates" in prompt
    assert "1. exploitation" in prompt
    assert "2. failure-driven" in prompt
    assert "3. simplification" in prompt
    assert "4. exploration" in prompt
    assert "5. exploitation" in prompt


@pytest.mark.parametrize("proposal_count", [0, -1, True])
def test_prompt_rejects_invalid_proposal_count(proposal_count: int) -> None:
    with pytest.raises(ValueError, match="proposal_count"):
        build_proposer_prompt(_evidence(), proposal_count=proposal_count)


def test_prompt_contains_only_bounded_summary_evidence() -> None:
    evidence = _evidence()
    candidates = tuple(
        CandidateEvidence(
            name=f"candidate_{index}",
            hypothesis=f"summary {index}",
            operators=("topology",),
            accuracy_500=0.5,
            accuracy_5000=0.6,
            feature_dim=4,
        )
        for index in range(20)
    )
    prompt = build_proposer_prompt(
        PromptEvidence(
            generation=evidence.generation,
            top_candidates=candidates,
            confusion_pairs=tuple(
                ConfusionEvidence(3, 5, count) for count in range(1, 20)
            ),
            ablations=tuple(
                AblationEvidence("candidate", "spatial", 0.7, 0.6) for _ in range(20)
            ),
            explored_representations=tuple(
                f"representation_{index}" for index in range(30)
            ),
        )
    )
    payload_text = prompt.split(
        "Validation evidence (search/validation split only):\n", 1
    )[1]
    payload = json.loads(payload_text)

    assert len(payload["top_candidates"]) == 5
    assert len(payload["confusion_pairs"]) == 5
    assert len(payload["ablations"]) == 10
    assert len(payload["explored_representations"]) == 20
    assert payload["explored_representations"][0] == "representation_10"
    assert "train_images" not in payload and "test_images" not in payload


def test_evidence_rejects_invalid_measurements() -> None:
    with pytest.raises(ValueError, match="accuracy_500"):
        CandidateEvidence("candidate", "hypothesis", ("topology",), 1.1, 0.9, 4)
    with pytest.raises(ValueError, match="incorrect pair"):
        ConfusionEvidence(3, 3, 2)
    with pytest.raises(ValueError, match="positive"):
        ConfusionEvidence(3, 5, 0)
    with pytest.raises(ValueError, match="accuracy_with_operator"):
        AblationEvidence("candidate", "spatial", float("nan"), 0.5)
