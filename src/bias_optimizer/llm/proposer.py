"""LLM proposer for four bounded, evidence-driven bias hypotheses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from bias_optimizer.domain.bias import BiasSpec
from bias_optimizer.llm.client import LLMClient
from bias_optimizer.llm.ollama_client import OllamaLLMClient
from bias_optimizer.llm.prompts import (
    PROPOSAL_CATEGORIES,
    PromptEvidence,
    build_proposer_prompt,
)
from bias_optimizer.llm.response_archive import JsonlResponseArchive
from bias_optimizer.llm.structured import StructuredBiasClient

ProposalCategory = Literal[
    "exploitation",
    "failure-driven",
    "simplification",
    "exploration",
]
_MAX_PRIOR_REPRESENTATIONS = 20


@dataclass(frozen=True, slots=True)
class ProposedBias:
    """One validated proposal with its category and model prompt metadata."""

    category: ProposalCategory
    bias: BiasSpec
    model: str
    prompt: str
    raw_response: str


class LLMProposer:
    """Turn compact search evidence into validated, deduplicated proposals."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        response_archive: JsonlResponseArchive | None = None,
        structured_client: StructuredBiasClient | None = None,
    ) -> None:
        if structured_client is not None and client is not None:
            raise ValueError("pass either client or structured_client, not both")
        if structured_client is not None:
            self._structured_client = structured_client
        else:
            llm_client = (
                client if client is not None else OllamaLLMClient.from_environment()
            )
            archive = (
                response_archive
                if response_archive is not None
                else JsonlResponseArchive()
            )
            self._structured_client = StructuredBiasClient(llm_client, archive=archive)

    def propose(
        self,
        evidence: PromptEvidence,
        *,
        previous_biases: Iterable[BiasSpec] = (),
        count: int = len(PROPOSAL_CATEGORIES),
    ) -> tuple[ProposedBias, ...]:
        """Generate categorized proposals from summarized search history."""
        if not isinstance(evidence, PromptEvidence):
            raise TypeError("LLMProposer.propose requires PromptEvidence")
        if type(count) is not int or count <= 0:
            raise ValueError("count must be a positive integer")
        previous = tuple(previous_biases)
        if not all(isinstance(bias, BiasSpec) for bias in previous):
            raise TypeError("previous_biases must contain BiasSpec values")

        explored = list(evidence.explored_representations)
        seen_representations = set(explored)
        for bias in previous[-_MAX_PRIOR_REPRESENTATIONS:]:
            representation = _representation_summary(bias)
            if representation not in seen_representations:
                explored.append(representation)
                seen_representations.add(representation)
        bounded_evidence = PromptEvidence(
            generation=evidence.generation,
            top_candidates=evidence.top_candidates,
            confusion_pairs=evidence.confusion_pairs,
            ablations=evidence.ablations,
            explored_representations=tuple(explored[-_MAX_PRIOR_REPRESENTATIONS:]),
        )
        prompt = build_proposer_prompt(bounded_evidence, proposal_count=count)
        response = self._structured_client.generate(
            prompt,
            expected_count=count,
        )
        final_attempt = response.records[-1]
        seen = {_representation_hash(bias) for bias in previous}
        proposals: list[ProposedBias] = []
        categories = tuple(
            PROPOSAL_CATEGORIES[index % len(PROPOSAL_CATEGORIES)]
            for index in range(count)
        )
        for category, bias in zip(categories, response.proposals, strict=True):
            signature = _representation_hash(bias)
            if signature in seen:
                continue
            seen.add(signature)
            proposals.append(
                ProposedBias(
                    category=category,
                    bias=bias,
                    model=response.model,
                    prompt=final_attempt.prompt,
                    raw_response=final_attempt.response,
                )
            )
        return tuple(proposals)


def _representation_hash(bias: BiasSpec) -> str:
    operators = sorted(
        json.dumps(operator.to_dict(), sort_keys=True, separators=(",", ":"))
        for operator in bias.operators
    )
    canonical = json.dumps(operators, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _representation_summary(bias: BiasSpec) -> str:
    operators = sorted(
        json.dumps(operator.to_dict(), sort_keys=True, separators=(",", ":"))
        for operator in bias.operators
    )
    return json.dumps(
        [json.loads(operator) for operator in operators],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
