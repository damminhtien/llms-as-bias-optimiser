"""JSON parsing, registry validation, and one-retry structured generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from bias_optimizer.compiler.bias_compiler import BiasCompiler
from bias_optimizer.domain.bias import BiasSpec
from bias_optimizer.llm.client import LLMClient
from bias_optimizer.llm.response_archive import (
    JsonlResponseArchive,
    LLMResponseRecord,
)


class StructuredOutputError(ValueError):
    """Raised when both structured-output attempts fail schema validation."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True, slots=True)
class StructuredBiasResponse:
    """Validated candidates with the exact prompts and raw responses."""

    proposals: tuple[BiasSpec, ...]
    records: tuple[LLMResponseRecord, ...]

    @property
    def model(self) -> str:
        return self.records[-1].model


def parse_bias_response(
    raw_response: str,
    *,
    compiler: BiasCompiler | None = None,
) -> tuple[BiasSpec, ...]:
    """Parse a proposals object and validate each candidate against the registry."""
    if not isinstance(raw_response, str):
        raise TypeError("LLM response must be a string")
    payload = json.loads(raw_response)
    if not isinstance(payload, dict) or set(payload) != {"proposals"}:
        raise ValueError("response must be a JSON object containing only proposals")
    raw_proposals = payload["proposals"]
    if not isinstance(raw_proposals, list) or not raw_proposals:
        raise ValueError("proposals must be a non-empty JSON array")

    specs = tuple(BiasSpec.from_dict(item) for item in raw_proposals)
    validator = compiler if compiler is not None else BiasCompiler()
    for spec in specs:
        validator.compile(spec)
    return specs


@dataclass(frozen=True, slots=True)
class StructuredBiasClient:
    """Generate schema-valid bias proposals with exactly one repair attempt."""

    client: LLMClient
    archive: JsonlResponseArchive = field(default_factory=JsonlResponseArchive)
    compiler: BiasCompiler = field(default_factory=BiasCompiler)

    def generate(
        self,
        prompt: str,
        *,
        expected_count: int | None = None,
    ) -> StructuredBiasResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if expected_count is not None and (
            type(expected_count) is not int or expected_count <= 0
        ):
            raise ValueError("expected_count must be positive")

        model_value = getattr(self.client, "model", "unspecified")
        model = model_value if isinstance(model_value, str) else "unspecified"
        records: list[LLMResponseRecord] = []
        last_raw_response = ""
        for attempt in (1, 2):
            request_prompt = prompt if attempt == 1 else self._retry_prompt(prompt)
            raw_response = self.client.generate(request_prompt)
            last_raw_response = raw_response
            usage = getattr(self.client, "last_usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            response_tokens = getattr(usage, "response_tokens", None)
            try:
                proposals = parse_bias_response(
                    raw_response,
                    compiler=self.compiler,
                )
                if expected_count is not None and len(proposals) != expected_count:
                    raise ValueError(
                        f"expected {expected_count} proposals, got {len(proposals)}"
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                record = LLMResponseRecord(
                    prompt=request_prompt,
                    response=raw_response,
                    model=model,
                    attempt=attempt,
                    accepted=False,
                    error=str(exc)[:500],
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                )
                records.append(record)
                self.archive.record(record)
                if attempt == 2:
                    raise StructuredOutputError(
                        "LLM response failed schema validation after one retry",
                        raw_response,
                    ) from exc
                continue

            record = LLMResponseRecord(
                prompt=request_prompt,
                response=raw_response,
                model=model,
                attempt=attempt,
                accepted=True,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
            )
            records.append(record)
            self.archive.record(record)
            return StructuredBiasResponse(proposals, tuple(records))

        raise StructuredOutputError(
            "LLM response failed schema validation after one retry",
            last_raw_response,
        )

    @staticmethod
    def _retry_prompt(prompt: str) -> str:
        return (
            f"{prompt}\n\nYour previous response did not match the required JSON "
            "schema. Return a corrected JSON object containing only a non-empty "
            "proposals array. Do not include commentary or executable code."
        )
