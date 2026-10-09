"""Tests for JSON-only LLM responses, validation retries, and raw archives."""

import json

import pytest

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.llm import (
    JsonlResponseArchive,
    LLMTokenUsage,
    MockLLMClient,
    StructuredBiasClient,
    StructuredOutputError,
    parse_bias_response,
)


def _bias() -> BiasSpec:
    return BiasSpec(
        name="qwen_structured_candidate",
        hypothesis="Vertical occupancy may complement raw shape features.",
        operators=(OperatorSpec("topology"), OperatorSpec("spatial")),
        prediction="Should improve validation accuracy at 500 examples.",
        falsification="Reject if the validation score does not improve.",
    )


def _response(bias: BiasSpec | None = None) -> str:
    candidate = bias if bias is not None else _bias()
    return json.dumps({"proposals": [candidate.to_dict()]})


def test_parser_returns_valid_bias_specs() -> None:
    parsed = parse_bias_response(_response())

    assert parsed == (_bias(),)


@pytest.mark.parametrize(
    "raw_response",
    [
        "not JSON",
        '```json\n{"proposals": []}\n```',
        '{"proposals": [], "python": "ignored"}',
    ],
)
def test_parser_rejects_malformed_or_non_json_only_responses(raw_response: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_bias_response(raw_response)


def test_structured_client_retries_once_and_archives_both_raw_responses(
    tmp_path,
) -> None:
    invalid = '{"proposals": []}'
    valid = _response()
    mock = MockLLMClient(invalid, valid)
    archive = JsonlResponseArchive(tmp_path / "responses.jsonl")
    structured_client = StructuredBiasClient(mock, archive=archive)

    result = structured_client.generate("Propose one bias.", expected_count=1)

    assert result.proposals == (_bias(),)
    assert result.model == "unspecified"
    assert len(result.records) == 2
    assert result.records[0].accepted is False
    assert result.records[1].accepted is True
    assert len(mock.prompts) == 2
    assert "did not match" in mock.prompts[1]
    records = [
        json.loads(line)
        for line in archive.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["response"] for record in records] == [invalid, valid]
    assert [record["attempt"] for record in records] == [1, 2]


def test_structured_client_stops_after_exactly_one_failed_retry(tmp_path) -> None:
    mock = MockLLMClient("invalid first", "invalid second", "unused")
    structured_client = StructuredBiasClient(
        mock,
        archive=JsonlResponseArchive(tmp_path / "responses.jsonl"),
    )

    with pytest.raises(StructuredOutputError, match="after one retry") as error:
        structured_client.generate("Propose one bias.")

    assert error.value.raw_response == "invalid second"
    assert len(mock.prompts) == 2
    assert len((tmp_path / "responses.jsonl").read_text().splitlines()) == 2


def test_structured_client_retries_when_candidate_count_is_wrong(tmp_path) -> None:
    mock = MockLLMClient(_response(), _response())
    structured_client = StructuredBiasClient(
        mock,
        archive=JsonlResponseArchive(tmp_path / "responses.jsonl"),
    )

    with pytest.raises(StructuredOutputError, match="after one retry"):
        structured_client.generate("Propose two biases.", expected_count=2)

    assert len(mock.prompts) == 2


def test_structured_client_archives_provider_token_usage(tmp_path) -> None:
    class UsageClient:
        model = "qwen3.5:35b-mlx"
        last_usage = None

        def generate(self, prompt: str) -> str:
            self.last_usage = LLMTokenUsage(prompt_tokens=31, response_tokens=12)
            return _response()

    archive = JsonlResponseArchive(tmp_path / "responses.jsonl")
    result = StructuredBiasClient(UsageClient(), archive=archive).generate(
        "Propose one bias.", expected_count=1
    )

    record = result.records[0]
    assert record.prompt_tokens == 31
    assert record.response_tokens == 12
    assert record.total_tokens == 43
    saved = json.loads(archive.path.read_text(encoding="utf-8"))
    assert saved["total_tokens"] == 43


def test_unknown_operator_is_rejected_without_executing_generated_code(
    tmp_path,
) -> None:
    malicious = BiasSpec(
        name="untrusted",
        hypothesis="This content is never executed.",
        operators=(OperatorSpec("python_exec", {"source": "raise SystemExit()"}),),
        prediction="unknown operator",
        falsification="must fail validation",
    )
    mock = MockLLMClient(_response(malicious), _response(malicious))
    structured_client = StructuredBiasClient(
        mock,
        archive=JsonlResponseArchive(tmp_path / "responses.jsonl"),
    )

    with pytest.raises(StructuredOutputError, match="after one retry"):
        structured_client.generate("Propose one bias.")

    assert len(mock.prompts) == 2
