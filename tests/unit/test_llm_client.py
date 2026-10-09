"""Tests for the provider-neutral LLM contract and Ollama adapter."""

import json
from io import BytesIO

import pytest

from bias_optimizer.llm import MockLLMClient, OllamaLLMClient
from bias_optimizer.llm.ollama_client import (
    DEFAULT_OLLAMA_MODEL,
    OllamaClientError,
)


def test_mock_client_returns_queued_responses_and_records_prompts() -> None:
    client = MockLLMClient("first", "second")

    assert client.generate("prompt one") == "first"
    assert client.generate("prompt two") == "second"
    assert client.prompts == ["prompt one", "prompt two"]
    with pytest.raises(RuntimeError, match="no queued response"):
        client.generate("prompt three")


def test_ollama_client_defaults_to_local_qwen_and_json_mode(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data)
        return BytesIO(
            b'{"message":{"content":"{\\"ok\\":true}"},'
            b'"prompt_eval_count":23,"eval_count":6}'
        )

    monkeypatch.setattr("bias_optimizer.llm.ollama_client.urlopen", fake_urlopen)
    client = OllamaLLMClient()

    assert client.generate("return a JSON object") == '{"ok":true}'
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 120.0
    assert captured["payload"]["model"] == DEFAULT_OLLAMA_MODEL
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["temperature"] == 0
    assert client.last_usage.prompt_tokens == 23
    assert client.last_usage.response_tokens == 6


def test_ollama_client_reads_local_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:35b-mlx")

    client = OllamaLLMClient.from_environment()

    assert client.base_url == "http://localhost:11434"
    assert client.model == "qwen3.5:35b-mlx"


@pytest.mark.parametrize(
    "base_url",
    ["file:///tmp/model", "http://user:pass@localhost:11434", "http://localhost?x=1"],
)
def test_ollama_client_rejects_invalid_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        OllamaLLMClient(base_url=base_url)


def test_ollama_client_reports_invalid_provider_responses(monkeypatch) -> None:
    monkeypatch.setattr(
        "bias_optimizer.llm.ollama_client.urlopen",
        lambda request, timeout: BytesIO(b"not json"),
    )

    with pytest.raises(OllamaClientError, match="invalid JSON"):
        OllamaLLMClient().generate("return JSON")


def test_ollama_client_rejects_empty_prompts() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OllamaLLMClient().generate("  ")
