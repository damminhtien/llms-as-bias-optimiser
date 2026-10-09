"""Local Ollama provider adapter, defaulting to Qwen 3.5."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:35b-mlx"


class OllamaClientError(RuntimeError):
    """Raised when the local Ollama service fails its text-generation contract."""


@dataclass(frozen=True, slots=True)
class LLMTokenUsage:
    """Ollama-reported input and output token counts for one request."""

    prompt_tokens: int | None
    response_tokens: int | None


@dataclass(frozen=True, slots=True)
class OllamaLLMClient:
    """Generate JSON responses through the local Ollama chat endpoint."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float = 120.0
    _last_usage: LLMTokenUsage | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.base_url, str):
            raise TypeError("base_url must be a string")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) URL without credentials")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @classmethod
    def from_environment(cls) -> OllamaLLMClient:
        """Use Ollama environment overrides with the local Qwen default."""
        return cls(
            base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        )

    @property
    def last_usage(self) -> LLMTokenUsage | None:
        """Return token counts from the most recent successful response."""
        return self._last_usage

    def generate(self, prompt: str) -> str:
        return self._generate(prompt, "json", temperature=0.0)

    def generate_structured(
        self,
        prompt: str,
        schema: Mapping[str, object],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        timeout_seconds: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        """Generate a response constrained by a JSON Schema object."""
        if not isinstance(schema, Mapping):
            raise TypeError("schema must be a JSON Schema object")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be finite and in [0, 2]")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("seed must be a non-negative integer")
        if timeout_seconds is not None and (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if num_predict is not None and (
            type(num_predict) is not int or num_predict <= 0
        ):
            raise ValueError("num_predict must be a positive integer")
        return self._generate(
            prompt,
            dict(schema),
            temperature=float(temperature),
            seed=seed,
            timeout_seconds=timeout_seconds,
            num_predict=num_predict,
        )

    def _generate(
        self,
        prompt: str,
        format_value: str | dict[str, object],
        *,
        temperature: float,
        seed: int | None = None,
        timeout_seconds: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        options: dict[str, int | float] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        if num_predict is not None:
            options["num_predict"] = num_predict
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": format_value,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": options,
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=(
                    self.timeout_seconds if timeout_seconds is None else timeout_seconds
                ),
            ) as response:
                raw_response = response.read()
        except HTTPError as exc:
            detail = exc.read(500).decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise OllamaClientError(f"Ollama returned HTTP {exc.code}{suffix}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise OllamaClientError(
                f"could not reach Ollama at {self.base_url}"
            ) from exc

        try:
            result = json.loads(raw_response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaClientError("Ollama returned invalid JSON") from exc
        object.__setattr__(
            self,
            "_last_usage",
            LLMTokenUsage(
                prompt_tokens=_token_count(result, "prompt_eval_count"),
                response_tokens=_token_count(result, "eval_count"),
            ),
        )
        message = result.get("message") if isinstance(result, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaClientError("Ollama response is missing message.content")
        return content


def _token_count(value: object, field_name: str) -> int | None:
    if not isinstance(value, dict):
        return None
    count = value.get(field_name)
    return count if type(count) is int and count >= 0 else None
