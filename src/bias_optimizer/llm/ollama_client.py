"""Local Ollama provider adapter, defaulting to Qwen 3.5."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from math import isfinite
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:35b-mlx"


class OllamaClientError(RuntimeError):
    """Raised when the local Ollama service fails its text-generation contract."""


@dataclass(frozen=True, slots=True)
class OllamaLLMClient:
    """Generate JSON responses through the local Ollama chat endpoint."""

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float = 120.0

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

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0},
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
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
        message = result.get("message") if isinstance(result, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaClientError("Ollama response is missing message.content")
        return content
