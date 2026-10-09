"""JSONL persistence for raw LLM prompts and responses."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LLMResponseRecord:
    """One raw model request and its schema-validation outcome."""

    prompt: str
    response: str
    model: str
    attempt: int
    accepted: bool
    error: str | None = None
    prompt_tokens: int | None = None
    response_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("prompt_tokens", "response_tokens"):
            count = getattr(self, field_name)
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.response_tokens is None:
            return None
        return self.prompt_tokens + self.response_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class JsonlResponseArchive:
    """Append structured LLM request records to a local JSONL file."""

    path: Path = Path("results/llm_responses.jsonl")

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def record_count(self) -> int:
        """Return the number of durable request records already in the archive."""
        if not self.path.exists():
            return 0
        with self.path.open(encoding="utf-8") as archive:
            return sum(1 for line in archive if line.strip())

    def record(self, response: LLMResponseRecord) -> None:
        if not isinstance(response, LLMResponseRecord):
            raise TypeError("response archive requires an LLMResponseRecord")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as archive:
            archive.write(line + "\n")
            archive.flush()
            os.fsync(archive.fileno())
