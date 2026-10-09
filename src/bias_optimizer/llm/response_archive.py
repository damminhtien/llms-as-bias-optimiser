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

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "model": self.model,
            "attempt": self.attempt,
            "accepted": self.accepted,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class JsonlResponseArchive:
    """Append structured LLM request records to a local JSONL file."""

    path: Path = Path("results/llm_responses.jsonl")

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def record(self, response: LLMResponseRecord) -> None:
        if not isinstance(response, LLMResponseRecord):
            raise TypeError("response archive requires an LLMResponseRecord")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as archive:
            archive.write(line + "\n")
            archive.flush()
            os.fsync(archive.fileno())
