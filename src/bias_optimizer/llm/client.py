"""Provider-neutral LLM protocol and deterministic test double."""

from collections import deque
from typing import Protocol


class LLMClient(Protocol):
    """Generate text from a prompt without accessing experiment data."""

    def generate(self, prompt: str) -> str:
        """Return the provider response text."""


class MockLLMClient:
    """Return queued responses while recording prompts for offline tests."""

    def __init__(self, *responses: str) -> None:
        if not responses:
            raise ValueError("MockLLMClient requires at least one response")
        if not all(isinstance(response, str) for response in responses):
            raise TypeError("mock responses must be strings")
        self._responses = deque(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.prompts.append(prompt)
        if not self._responses:
            raise RuntimeError("MockLLMClient has no queued response")
        return self._responses.popleft()
