"""Provider-neutral LLM client protocol."""

from typing import Protocol


class LLMClient(Protocol):
    """Generate text from a prompt without accessing experiment data."""

    def generate(self, prompt: str) -> str:
        """Return the provider response text."""
