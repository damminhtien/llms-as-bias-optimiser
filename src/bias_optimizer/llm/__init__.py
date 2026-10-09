"""LLM proposal boundary; evaluation remains local and deterministic."""

from bias_optimizer.llm.client import LLMClient, MockLLMClient
from bias_optimizer.llm.ollama_client import (
    OllamaClientError,
    OllamaLLMClient,
)
from bias_optimizer.llm.proposer import LLMProposer, ProposedBias
from bias_optimizer.llm.response_archive import JsonlResponseArchive, LLMResponseRecord
from bias_optimizer.llm.structured import (
    StructuredBiasClient,
    StructuredBiasResponse,
    StructuredOutputError,
    parse_bias_response,
)

__all__ = [
    "JsonlResponseArchive",
    "LLMClient",
    "LLMProposer",
    "LLMResponseRecord",
    "MockLLMClient",
    "OllamaClientError",
    "OllamaLLMClient",
    "ProposedBias",
    "StructuredBiasClient",
    "StructuredBiasResponse",
    "StructuredOutputError",
    "parse_bias_response",
]
