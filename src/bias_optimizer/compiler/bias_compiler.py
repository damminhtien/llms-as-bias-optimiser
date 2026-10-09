"""Compile validated bias specifications into deterministic feature pipelines."""

from __future__ import annotations

from bias_optimizer.domain.bias import BiasSpec
from bias_optimizer.features.pipeline import FeaturePipeline
from bias_optimizer.features.registry import OperatorRegistry


class BiasCompilationError(ValueError):
    """Raised when a bias cannot be compiled through the fixed allow-list."""


class BiasCompiler:
    """Resolve approved operator specifications without executing generated code."""

    def __init__(self) -> None:
        self._registry = OperatorRegistry()

    def compile(self, spec: BiasSpec) -> FeaturePipeline:
        if not isinstance(spec, BiasSpec):
            raise TypeError("BiasCompiler.compile requires a BiasSpec")
        try:
            operators = tuple(self._registry.build(item) for item in spec.operators)
            return FeaturePipeline(operators)
        except (TypeError, ValueError) as exc:
            raise BiasCompilationError(
                f"cannot compile bias {spec.name!r}: {exc}"
            ) from exc
