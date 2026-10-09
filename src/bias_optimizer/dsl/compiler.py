"""Compile typed expression trees into deterministic per-image feature programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.primitives import execute_primitive
from bias_optimizer.dsl.types import ValueType
from bias_optimizer.dsl.validator import ProgramConstraints, validate_program
from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image


class ProgramCompilationError(ValueError):
    """Raised when a candidate is invalid for the typed DSL or selected track."""


@dataclass(frozen=True, slots=True)
class ProgramPipeline:
    """Executable form of one validated AST; no generated source is evaluated."""

    program: Expr
    feature_dim: int

    def transform(self, image: Image) -> FeatureVector:
        validated = validated_image(image)
        cache: dict[str, Any] = {}

        def evaluate(node: Expr) -> Any:
            key = node.to_json()
            if key not in cache:
                args = tuple(evaluate(child) for child in node.args)
                cache[key] = execute_primitive(
                    node.op,
                    args,
                    node.parameter_values,
                    validated,
                )
            return cache[key]

        value = evaluate(self.program)
        if self.program.op == "image" or not np.isscalar(value):
            features = np.asarray(value, dtype=np.float32).reshape(-1)
        else:
            features = np.asarray([value], dtype=np.float32)
        if features.ndim != 1 or len(features) != self.feature_dim:
            raise ValueError(
                f"program returned {features.size} features; expected {self.feature_dim}"
            )
        if not np.isfinite(features).all():
            raise ValueError("program returned non-finite features")
        return features


class ProgramCompiler:
    """Validate expression types and compile into an allow-listed interpreter."""

    def __init__(self, constraints: ProgramConstraints | None = None) -> None:
        self.constraints = constraints or ProgramConstraints()

    def compile(self, program: Expr) -> ProgramPipeline:
        try:
            info = validate_program(program, self.constraints)
            dimension = 1 if info.value_type is ValueType.SCALAR else info.dimension
            if dimension is None:
                raise ValueError("program feature dimension is unknown")
            return ProgramPipeline(program=program, feature_dim=dimension)
        except (TypeError, ValueError) as exc:
            raise ProgramCompilationError(
                f"cannot compile representation program: {exc}"
            ) from exc
