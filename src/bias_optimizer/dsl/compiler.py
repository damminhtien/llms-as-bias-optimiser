"""Compile typed expression trees into deterministic per-image feature programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bias_optimizer.cache.subexpression_cache import SubexpressionCache
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
        return self._transform(image)

    def transform_with_cache(
        self,
        image: Image,
        *,
        cache: SubexpressionCache,
        sample_key: str,
    ) -> FeatureVector:
        """Transform one row and reuse expensive identical subtrees across ASTs."""
        if not isinstance(cache, SubexpressionCache):
            raise TypeError("cache must be a SubexpressionCache")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("sample_key must be a non-empty string")
        return self._transform(image, shared_cache=cache, sample_key=sample_key)

    def transform_with_sequence_shuffle(
        self,
        image: Image,
        *,
        target_op: str,
        seed: int,
    ) -> FeatureVector:
        """Destroy sequence order at a typed sequence node, preserving each multiset."""
        if not isinstance(target_op, str) or not target_op:
            raise ValueError("target_op must be a non-empty primitive name")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        return self._transform(
            image,
            shuffle_sequence_at=target_op,
            shuffle_seed=seed,
        )

    def _transform(
        self,
        image: Image,
        *,
        shared_cache: SubexpressionCache | None = None,
        sample_key: str | None = None,
        shuffle_sequence_at: str | None = None,
        shuffle_seed: int = 0,
    ) -> FeatureVector:
        validated = validated_image(image)
        cache: dict[str, Any] = {}
        rng = np.random.default_rng(shuffle_seed)

        def evaluate(node: Expr) -> Any:
            key = node.to_json()
            if key not in cache:
                value = None
                if shared_cache is not None and sample_key is not None:
                    value = shared_cache.get(sample_key, key, operation=node.op)
                if value is None:
                    args = tuple(evaluate(child) for child in node.args)
                    value = execute_primitive(
                        node.op,
                        args,
                        node.parameter_values,
                        validated,
                    )
                    if node.op == shuffle_sequence_at:
                        from bias_optimizer.dsl.primitives import SequenceValue

                        if not isinstance(value, SequenceValue):
                            raise TypeError(
                                f"{node.op} did not produce an ordered sequence"
                            )
                        value = SequenceValue(
                            tuple(rng.permutation(group) for group in value.groups)
                        )
                    if shared_cache is not None and sample_key is not None:
                        shared_cache.set(
                            sample_key,
                            key,
                            value,
                            operation=node.op,
                        )
                cache[key] = value
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
