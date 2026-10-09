"""Compile typed expression trees into deterministic per-image feature programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.primitives import SequenceValue, execute_primitive
from bias_optimizer.dsl.types import ValueType
from bias_optimizer.dsl.validator import ProgramConstraints, validate_program
from bias_optimizer.features._image import validated_image
from bias_optimizer.features.base import FeatureVector, Image
from bias_optimizer.features.skeleton_graph import skeleton_to_graph


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

    def transform_with_spatial_reassignment(
        self,
        image: Image,
        *,
        target_op: str,
        seed: int,
    ) -> FeatureVector:
        """Shuffle event locations while preserving values and location marginals."""
        _validate_intervention_target(target_op, seed)
        return self._transform(
            image,
            intervention="spatial_reassignment",
            target_op=target_op,
            shuffle_seed=seed,
        )

    def transform_with_cross_pair_shuffle(
        self,
        image: Image,
        *,
        target_op: str = "cross_histogram",
        seed: int,
    ) -> FeatureVector:
        """Break a cross-sequence relation while preserving both value marginals."""
        _validate_intervention_target(target_op, seed)
        return self._transform(
            image,
            intervention="cross_pair_shuffle",
            target_op=target_op,
            shuffle_seed=seed,
        )

    def transform_with_centroid_perturbation(
        self,
        image: Image,
        *,
        target_measure: str,
        seed: int,
    ) -> FeatureVector:
        """Break path-centroid assignment while preserving path counts and flags."""
        _validate_intervention_target(target_measure, seed)
        if target_measure not in {"centroid_x", "centroid_y"}:
            raise ValueError("centroid perturbation requires centroid_x or centroid_y")
        return self._transform(
            image,
            intervention="centroid_perturbation",
            target_op="path_summary",
            target_measure=target_measure,
            shuffle_seed=seed,
        )

    def transform_with_topology_rewire(
        self,
        image: Image,
        *,
        target_op: str,
        seed: int,
    ) -> FeatureVector:
        """Change connectivity/cycle structure while preserving foreground count."""
        _validate_intervention_target(target_op, seed)
        return self._transform(
            image,
            intervention="topology_rewire",
            target_op=target_op,
            shuffle_seed=seed,
        )

    def _transform(
        self,
        image: Image,
        *,
        shared_cache: SubexpressionCache | None = None,
        sample_key: str | None = None,
        shuffle_sequence_at: str | None = None,
        intervention: str | None = None,
        target_op: str | None = None,
        target_measure: str | None = None,
        shuffle_seed: int = 0,
    ) -> FeatureVector:
        validated = validated_image(image)
        cache: dict[str, Any] = {}
        rng = np.random.default_rng(shuffle_seed)
        intervention_count = 0

        def evaluate(node: Expr) -> Any:
            nonlocal intervention_count
            key = node.to_json()
            if key not in cache:
                value = None
                if shared_cache is not None and sample_key is not None:
                    value = shared_cache.get(sample_key, key, operation=node.op)
                if value is None:
                    args = tuple(evaluate(child) for child in node.args)
                    if intervention == "cross_pair_shuffle" and node.op == target_op:
                        if node.op != "cross_histogram" or not args:
                            raise ValueError(
                                "cross-pair shuffle requires a cross_histogram node"
                            )
                        args = (
                            _shuffle_sequence_values(args[0], rng),
                            *args[1:],
                        )
                    value = execute_primitive(
                        node.op,
                        args,
                        node.parameter_values,
                        validated,
                    )
                    if node.op == shuffle_sequence_at:
                        value = _shuffle_sequence_values(
                            value, rng, shuffle_locations=True
                        )
                        intervention_count += 1
                    if intervention == "spatial_reassignment" and node.op == target_op:
                        value = _shuffle_sequence_locations(value, rng)
                        intervention_count += 1
                    if (
                        intervention == "centroid_perturbation"
                        and node.op == target_op
                        and node.parameter_values.get("measure") == target_measure
                    ):
                        value = _perturb_centroid_values(value, rng)
                        intervention_count += 1
                    if intervention == "topology_rewire" and node.op == target_op:
                        value = _rewire_topology(value, rng)
                        intervention_count += 1
                    if intervention == "cross_pair_shuffle" and node.op == target_op:
                        intervention_count += 1
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
        if (
            shuffle_sequence_at is not None or intervention is not None
        ) and not intervention_count:
            raise ValueError(
                f"intervention target {target_op or shuffle_sequence_at!r} was not found"
            )
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


def _shuffle_sequence_values(
    value: Any, rng: np.random.Generator, *, shuffle_locations: bool = False
) -> Any:
    if not isinstance(value, SequenceValue):
        raise TypeError("intervention target did not produce an ordered sequence")
    permutations = tuple(rng.permutation(len(group)) for group in value.groups)
    groups = tuple(
        group[permutation]
        for group, permutation in zip(value.groups, permutations, strict=True)
    )
    locations = value.locations
    if locations and shuffle_locations:
        locations = tuple(
            points[permutation]
            for points, permutation in zip(value.locations, permutations, strict=True)
        )
    return SequenceValue(groups, locations)


def _shuffle_sequence_locations(value: Any, rng: np.random.Generator) -> Any:
    if not isinstance(value, SequenceValue) or not value.locations:
        raise ValueError("spatial reassignment requires located sequence values")
    locations = tuple(
        points[rng.permutation(len(points))] for points in value.locations
    )
    return SequenceValue(value.groups, locations)


def _perturb_centroid_values(value: Any, rng: np.random.Generator) -> Any:
    if not isinstance(value, SequenceValue) or not value.locations:
        raise ValueError("centroid perturbation requires located path summaries")
    groups = []
    for group in value.groups:
        if len(group) > 1:
            groups.append(group[rng.permutation(len(group))])
        elif len(group) == 1:
            groups.append(1.0 - group)
        else:
            groups.append(group.copy())
    return SequenceValue(tuple(groups), value.locations)


def _topology_signature(mask: np.ndarray) -> tuple[int, int]:
    graph = skeleton_to_graph(np.asarray(mask, dtype=np.bool_))
    component_count = nx.number_connected_components(graph) if graph else 0
    cycle_rank = graph.number_of_edges() - graph.number_of_nodes() + component_count
    return component_count, cycle_rank


def _rewire_topology(value: Any, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(value, dtype=np.bool_)
    if mask.ndim != 2:
        raise TypeError("topology rewire requires a two-dimensional binary image")
    foreground = np.argwhere(mask)
    background = np.argwhere(~mask)
    if not len(foreground) or not len(background):
        raise ValueError("topology rewire requires both foreground and background")
    original_signature = _topology_signature(mask)
    for _ in range(16):
        source = foreground[int(rng.integers(len(foreground)))]
        destination = background[int(rng.integers(len(background)))]
        candidate = mask.copy()
        candidate[tuple(source)] = False
        candidate[tuple(destination)] = True
        if _topology_signature(candidate) != original_signature:
            return candidate
    raise ValueError("could not change topology while preserving foreground count")


def _validate_intervention_target(target_op: str, seed: int) -> None:
    if not isinstance(target_op, str) or not target_op:
        raise ValueError("target_op must be a non-empty primitive name")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")


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
