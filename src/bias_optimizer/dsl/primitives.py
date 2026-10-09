"""The bounded, deterministic atomic operations available to DSL programs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import pi
from types import MappingProxyType
from typing import Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from skimage.morphology import skeletonize

from bias_optimizer.dsl.types import TypeInfo, ValueType
from bias_optimizer.features.skeleton_graph import (
    SkeletonGraph,
    extract_graph_paths,
    skeleton_to_graph,
)

ParamValidator = Callable[[Mapping[str, Any]], None]
DimensionResolver = Callable[[tuple[TypeInfo, ...], Mapping[str, Any]], int | None]
Executor = Callable[[tuple[Any, ...], Mapping[str, Any], NDArray[np.float32]], Any]
TypeSet = frozenset[ValueType]


@dataclass(frozen=True, slots=True)
class SequenceValue:
    groups: tuple[NDArray[np.float64], ...]


def _finite_number(params: Mapping[str, Any], name: str, default: float) -> float:
    value = params.get(name, default)
    if type(value) not in (int, float) or not np.isfinite(value):
        raise TypeError(f"parameter {name!r} must be a finite number")
    return float(value)


def _integer(params: Mapping[str, Any], name: str, default: int) -> int:
    value = params.get(name, default)
    if type(value) is not int:
        raise TypeError(f"parameter {name!r} must be an integer")
    return value


def _number_list(
    params: Mapping[str, Any], name: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    value = params.get(name, default)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"parameter {name!r} must be a non-empty array")
    if any(type(item) not in (int, float) or not np.isfinite(item) for item in value):
        raise TypeError(f"parameter {name!r} must contain finite numbers")
    return tuple(float(item) for item in value)


def _integer_list(
    params: Mapping[str, Any], name: str, default: tuple[int, ...]
) -> tuple[int, ...]:
    value = params.get(name, default)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"parameter {name!r} must be a non-empty array")
    if any(type(item) is not int for item in value):
        raise TypeError(f"parameter {name!r} must contain integers")
    return tuple(value)


def _no_params(params: Mapping[str, Any]) -> None:
    if params:
        raise ValueError(f"operation accepts no parameters: {sorted(params)}")


def _threshold_params(params: Mapping[str, Any]) -> None:
    threshold = _finite_number(params, "value", 0.5)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold value must be in [0, 1]")
    if set(params) - {"value"}:
        raise ValueError("threshold accepts only the value parameter")


def _skeleton_params(params: Mapping[str, Any]) -> None:
    if (
        "threshold" in params
        and not 0.0 <= _finite_number(params, "threshold", 0.5) <= 1.0
    ):
        raise ValueError("skeleton threshold must be in [0, 1]")
    if set(params) - {"threshold"}:
        raise ValueError("skeletonize accepts only the threshold parameter")


def _sign_params(params: Mapping[str, Any]) -> None:
    epsilon = _finite_number(params, "epsilon", 0.0)
    if not 0 <= epsilon <= pi:
        raise ValueError("sign epsilon must be in [0, pi]")
    if set(params) - {"epsilon"}:
        raise ValueError("sign accepts only the epsilon parameter")


def _histogram_params(params: Mapping[str, Any]) -> None:
    bins = _integer(params, "bins", 8)
    low = _finite_number(params, "low", -pi)
    high = _finite_number(params, "high", pi)
    if not 2 <= bins <= 32:
        raise ValueError("histogram bins must be between 2 and 32")
    if not -128 <= low < high <= 128:
        raise ValueError("histogram range must satisfy -128 <= low < high <= 128")
    if set(params) - {"bins", "low", "high"}:
        raise ValueError("histogram accepts only bins, low, and high")


def _moments_params(params: Mapping[str, Any]) -> None:
    orders = _integer_list(params, "orders", (1, 2, 3))
    if len(orders) > 8 or any(not 1 <= order <= 8 for order in orders):
        raise ValueError("moment orders must contain at most 8 values in [1, 8]")
    if len(set(orders)) != len(orders):
        raise ValueError("moment orders cannot contain duplicates")
    if set(params) - {"orders"}:
        raise ValueError("moments accepts only the orders parameter")


def _quantiles_params(params: Mapping[str, Any]) -> None:
    quantiles = _number_list(params, "quantiles", (0.1, 0.25, 0.5, 0.75, 0.9))
    if len(quantiles) > 16 or any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("quantiles must contain at most 16 values in [0, 1]")
    if tuple(sorted(set(quantiles))) != quantiles:
        raise ValueError("quantiles must be strictly increasing")
    if set(params) - {"quantiles"}:
        raise ValueError("quantiles accepts only the quantiles parameter")


def _autocorrelation_params(params: Mapping[str, Any]) -> None:
    lags = _integer_list(params, "lags", (1, 2, 4))
    if len(lags) > 16 or any(not 1 <= lag <= 32 for lag in lags):
        raise ValueError(
            "autocorrelation lags must contain at most 16 values in [1, 32]"
        )
    if tuple(sorted(set(lags))) != lags:
        raise ValueError("autocorrelation lags must be strictly increasing")
    if set(params) - {"lags"}:
        raise ValueError("autocorrelation accepts only the lags parameter")


def _spatial_params(params: Mapping[str, Any]) -> None:
    rows = _integer(params, "rows", 1)
    cols = _integer(params, "cols", 3)
    if not 1 <= rows <= 7 or not 1 <= cols <= 7 or rows * cols > 64:
        raise ValueError(
            "spatial rows and cols must be in [1, 7] with at most 64 cells"
        )
    if set(params) - {"rows", "cols"}:
        raise ValueError("spatial_split accepts only rows and cols")


def _ratio_params(params: Mapping[str, Any]) -> None:
    epsilon = _finite_number(params, "epsilon", 1e-6)
    if not 1e-12 <= epsilon <= 1.0:
        raise ValueError("ratio epsilon must be in [1e-12, 1]")
    if set(params) - {"epsilon"}:
        raise ValueError("ratio accepts only the epsilon parameter")


def _same_dimension(args: tuple[TypeInfo, ...], _: Mapping[str, Any]) -> int | None:
    dimensions = [item.dimension for item in args]
    if any(dimension is None for dimension in dimensions):
        raise ValueError("vector dimensions must be statically known")
    if len(set(dimensions)) != 1:
        raise ValueError("ratio inputs must have the same vector dimension")
    return dimensions[0]


def _sequence_groups(values: Any) -> tuple[NDArray[np.float64], ...]:
    """Normalize flat or path-separated sequences while retaining boundaries."""
    if isinstance(values, SequenceValue):
        return values.groups
    return (np.asarray(values, dtype=np.float64).reshape(-1),)


def _flatten_sequence(values: Any) -> NDArray[np.float64]:
    groups = tuple(group for group in _sequence_groups(values) if len(group))
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.float64)


def _concat_dimension(args: tuple[TypeInfo, ...], _: Mapping[str, Any]) -> int:
    if any(item.dimension is None for item in args):
        raise ValueError("concat inputs must have statically known dimensions")
    return sum(item.dimension for item in args if item.dimension is not None)


def _identity_dimension(args: tuple[TypeInfo, ...], _: Mapping[str, Any]) -> int | None:
    return args[0].dimension


def _unknown_dimension(_: tuple[TypeInfo, ...], __: Mapping[str, Any]) -> None:
    return None


def _histogram_dimension(_: tuple[TypeInfo, ...], params: Mapping[str, Any]) -> int:
    return _integer(params, "bins", 8)


def _orders_dimension(_: tuple[TypeInfo, ...], params: Mapping[str, Any]) -> int:
    return len(_integer_list(params, "orders", (1, 2, 3)))


def _quantiles_dimension(_: tuple[TypeInfo, ...], params: Mapping[str, Any]) -> int:
    return len(_number_list(params, "quantiles", (0.1, 0.25, 0.5, 0.75, 0.9)))


def _lags_dimension(_: tuple[TypeInfo, ...], params: Mapping[str, Any]) -> int:
    return len(_integer_list(params, "lags", (1, 2, 4)))


def _spatial_dimension(_: tuple[TypeInfo, ...], params: Mapping[str, Any]) -> int:
    return _integer(params, "rows", 1) * _integer(params, "cols", 3)


def _fixed_dimension(value: int) -> DimensionResolver:
    return lambda _args, _params: value


@dataclass(frozen=True, slots=True)
class PrimitiveDefinition:
    """Static contract for one safe DSL primitive."""

    input_types: tuple[TypeSet, ...]
    output_type: ValueType
    validate_params: ParamValidator
    output_dimension: DimensionResolver
    family: str | None = None
    variadic_input: TypeSet | None = None
    minimum_args: int = 0

    def validate_inputs(self, actual: tuple[ValueType, ...]) -> None:
        if self.variadic_input is not None:
            if len(actual) < self.minimum_args:
                raise ValueError(
                    f"operation requires at least {self.minimum_args} arguments"
                )
            if any(value not in self.variadic_input for value in actual):
                expected = ", ".join(
                    sorted(value.value for value in self.variadic_input)
                )
                raise TypeError(f"operation arguments must have types: {expected}")
            return
        if len(actual) != len(self.input_types):
            raise ValueError(
                f"operation expects {len(self.input_types)} arguments, got {len(actual)}"
            )
        for index, (value, accepted) in enumerate(
            zip(actual, self.input_types, strict=True)
        ):
            if value not in accepted:
                expected = ", ".join(sorted(item.value for item in accepted))
                raise TypeError(
                    f"argument {index} must have type {expected}, got {value.value}"
                )


def _accepts(*types: ValueType) -> TypeSet:
    return frozenset(types)


_IMAGE = _accepts(ValueType.IMAGE)
_BINARY = _accepts(ValueType.BINARY_IMAGE)
_SKELETON = _accepts(ValueType.SKELETON)
_GRAPH = _accepts(ValueType.GRAPH)
_PATHS = _accepts(ValueType.PATH_SET)
_SEQUENCE = _accepts(ValueType.SEQUENCE)
_SEQUENCE_TYPES = _accepts(ValueType.SEQUENCE, ValueType.ANGLE_SEQUENCE)
_ANGLE_SEQUENCE = _accepts(ValueType.ANGLE_SEQUENCE)
_VECTOR = _accepts(ValueType.VECTOR)
_SCALAR_COUNT_INPUTS = _accepts(
    ValueType.GRAPH,
    ValueType.PATH_SET,
    ValueType.SEQUENCE,
    ValueType.ANGLE_SEQUENCE,
)

PRIMITIVES: Mapping[str, PrimitiveDefinition] = MappingProxyType(
    {
        "image": PrimitiveDefinition(
            (), ValueType.IMAGE, _no_params, _unknown_dimension
        ),
        "threshold": PrimitiveDefinition(
            (_IMAGE,),
            ValueType.BINARY_IMAGE,
            _threshold_params,
            _unknown_dimension,
            "topology",
        ),
        "skeletonize": PrimitiveDefinition(
            (_accepts(ValueType.IMAGE, ValueType.BINARY_IMAGE),),
            ValueType.SKELETON,
            _skeleton_params,
            _unknown_dimension,
            "graph_structure",
        ),
        "graph": PrimitiveDefinition(
            (_SKELETON,),
            ValueType.GRAPH,
            _no_params,
            _unknown_dimension,
            "graph_structure",
        ),
        "connected_components": PrimitiveDefinition(
            (_BINARY,), ValueType.SCALAR, _no_params, _fixed_dimension(1), "topology"
        ),
        "cycle_rank": PrimitiveDefinition(
            (_GRAPH,), ValueType.SCALAR, _no_params, _fixed_dimension(1), "topology"
        ),
        "paths": PrimitiveDefinition(
            (_GRAPH,),
            ValueType.PATH_SET,
            _no_params,
            _unknown_dimension,
            "graph_structure",
        ),
        "degree_sequence": PrimitiveDefinition(
            (_GRAPH,),
            ValueType.SEQUENCE,
            _no_params,
            _unknown_dimension,
            "graph_structure",
        ),
        "edge_lengths": PrimitiveDefinition(
            (_PATHS,), ValueType.SEQUENCE, _no_params, _unknown_dimension, "geometry"
        ),
        "angles": PrimitiveDefinition(
            (_PATHS,),
            ValueType.ANGLE_SEQUENCE,
            _no_params,
            _unknown_dimension,
            "geometry",
        ),
        "delta": PrimitiveDefinition(
            (_SEQUENCE,),
            ValueType.SEQUENCE,
            _no_params,
            _unknown_dimension,
            "stroke_dynamics",
        ),
        "delta_angle": PrimitiveDefinition(
            (_ANGLE_SEQUENCE,),
            ValueType.ANGLE_SEQUENCE,
            _no_params,
            _unknown_dimension,
            "stroke_dynamics",
        ),
        "sign": PrimitiveDefinition(
            (_SEQUENCE_TYPES,),
            ValueType.SEQUENCE,
            _sign_params,
            _unknown_dimension,
            "stroke_dynamics",
        ),
        "run_length_encode": PrimitiveDefinition(
            (_SEQUENCE,),
            ValueType.SEQUENCE,
            _no_params,
            _unknown_dimension,
            "stroke_dynamics",
        ),
        "histogram": PrimitiveDefinition(
            (_SEQUENCE_TYPES,),
            ValueType.VECTOR,
            _histogram_params,
            _histogram_dimension,
        ),
        "moments": PrimitiveDefinition(
            (_SEQUENCE_TYPES,), ValueType.VECTOR, _moments_params, _orders_dimension
        ),
        "quantiles": PrimitiveDefinition(
            (_SEQUENCE_TYPES,),
            ValueType.VECTOR,
            _quantiles_params,
            _quantiles_dimension,
        ),
        "autocorrelation": PrimitiveDefinition(
            (_SEQUENCE_TYPES,),
            ValueType.VECTOR,
            _autocorrelation_params,
            _lags_dimension,
            "frequency_scale",
        ),
        "normalize": PrimitiveDefinition(
            (_VECTOR,), ValueType.VECTOR, _no_params, _identity_dimension
        ),
        "spatial_split": PrimitiveDefinition(
            (_IMAGE,),
            ValueType.VECTOR,
            _spatial_params,
            _spatial_dimension,
            "spatial_relations",
        ),
        "flatten_pixels": PrimitiveDefinition(
            (_IMAGE,), ValueType.VECTOR, _no_params, _fixed_dimension(784), "raw_pixels"
        ),
        "ratio": PrimitiveDefinition(
            (_VECTOR, _VECTOR),
            ValueType.VECTOR,
            _ratio_params,
            _same_dimension,
            "compositional",
        ),
        "count": PrimitiveDefinition(
            (_SCALAR_COUNT_INPUTS,), ValueType.SCALAR, _no_params, _fixed_dimension(1)
        ),
        "concat": PrimitiveDefinition(
            (),
            ValueType.VECTOR,
            _no_params,
            _concat_dimension,
            "compositional",
            _VECTOR,
            2,
        ),
    }
)


def _cycle_rank(graph: SkeletonGraph) -> int:
    return int(
        graph.number_of_edges()
        - graph.number_of_nodes()
        + nx.number_connected_components(graph)
    )


def _path_edges(
    paths: Sequence[Sequence[tuple[int, int]]],
) -> tuple[NDArray[np.float64], ...]:
    groups: list[NDArray[np.float64]] = []
    for path in paths:
        if len(path) < 2:
            continue
        points = np.asarray(path, dtype=np.float64)
        groups.append(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return tuple(groups)


def _path_angles(
    paths: Sequence[Sequence[tuple[int, int]]],
) -> tuple[NDArray[np.float64], ...]:
    groups: list[NDArray[np.float64]] = []
    for path in paths:
        if len(path) < 2:
            continue
        points = np.asarray(path, dtype=np.float64)
        deltas = np.diff(points, axis=0)
        groups.append(np.arctan2(-deltas[:, 0], deltas[:, 1]))
    return tuple(groups)


def _spatial_vector(
    image: NDArray[np.float32], rows: int, cols: int
) -> NDArray[np.float32]:
    total = float(image.sum())
    if total == 0.0:
        return np.zeros(rows * cols, dtype=np.float32)
    values = [
        float(cell.sum()) / total
        for row in np.array_split(image, rows, axis=0)
        for cell in np.array_split(row, cols, axis=1)
    ]
    return np.asarray(values, dtype=np.float32)


def execute_primitive(
    op: str,
    args: tuple[Any, ...],
    params: Mapping[str, Any],
    image: NDArray[np.float32],
) -> Any:
    """Execute one validated primitive on a single image and its child values."""
    if op == "image":
        return image
    if op == "threshold":
        return image >= _finite_number(params, "value", 0.5)
    if op == "skeletonize":
        source = np.asarray(args[0])
        if source.dtype != np.bool_:
            source = source >= _finite_number(params, "threshold", 0.5)
        return np.asarray(skeletonize(source), dtype=np.bool_)
    if op == "graph":
        return skeleton_to_graph(args[0])
    if op == "connected_components":
        _, count = ndimage.label(args[0], structure=np.ones((3, 3), dtype=np.uint8))
        return float(count)
    if op == "cycle_rank":
        return float(_cycle_rank(args[0]))
    if op == "paths":
        return extract_graph_paths(args[0])
    if op == "degree_sequence":
        return np.asarray([degree for _, degree in args[0].degree], dtype=np.float64)
    if op == "edge_lengths":
        return SequenceValue(_path_edges(args[0]))
    if op == "angles":
        return SequenceValue(_path_angles(args[0]))
    if op == "delta":
        return SequenceValue(
            tuple(np.diff(group) for group in _sequence_groups(args[0]))
        )
    if op == "delta_angle":
        return SequenceValue(
            tuple(
                (differences + pi) % (2 * pi) - pi
                for differences in (
                    np.diff(group) for group in _sequence_groups(args[0])
                )
            )
        )
    if op == "sign":
        epsilon = _finite_number(params, "epsilon", 0.0)
        return SequenceValue(
            tuple(
                np.where(np.abs(values) <= epsilon, 0.0, np.sign(values))
                for values in _sequence_groups(args[0])
            )
        )
    if op == "run_length_encode":
        encoded = []
        for values in _sequence_groups(args[0]):
            if values.size == 0:
                encoded.append(np.empty(0, dtype=np.float64))
                continue
            boundaries = np.r_[
                0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)
            ]
            runs = []
            for start, end in pairwise(boundaries):
                value = float(np.sign(values[start]))
                runs.append(value * int(end - start))
            encoded.append(np.asarray(runs, dtype=np.float64))
        return SequenceValue(tuple(encoded))
    if op == "histogram":
        values = _flatten_sequence(args[0])
        bins = _integer(params, "bins", 8)
        low = _finite_number(params, "low", -pi)
        high = _finite_number(params, "high", pi)
        counts, _ = np.histogram(values, bins=bins, range=(low, high))
        total = int(counts.sum())
        return (counts / total if total else counts).astype(np.float32)
    if op == "moments":
        values = _flatten_sequence(args[0])
        orders = _integer_list(params, "orders", (1, 2, 3))
        if values.size == 0:
            return np.zeros(len(orders), dtype=np.float32)
        centered = values - float(values.mean())
        result = [
            float(values.mean()) if order == 1 else float(np.mean(centered**order))
            for order in orders
        ]
        return np.asarray(result, dtype=np.float32)
    if op == "quantiles":
        values = _flatten_sequence(args[0])
        quantiles = _number_list(params, "quantiles", (0.1, 0.25, 0.5, 0.75, 0.9))
        if values.size == 0:
            return np.zeros(len(quantiles), dtype=np.float32)
        return np.asarray(np.quantile(values, quantiles), dtype=np.float32)
    if op == "autocorrelation":
        lags = _integer_list(params, "lags", (1, 2, 4))
        centered_groups = tuple(
            group - group.mean() for group in _sequence_groups(args[0]) if group.size
        )
        denominator = sum(float(np.dot(group, group)) for group in centered_groups)
        result = []
        for lag in lags:
            if denominator == 0:
                result.append(0.0)
            else:
                numerator = sum(
                    float(np.dot(group[:-lag], group[lag:]))
                    for group in centered_groups
                    if lag < len(group)
                )
                result.append(numerator / denominator)
        return np.asarray(result, dtype=np.float32)
    if op == "normalize":
        values = np.asarray(args[0], dtype=np.float64)
        magnitude = float(np.abs(values).sum())
        return (values / magnitude if magnitude else values).astype(np.float32)
    if op == "spatial_split":
        return _spatial_vector(
            image, _integer(params, "rows", 1), _integer(params, "cols", 3)
        )
    if op == "flatten_pixels":
        return image.reshape(-1).copy()
    if op == "ratio":
        numerator = np.asarray(args[0], dtype=np.float64)
        denominator = np.asarray(args[1], dtype=np.float64)
        epsilon = _finite_number(params, "epsilon", 1e-6)
        safe = np.where(
            np.abs(denominator) < epsilon,
            np.where(denominator < 0, -epsilon, epsilon),
            denominator,
        )
        return (numerator / safe).astype(np.float32)
    if op == "count":
        value = args[0]
        if isinstance(value, nx.Graph):
            return float(value.number_of_nodes())
        if isinstance(value, SequenceValue):
            return float(sum(len(group) for group in value.groups))
        return float(len(value))
    if op == "concat":
        return np.concatenate([np.asarray(value, dtype=np.float32) for value in args])
    raise ValueError(f"unknown DSL primitive {op!r}")
