"""Declarative, provider-independent descriptions of candidate biases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any


def _freeze_json(value: Any) -> Any:
    """Copy JSON-compatible parameter values into immutable containers."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("operator parameter keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise TypeError(
        "operator parameters must be finite JSON values, "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """A named, allow-listed feature operator and its parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("operator name cannot be empty")
        if not isinstance(self.params, Mapping):
            raise TypeError("operator params must be a mapping")
        object.__setattr__(self, "params", _freeze_json(self.params))


@dataclass(frozen=True, slots=True)
class BiasSpec:
    """A testable representation hypothesis composed of approved operators."""

    name: str
    hypothesis: str
    operators: tuple[OperatorSpec, ...]
    prediction: str
    falsification: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("bias name cannot be empty")
        operators = tuple(self.operators)
        if not all(isinstance(operator, OperatorSpec) for operator in operators):
            raise TypeError("bias operators must contain OperatorSpec values")
        object.__setattr__(self, "operators", operators)
