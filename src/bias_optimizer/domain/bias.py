"""Declarative, provider-independent descriptions of candidate biases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

_OPERATOR_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


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
        f"operator parameters must be finite JSON values, got {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_text(field_name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _require_fields(
    model_name: str, value: Mapping[str, Any], expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"invalid {model_name} fields; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """A named, allow-listed feature operator and its parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("operator name", self.name)
        if _OPERATOR_NAME.fullmatch(self.name) is None:
            raise ValueError(
                "operator name must be a lowercase identifier using letters, "
                "digits, and underscores"
            )
        if not isinstance(self.params, Mapping):
            raise TypeError("operator params must be a mapping")
        object.__setattr__(self, "params", _freeze_json(self.params))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": _thaw_json(self.params)}

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperatorSpec:
        if not isinstance(value, Mapping):
            raise TypeError("operator spec must be a mapping")
        _require_fields("operator spec", value, {"name", "params"})
        return cls(name=value["name"], params=value["params"])

    @classmethod
    def from_json(cls, value: str) -> OperatorSpec:
        decoded = json.loads(value)
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class BiasSpec:
    """A testable representation hypothesis composed of approved operators."""

    name: str
    hypothesis: str
    operators: tuple[OperatorSpec, ...]
    prediction: str
    falsification: str

    def __post_init__(self) -> None:
        for field_name in ("name", "hypothesis", "prediction", "falsification"):
            _require_text(field_name, getattr(self, field_name))
        if not isinstance(self.operators, (tuple, list)):
            raise TypeError("bias operators must be a tuple or list")
        operators = tuple(self.operators)
        if not operators:
            raise ValueError("bias must contain at least one operator")
        if not all(isinstance(operator, OperatorSpec) for operator in operators):
            raise TypeError("bias operators must contain OperatorSpec values")
        object.__setattr__(self, "operators", operators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "operators": [operator.to_dict() for operator in self.operators],
            "prediction": self.prediction,
            "falsification": self.falsification,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BiasSpec:
        if not isinstance(value, Mapping):
            raise TypeError("bias spec must be a mapping")
        _require_fields(
            "bias spec",
            value,
            {"name", "hypothesis", "operators", "prediction", "falsification"},
        )
        operator_values = value["operators"]
        if not isinstance(operator_values, list):
            raise TypeError("bias spec operators must be a list")
        return cls(
            name=value["name"],
            hypothesis=value["hypothesis"],
            operators=tuple(OperatorSpec.from_dict(item) for item in operator_values),
            prediction=value["prediction"],
            falsification=value["falsification"],
        )

    @classmethod
    def from_json(cls, value: str) -> BiasSpec:
        decoded = json.loads(value)
        return cls.from_dict(decoded)


def bias_spec_hash(spec: BiasSpec) -> str:
    """Return the stable SHA-256 identity of a canonical bias specification."""
    if not isinstance(spec, BiasSpec):
        raise TypeError("bias_spec_hash requires a BiasSpec")
    return hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest()
