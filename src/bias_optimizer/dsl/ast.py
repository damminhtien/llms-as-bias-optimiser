"""Immutable, JSON-compatible expression trees for discovered representations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("expression parameter keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise TypeError(
        f"parameters must be finite JSON values, got {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Expr:
    """A pure DSL node. It never contains executable Python or callables."""

    op: str
    args: tuple[Expr, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op.strip():
            raise ValueError("expression op must be a non-empty string")
        if (
            not self.op.isascii()
            or not self.op.replace("_", "").isalnum()
            or self.op.lower() != self.op
        ):
            raise ValueError("expression op must be a lowercase identifier")
        args = tuple(self.args)
        if not all(isinstance(arg, Expr) for arg in args):
            raise TypeError("expression args must contain Expr values")
        if not isinstance(self.params, Mapping):
            raise TypeError("expression params must be a mapping")
        frozen = _freeze(self.params)
        object.__setattr__(self, "args", args)
        object.__setattr__(self, "params", frozen)

    @property
    def parameter_values(self) -> dict[str, Any]:
        return _thaw(self.params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [arg.to_dict() for arg in self.args],
            "params": self.parameter_values,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def __hash__(self) -> int:
        return hash(self.to_json())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Expr:
        budget = [0]

        def parse(node: Mapping[str, Any], depth: int) -> Expr:
            if depth > 128:
                raise ValueError("expression nesting exceeds the parser limit")
            budget[0] += 1
            if budget[0] > 127:
                raise ValueError("expression exceeds the 127-node parser limit")
            expected = {"op", "args", "params"}
            if (
                not isinstance(node, Mapping)
                or "op" not in node
                or set(node) - expected
            ):
                raise ValueError("expression may contain only op, args, and params")
            args = node.get("args", [])
            params = node.get("params", {})
            if not isinstance(args, list) or not isinstance(params, Mapping):
                raise TypeError(
                    "expression args must be a list and params must be an object"
                )
            return cls(
                op=node["op"],
                args=tuple(parse(child, depth + 1) for child in args),
                params=params,
            )

        return parse(value, 0)

    @classmethod
    def from_json(cls, value: str) -> Expr:
        decoded = json.loads(value)
        return cls.from_dict(decoded)
