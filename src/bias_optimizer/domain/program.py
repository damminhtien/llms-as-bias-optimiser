"""Research hypotheses whose representations are typed DSL programs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from bias_optimizer.dsl.ast import Expr

_MAX_NAME_LENGTH = 128
_MAX_DESCRIPTION_LENGTH = 2_000


def _text(name: str, value: object, *, max_length: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")


@dataclass(frozen=True, slots=True)
class ProgramBiasSpec:
    """A falsifiable mechanism plus an executable representation expression."""

    name: str
    hypothesis: str
    mechanism: str
    program: Expr
    prediction: str
    falsification: str

    def __post_init__(self) -> None:
        _text("name", self.name, max_length=_MAX_NAME_LENGTH)
        for field_name in (
            "hypothesis",
            "mechanism",
            "prediction",
            "falsification",
        ):
            _text(
                field_name,
                getattr(self, field_name),
                max_length=_MAX_DESCRIPTION_LENGTH,
            )
        if not isinstance(self.program, Expr):
            raise TypeError("program must be an Expr")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "program": self.program.to_dict(),
            "prediction": self.prediction,
            "falsification": self.falsification,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProgramBiasSpec:
        expected = {
            "name",
            "hypothesis",
            "mechanism",
            "program",
            "prediction",
            "falsification",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("program bias must contain exactly the required fields")
        if not isinstance(value["program"], Mapping):
            raise TypeError("program must be a JSON object")
        return cls(
            name=value["name"],
            hypothesis=value["hypothesis"],
            mechanism=value["mechanism"],
            program=Expr.from_dict(value["program"]),
            prediction=value["prediction"],
            falsification=value["falsification"],
        )

    @classmethod
    def from_json(cls, value: str) -> ProgramBiasSpec:
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise TypeError("program bias JSON must decode to an object")
        return cls.from_dict(decoded)


def program_bias_hash(spec: ProgramBiasSpec) -> str:
    """Hash the full, canonical hypothesis and expression for durable identity."""
    if not isinstance(spec, ProgramBiasSpec):
        raise TypeError("program_bias_hash requires a ProgramBiasSpec")
    return hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest()
