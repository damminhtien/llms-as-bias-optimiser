"""Durable evaluation records for program-synthesis search runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class ProgramSearchFailure:
    """Durable reason a structurally valid proposal was not archived."""

    generation: int
    track: str
    mechanism_focus: str
    bias: ProgramBiasSpec
    failure_type: str
    message: str

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self.track not in {"augmentation", "discovery"}:
            raise ValueError("track must be augmentation or discovery")
        if not isinstance(self.bias, ProgramBiasSpec):
            raise TypeError("bias must be a ProgramBiasSpec")
        for name in ("mechanism_focus", "failure_type", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    @property
    def candidate_id(self) -> str:
        return program_bias_hash(self.bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "track": self.track,
            "mechanism_focus": self.mechanism_focus,
            "bias": self.bias.to_dict(),
            "failure_type": self.failure_type,
            "message": self.message[:1_000],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class ProgramSearchRecord:
    generation: int
    track: str
    bias: ProgramBiasSpec
    evaluation: Evaluation
    source: str
    order: str
    spatial: str
    composition: str
    complexity: str
    novelty: float
    parent_ids: tuple[str, ...] = ()
    prompt: str | None = None
    model: str | None = None
    legacy_niche: str | None = None
    legacy_complexity_bin: str | None = None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self.track not in {"augmentation", "discovery"}:
            raise ValueError("track must be augmentation or discovery")
        if not isinstance(self.bias, ProgramBiasSpec):
            raise TypeError("bias must be a ProgramBiasSpec")
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError("evaluation must be an Evaluation")
        descriptor_values = {
            "source": (
                self.source,
                {"topology", "graph", "path_geometry", "pixel", "mixed"},
            ),
            "order": (self.order, {"orderless", "first_order", "higher_order"}),
            "spatial": (self.spatial, {"global", "localized"}),
            "composition": (self.composition, {"single", "composite"}),
            "complexity": (self.complexity, {"compact", "moderate", "deep"}),
        }
        for name, (value, allowed) in descriptor_values.items():
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"invalid descriptor {name}: {value!r}")
        if not isfinite(self.novelty) or not 0 <= self.novelty <= 1:
            raise ValueError("novelty must be finite and between 0 and 1")
        parents = tuple(self.parent_ids)
        if not all(
            isinstance(item, str) and _SHA256.fullmatch(item) for item in parents
        ):
            raise ValueError("parent_ids must contain SHA-256 candidate IDs")
        if len(set(parents)) != len(parents):
            raise ValueError("parent_ids cannot contain duplicates")
        object.__setattr__(self, "parent_ids", parents)
        if (self.prompt is None) != (self.model is None):
            raise ValueError("prompt and model metadata must be provided together")
        for name in ("prompt", "model"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string")
        if (self.legacy_niche is None) != (self.legacy_complexity_bin is None):
            raise ValueError("legacy descriptor fields must be provided together")

    @property
    def candidate_id(self) -> str:
        return program_bias_hash(self.bias)

    @property
    def descriptor_cell(self) -> tuple[str, str, str, str, str]:
        return (
            self.source,
            self.order,
            self.spatial,
            self.composition,
            self.complexity,
        )

    @property
    def descriptor(self) -> dict[str, str]:
        return {
            "source": self.source,
            "order": self.order,
            "spatial": self.spatial,
            "composition": self.composition,
            "complexity": self.complexity,
        }

    @property
    def niche(self) -> str:
        """Compatibility label for older reporting code."""
        if self.legacy_niche is not None:
            return self.legacy_niche
        return "/".join(self.descriptor_cell[:4])

    @property
    def complexity_bin(self) -> str:
        """Compatibility alias for the fifth descriptor axis."""
        if self.legacy_complexity_bin is not None:
            return self.legacy_complexity_bin
        return self.complexity

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "track": self.track,
            "bias": self.bias.to_dict(),
            "evaluation": {
                "accuracy_500": self.evaluation.accuracy_500,
                "accuracy_5000": self.evaluation.accuracy_5000,
                "feature_dim": self.evaluation.feature_dim,
                "feature_runtime_ms": self.evaluation.feature_runtime_ms,
                "training_runtime_ms": self.evaluation.training_runtime_ms,
                "inference_runtime_ms": self.evaluation.inference_runtime_ms,
                "confusion_matrix": self.evaluation.confusion_matrix.tolist(),
            },
            "descriptor": self.descriptor,
            "novelty": self.novelty,
            "parent_ids": list(self.parent_ids),
            "prompt": self.prompt,
            "model": self.model,
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
    def from_dict(cls, value: dict[str, Any]) -> ProgramSearchRecord:
        if not isinstance(value, dict):
            raise TypeError("program search record must be a JSON object")
        legacy_expected = {
            "candidate_id",
            "generation",
            "track",
            "bias",
            "evaluation",
            "niche",
            "complexity_bin",
            "novelty",
            "parent_ids",
            "prompt",
            "model",
        }
        expected = legacy_expected - {"niche", "complexity_bin"} | {"descriptor"}
        legacy = set(value) == legacy_expected
        if not legacy and set(value) != expected:
            raise ValueError("program search record has unexpected fields")
        evaluation_value = value["evaluation"]
        evaluation_fields = {
            "accuracy_500",
            "accuracy_5000",
            "feature_dim",
            "feature_runtime_ms",
            "training_runtime_ms",
            "inference_runtime_ms",
            "confusion_matrix",
        }
        if (
            not isinstance(evaluation_value, dict)
            or set(evaluation_value) != evaluation_fields
        ):
            raise ValueError("program evaluation has unexpected fields")
        bias = ProgramBiasSpec.from_dict(value["bias"])
        if legacy:
            # V2 archives remain readable. Their one-axis niche is ignored and
            # the V3 behavioral descriptor is recomputed from the frozen AST.
            from bias_optimizer.novelty.descriptors import describe_program

            descriptor = describe_program(bias.program).to_dict()
        else:
            descriptor = value["descriptor"]
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "source",
                "order",
                "spatial",
                "composition",
                "complexity",
            }:
                raise ValueError("program descriptor has unexpected fields")
        record = cls(
            generation=value["generation"],
            track=value["track"],
            bias=bias,
            evaluation=Evaluation(
                accuracy_500=evaluation_value["accuracy_500"],
                accuracy_5000=evaluation_value["accuracy_5000"],
                feature_dim=evaluation_value["feature_dim"],
                feature_runtime_ms=evaluation_value["feature_runtime_ms"],
                training_runtime_ms=evaluation_value["training_runtime_ms"],
                inference_runtime_ms=evaluation_value["inference_runtime_ms"],
                confusion_matrix=np.asarray(
                    evaluation_value["confusion_matrix"], dtype=np.int64
                ),
            ),
            source=descriptor["source"],
            order=descriptor["order"],
            spatial=descriptor["spatial"],
            composition=descriptor["composition"],
            complexity=descriptor["complexity"],
            novelty=value["novelty"],
            parent_ids=tuple(value["parent_ids"]),
            prompt=value["prompt"],
            model=value["model"],
            legacy_niche=value["niche"] if legacy else None,
            legacy_complexity_bin=value["complexity_bin"] if legacy else None,
        )
        if value["candidate_id"] != record.candidate_id:
            raise ValueError("candidate_id does not match the canonical program bias")
        return record

    @classmethod
    def from_json(cls, value: str) -> ProgramSearchRecord:
        payload = json.loads(value)
        return cls.from_dict(payload)
