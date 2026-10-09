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
class ProgramSearchRecord:
    generation: int
    track: str
    bias: ProgramBiasSpec
    evaluation: Evaluation
    niche: str
    complexity_bin: str
    novelty: float
    parent_ids: tuple[str, ...] = ()
    prompt: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if self.track not in {"augmentation", "discovery"}:
            raise ValueError("track must be augmentation or discovery")
        if not isinstance(self.bias, ProgramBiasSpec):
            raise TypeError("bias must be a ProgramBiasSpec")
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError("evaluation must be an Evaluation")
        if (
            not isinstance(self.niche, str)
            or not self.niche.strip()
            or not isinstance(self.complexity_bin, str)
            or not self.complexity_bin.strip()
        ):
            raise ValueError("niche and complexity_bin must be non-empty")
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

    @property
    def candidate_id(self) -> str:
        return program_bias_hash(self.bias)

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
            "niche": self.niche,
            "complexity_bin": self.complexity_bin,
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
        expected = {
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
        if set(value) != expected:
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
        record = cls(
            generation=value["generation"],
            track=value["track"],
            bias=ProgramBiasSpec.from_dict(value["bias"]),
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
            niche=value["niche"],
            complexity_bin=value["complexity_bin"],
            novelty=value["novelty"],
            parent_ids=tuple(value["parent_ids"]),
            prompt=value["prompt"],
            model=value["model"],
        )
        if value["candidate_id"] != record.candidate_id:
            raise ValueError("candidate_id does not match the canonical program bias")
        return record

    @classmethod
    def from_json(cls, value: str) -> ProgramSearchRecord:
        payload = json.loads(value)
        return cls.from_dict(payload)
