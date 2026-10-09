"""Immutable records for reproducible candidate search."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from bias_optimizer.domain.bias import BiasSpec, bias_spec_hash
from bias_optimizer.domain.evaluation import Evaluation

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATION_FIELDS = {
    "accuracy_500",
    "accuracy_5000",
    "feature_dim",
    "feature_runtime_ms",
    "training_runtime_ms",
    "inference_runtime_ms",
    "confusion_matrix",
}
_RECORD_FIELDS = {
    "candidate_id",
    "generation",
    "bias",
    "evaluation",
    "parent_ids",
    "prompt",
    "model",
}


@dataclass(frozen=True, slots=True)
class SearchRecord:
    """One evaluated bias and enough provenance to reproduce its proposal."""

    generation: int
    bias: BiasSpec
    evaluation: Evaluation
    parent_ids: tuple[str, ...] = ()
    prompt: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not isinstance(self.bias, BiasSpec):
            raise TypeError("bias must be a BiasSpec")
        if not isinstance(self.evaluation, Evaluation):
            raise TypeError("evaluation must be an Evaluation")

        parent_ids = tuple(self.parent_ids)
        if not all(
            isinstance(parent_id, str) and _SHA256.fullmatch(parent_id)
            for parent_id in parent_ids
        ):
            raise ValueError("parent_ids must contain SHA-256 candidate IDs")
        if len(set(parent_ids)) != len(parent_ids):
            raise ValueError("parent_ids cannot contain duplicates")
        object.__setattr__(self, "parent_ids", parent_ids)

        if (self.prompt is None) != (self.model is None):
            raise ValueError("prompt and model metadata must be provided together")
        for field_name in ("prompt", "model"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string")

    @property
    def candidate_id(self) -> str:
        """Return the deterministic hash of this record's canonical BiasSpec."""
        return bias_spec_hash(self.bias)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record as JSON-compatible values."""
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
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
    def from_dict(cls, value: dict[str, Any]) -> SearchRecord:
        if not isinstance(value, dict):
            raise TypeError("search record must be a JSON object")
        if set(value) != _RECORD_FIELDS:
            missing = sorted(_RECORD_FIELDS - set(value))
            extra = sorted(set(value) - _RECORD_FIELDS)
            raise ValueError(
                f"invalid search record fields; missing={missing}, extra={extra}"
            )

        evaluation_value = value["evaluation"]
        if not isinstance(evaluation_value, dict):
            raise TypeError("evaluation must be a JSON object")
        if set(evaluation_value) != _EVALUATION_FIELDS:
            missing = sorted(_EVALUATION_FIELDS - set(evaluation_value))
            extra = sorted(set(evaluation_value) - _EVALUATION_FIELDS)
            raise ValueError(
                f"invalid evaluation fields; missing={missing}, extra={extra}"
            )

        record = cls(
            generation=value["generation"],
            bias=BiasSpec.from_dict(value["bias"]),
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
            parent_ids=tuple(value["parent_ids"]),
            prompt=value["prompt"],
            model=value["model"],
        )
        candidate_id = value["candidate_id"]
        if not isinstance(candidate_id, str) or not _SHA256.fullmatch(candidate_id):
            raise ValueError("candidate_id must be a SHA-256 hash")
        if candidate_id != record.candidate_id:
            raise ValueError("candidate_id does not match the canonical BiasSpec")
        return record

    @classmethod
    def from_json(cls, value: str) -> SearchRecord:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("search record is not valid JSON") from exc
        return cls.from_dict(payload)
