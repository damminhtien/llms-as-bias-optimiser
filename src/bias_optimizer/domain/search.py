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
_OPTIONAL_RECORD_FIELDS = {
    "proposal_category",
    "failure_feedback",
    "record_type",
    "base_candidate_id",
    "removed_operator",
}
_PROPOSAL_CATEGORIES = {
    "exploitation",
    "failure-driven",
    "simplification",
    "exploration",
}


@dataclass(frozen=True, slots=True)
class FailureFeedback:
    """Validation confusion count before and after a targeted proposal."""

    actual_digit: int
    predicted_digit: int
    baseline_count: int
    candidate_count: int

    def __post_init__(self) -> None:
        if type(self.actual_digit) is not int or not 0 <= self.actual_digit <= 9:
            raise ValueError("actual_digit must be between 0 and 9")
        if type(self.predicted_digit) is not int or not 0 <= self.predicted_digit <= 9:
            raise ValueError("predicted_digit must be between 0 and 9")
        if self.actual_digit == self.predicted_digit:
            raise ValueError("failure feedback must describe a confusion pair")
        if type(self.baseline_count) is not int or self.baseline_count <= 0:
            raise ValueError("baseline_count must be a positive integer")
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative integer")

    @property
    def improvement(self) -> int:
        """Return positive values when the targeted confusion count decreased."""
        return self.baseline_count - self.candidate_count

    def to_dict(self) -> dict[str, int]:
        return {
            "actual_digit": self.actual_digit,
            "predicted_digit": self.predicted_digit,
            "baseline_count": self.baseline_count,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FailureFeedback:
        expected = {
            "actual_digit",
            "predicted_digit",
            "baseline_count",
            "candidate_count",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("invalid failure feedback fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class SearchRecord:
    """One evaluated bias and enough provenance to reproduce its proposal."""

    generation: int
    bias: BiasSpec
    evaluation: Evaluation
    parent_ids: tuple[str, ...] = ()
    prompt: str | None = None
    model: str | None = None
    proposal_category: str | None = None
    failure_feedback: tuple[FailureFeedback, ...] = ()
    record_type: str = "candidate"
    base_candidate_id: str | None = None
    removed_operator: str | None = None

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
        if self.proposal_category is not None:
            if self.proposal_category not in _PROPOSAL_CATEGORIES:
                raise ValueError("proposal_category is not recognized")
            if self.prompt is None:
                raise ValueError("proposal_category requires prompt/model metadata")
        feedback = tuple(self.failure_feedback)
        if not all(isinstance(item, FailureFeedback) for item in feedback):
            raise TypeError("failure_feedback must contain FailureFeedback values")
        if feedback and self.proposal_category != "failure-driven":
            raise ValueError("failure feedback belongs to failure-driven proposals")
        object.__setattr__(self, "failure_feedback", feedback)

        if self.record_type not in {"candidate", "ablation"}:
            raise ValueError("record_type must be candidate or ablation")
        if self.record_type == "candidate":
            if self.base_candidate_id is not None or self.removed_operator is not None:
                raise ValueError("candidate records cannot include ablation metadata")
        else:
            if not isinstance(self.base_candidate_id, str) or not _SHA256.fullmatch(
                self.base_candidate_id
            ):
                raise ValueError("ablation records require a base candidate ID")
            if self.base_candidate_id not in parent_ids:
                raise ValueError("ablation base candidate must be listed as a parent")
            if (
                not isinstance(self.removed_operator, str)
                or not self.removed_operator.strip()
            ):
                raise ValueError("ablation records require a removed operator")
            if (
                self.prompt is not None
                or self.proposal_category is not None
                or feedback
            ):
                raise ValueError("ablation records cannot contain proposal metadata")

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
            "proposal_category": self.proposal_category,
            "failure_feedback": [item.to_dict() for item in self.failure_feedback],
            "record_type": self.record_type,
            "base_candidate_id": self.base_candidate_id,
            "removed_operator": self.removed_operator,
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
        missing = sorted(_RECORD_FIELDS - set(value))
        extra = sorted(set(value) - _RECORD_FIELDS - _OPTIONAL_RECORD_FIELDS)
        if missing or extra:
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
            proposal_category=value.get("proposal_category"),
            failure_feedback=tuple(
                FailureFeedback.from_dict(item)
                for item in value.get("failure_feedback", ())
            ),
            record_type=value.get("record_type", "candidate"),
            base_candidate_id=value.get("base_candidate_id"),
            removed_operator=value.get("removed_operator"),
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
