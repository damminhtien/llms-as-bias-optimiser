"""Paired contribution analysis for raw/HOG features augmented with frozen V3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score

from bias_optimizer.evaluation.representations import (
    RepresentationDefinition,
    build_concatenated_representation,
)
from bias_optimizer.evaluation.schema import EvaluationResult


def complementarity_definition(
    registry: dict[str, RepresentationDefinition], anchor: str, v3_name: str
) -> RepresentationDefinition:
    """Create an aligned HOG+V3 or raw+V3 representation definition."""
    return build_concatenated_representation(registry, anchor, v3_name)


def paired_complementarity(
    baseline: Sequence[EvaluationResult], augmented: Sequence[EvaluationResult]
) -> list[dict[str, object]]:
    """Compare the same evaluation examples and training split for each pair."""
    base_by_key = {_pair_key(result): result for result in baseline}
    augmented_by_key = {_pair_key(result): result for result in augmented}
    if base_by_key.keys() != augmented_by_key.keys():
        raise ValueError("complementarity runs must have matching dataset/size/seed keys")
    rows: list[dict[str, object]] = []
    for key in sorted(base_by_key):
        base = base_by_key[key]
        candidate = augmented_by_key[key]
        if not np.array_equal(base.targets, candidate.targets):
            raise ValueError("complementarity targets are not aligned")
        if not np.array_equal(base.evaluation_indices, candidate.evaluation_indices):
            raise ValueError("complementarity evaluation indices are not aligned")
        baseline_accuracy = float(accuracy_score(base.targets, base.predictions))
        augmented_accuracy = float(accuracy_score(candidate.targets, candidate.predictions))
        rows.append({
            "dataset": base.key.dataset,
            "train_size": base.key.train_size,
            "seed": base.key.seed,
            "anchor": base.key.representation,
            "augmented_representation": candidate.key.representation,
            "probe": base.key.probe,
            "baseline_accuracy": baseline_accuracy,
            "augmented_accuracy": augmented_accuracy,
            "delta_complementarity": augmented_accuracy - baseline_accuracy,
            "evaluation_split_id": base.key.evaluation_split_id,
        })
    return rows


def _pair_key(result: EvaluationResult) -> tuple[str, int, int, str]:
    return (
        result.key.dataset,
        result.key.train_size,
        result.key.seed,
        result.key.probe,
    )
