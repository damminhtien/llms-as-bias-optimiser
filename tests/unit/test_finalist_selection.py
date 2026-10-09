"""Tests for deterministic finalist selection and search freeze metadata."""

import hashlib
import json

import numpy as np

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.search import SearchRecord
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.search.archive import SearchArchive
from bias_optimizer.search.selection import select_finalists


def _bias(name: str, operator: str) -> BiasSpec:
    return BiasSpec(
        name=name,
        hypothesis=f"Hypothesis for {name}.",
        operators=(OperatorSpec(operator),),
        prediction="Validation accuracy should improve.",
        falsification="Reject if validation accuracy does not improve.",
    )


def _record(
    bias: BiasSpec,
    *,
    accuracy_500: float,
    feature_dim: int,
    runtime_ms: float,
) -> SearchRecord:
    return SearchRecord(
        generation=0,
        bias=bias,
        evaluation=Evaluation(
            accuracy_500=accuracy_500,
            accuracy_5000=0.8,
            feature_dim=feature_dim,
            feature_runtime_ms=runtime_ms,
            training_runtime_ms=0.0,
            inference_runtime_ms=0.0,
            confusion_matrix=np.eye(10, dtype=np.int64),
        ),
    )


def test_select_finalists_freezes_top_small_fast_and_stroke_flow(tmp_path) -> None:
    archive_path = tmp_path / "search.jsonl"
    output_path = tmp_path / "finalists.json"
    archive = SearchArchive(archive_path)
    records = [
        _record(
            _bias(f"candidate_{index}", "symmetry"),
            accuracy_500=0.9 - index * 0.03,
            feature_dim=20 + index,
            runtime_ms=20.0 + index,
        )
        for index in range(6)
    ]
    small = _record(
        _bias("smallest", "topology"),
        accuracy_500=0.5,
        feature_dim=4,
        runtime_ms=10,
    )
    fast = _record(
        _bias("fastest", "spatial"),
        accuracy_500=0.55,
        feature_dim=10,
        runtime_ms=1,
    )
    stroke = _record(
        initial_human_biases()[-1],
        accuracy_500=0.61,
        feature_dim=20,
        runtime_ms=8,
    )
    for record in (*records, small, fast, stroke):
        archive.add(record)

    result = select_finalists(archive_path=archive_path, output_path=output_path)

    assert result["finalist_selection_frozen"] is True
    assert result["test_set_accessed"] is False
    assert (
        result["search_archive_sha256"]
        == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )
    finalists = result["finalists"]
    by_name = {item["bias"]["name"]: item for item in finalists}
    assert by_name["candidate_0"]["selection_roles"] == [
        "top_5_low_data_accuracy_rank_1"
    ]
    assert "smallest_representation" in by_name["smallest"]["selection_roles"]
    assert "fastest_representation" in by_name["fastest"]["selection_roles"]
    assert (
        "original_stroke_flow_hypothesis"
        in by_name["topology_direction_curvature"]["selection_roles"]
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
