"""Tests for deterministic search records and JSONL persistence."""

import numpy as np
import pytest

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec, bias_spec_hash
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.search import SearchRecord
from bias_optimizer.search.archive import SearchArchive


def _bias(name: str) -> BiasSpec:
    return BiasSpec(
        name=name,
        hypothesis=f"Hypothesis for {name}.",
        operators=(OperatorSpec("topology"),),
        prediction="Validation accuracy should improve.",
        falsification="Reject if validation accuracy does not improve.",
    )


def _evaluation(accuracy_500: float = 0.8) -> Evaluation:
    matrix = np.zeros((10, 10), dtype=np.int64)
    matrix[3, 5] = 7
    return Evaluation(
        accuracy_500=accuracy_500,
        accuracy_5000=0.9,
        feature_dim=4,
        feature_runtime_ms=10.0,
        training_runtime_ms=5.0,
        inference_runtime_ms=2.0,
        confusion_matrix=matrix,
    )


def _record(
    name: str, *, accuracy_500: float = 0.8, generation: int = 0
) -> SearchRecord:
    return SearchRecord(
        generation=generation,
        bias=_bias(name),
        evaluation=_evaluation(accuracy_500),
        parent_ids=(bias_spec_hash(_bias("parent")),) if generation else (),
        prompt="Propose a compact validation hypothesis." if generation else None,
        model="qwen3.5:35b-mlx" if generation else None,
    )


def test_search_record_has_deterministic_hash_and_round_trips() -> None:
    record = _record("candidate", generation=1)

    restored = SearchRecord.from_json(record.to_json())

    assert record.candidate_id == bias_spec_hash(record.bias)
    assert restored.to_json() == record.to_json()
    np.testing.assert_array_equal(
        restored.evaluation.confusion_matrix,
        record.evaluation.confusion_matrix,
    )
    assert restored.parent_ids == record.parent_ids
    assert restored.model == "qwen3.5:35b-mlx"


def test_search_record_rejects_tampered_hash_and_unknown_fields() -> None:
    payload = _record("candidate").to_dict()
    payload["candidate_id"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        SearchRecord.from_dict(payload)

    payload = _record("candidate").to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match=r"extra=\['unexpected'\]"):
        SearchRecord.from_dict(payload)


def test_search_archive_persists_reloads_deduplicates_and_ranks(tmp_path) -> None:
    path = tmp_path / "nested" / "search.jsonl"
    archive = SearchArchive(path)
    lower = _record("lower", accuracy_500=0.7)
    higher = _record("higher", accuracy_500=0.9)

    archive.add(lower)
    archive.add(lower)
    archive.add(higher)

    assert archive.contains(lower.candidate_id)
    assert archive.get(higher.candidate_id) is not None
    assert [item.candidate_id for item in archive.top(1)] == [higher.candidate_id]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    reloaded = SearchArchive(path)
    assert [item.candidate_id for item in reloaded.records] == [
        lower.candidate_id,
        higher.candidate_id,
    ]
    assert [item.candidate_id for item in reloaded.top(2)] == [
        higher.candidate_id,
        lower.candidate_id,
    ]


def test_search_archive_rejects_conflicting_duplicate_and_corrupt_jsonl(
    tmp_path,
) -> None:
    path = tmp_path / "search.jsonl"
    archive = SearchArchive(path)
    record = _record("candidate")
    archive.add(record)

    with pytest.raises(ValueError, match="different record"):
        archive.add(_record("candidate", generation=1))

    path.write_text(path.read_text(encoding="utf-8") + "{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        SearchArchive(path)


def test_search_archive_top_requires_positive_integer(tmp_path) -> None:
    archive = SearchArchive(tmp_path / "search.jsonl")

    with pytest.raises(ValueError, match="positive integer"):
        archive.top(0)
    with pytest.raises(ValueError, match="positive integer"):
        archive.top(True)
