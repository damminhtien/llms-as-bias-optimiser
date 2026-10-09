"""Tests for the post-selection test-set evaluation boundary."""

import hashlib
import json

import numpy as np
import pytest

from bias_optimizer.data.mnist import MNISTFinalData
from bias_optimizer.domain.bias import bias_spec_hash
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.ml import final_evaluation


def _write_frozen_selection(tmp_path, *, frozen: bool = True):
    archive_path = tmp_path / "search.jsonl"
    archive_path.write_text("frozen search archive\n", encoding="utf-8")
    bias = initial_human_biases()[0]
    selection = {
        "finalist_selection_frozen": frozen,
        "test_set_accessed": False,
        "search_archive": str(archive_path),
        "search_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "finalists": [
            {
                "candidate_id": bias_spec_hash(bias),
                "bias": bias.to_dict(),
                "selection_roles": ["top_5_low_data_accuracy_rank_3"],
            }
        ],
    }
    finalists_path = tmp_path / "finalists.json"
    finalists_path.write_text(json.dumps(selection), encoding="utf-8")
    return finalists_path, archive_path


def _small_final_data() -> MNISTFinalData:
    train_labels = np.repeat(np.arange(10, dtype=np.int64), 4)
    train_images = np.zeros((len(train_labels), 28, 28), dtype=np.float32)
    for index, label in enumerate(train_labels):
        train_images[index, label : label + 2, label : label + 2] = 1.0
    test_labels = np.arange(10, dtype=np.int64)
    test_images = np.zeros((len(test_labels), 28, 28), dtype=np.float32)
    for index, label in enumerate(test_labels):
        test_images[index, label : label + 2, label : label + 2] = 1.0
    return MNISTFinalData(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def test_evaluation_rejects_unfrozen_selection_before_loading_test_data(
    tmp_path, monkeypatch
) -> None:
    finalists_path, _ = _write_frozen_selection(tmp_path, frozen=False)

    def fail_if_loaded(_data_dir):
        raise AssertionError("test data must not load before finalist freeze")

    monkeypatch.setattr(final_evaluation, "load_mnist_final_data", fail_if_loaded)
    with pytest.raises(ValueError, match="not frozen"):
        final_evaluation.run_final_evaluation(finalists_path=finalists_path)


def test_evaluation_rejects_changed_search_archive_before_loading_test_data(
    tmp_path, monkeypatch
) -> None:
    finalists_path, archive_path = _write_frozen_selection(tmp_path)
    archive_path.write_text("changed archive\n", encoding="utf-8")

    def fail_if_loaded(_data_dir):
        raise AssertionError("test data must not load after archive changes")

    monkeypatch.setattr(final_evaluation, "load_mnist_final_data", fail_if_loaded)
    with pytest.raises(ValueError, match="changed after finalist selection"):
        final_evaluation.run_final_evaluation(finalists_path=finalists_path)


def test_evaluation_reports_seed_summaries_and_confusion_counts(tmp_path) -> None:
    finalists_path, _ = _write_frozen_selection(tmp_path)
    output_path = tmp_path / "final_evaluation.json"
    report = final_evaluation.run_final_evaluation(
        finalists_path=finalists_path,
        output_path=output_path,
        cache_dir=tmp_path / "features",
        train_sizes=(10,),
        seeds=(11, 23),
        final_data=_small_final_data(),
    )

    assert report["test_set_accessed"] is True
    assert report["finalist_selection"]["frozen"] is True
    assert len(report["results"]) == 3
    raw = report["results"][0]
    assert raw["summaries"]["10"]["accuracy_std"] >= 0
    assert sum(map(sum, raw["summaries"]["10"]["confusion_matrix_sum"])) == 20
    assert len(raw["runs"]["10"]) == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
