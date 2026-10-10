from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from bias_optimizer.evaluation.protocol import (
    EvaluationDataset,
    SplitRegistry,
    load_frozen_protocol,
)


def _dataset() -> EvaluationDataset:
    rng = np.random.default_rng(4)
    train_images = rng.random((300, 28, 28), dtype=np.float32)
    train_labels = np.repeat(np.arange(3, dtype=np.int64), 100)
    eval_images = rng.random((45, 28, 28), dtype=np.float32)
    eval_labels = np.repeat(np.arange(3, dtype=np.int64), 15)
    return EvaluationDataset(
        name="synthetic",
        train_images=train_images,
        train_labels=train_labels,
        evaluation_images=eval_images,
        evaluation_labels=eval_labels,
        evaluation_indices=np.arange(45, dtype=np.int64),
        fingerprint="synthetic-dataset-v1",
        evaluation_split_id="synthetic-validation-v1",
        evaluation_partition="synthetic-validation",
        provenance={"source": "unit-test"},
    )


def test_seed_indices_are_stable_shared_and_distinct() -> None:
    dataset = _dataset()
    splits = SplitRegistry()
    first = splits.indices(dataset.train_labels, 90, 11)
    repeated = splits.indices(dataset.train_labels, 90, 11)
    other_seed = splits.indices(dataset.train_labels, 90, 23)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other_seed)
    assert splits.split_id(dataset, first) == splits.split_id(dataset, repeated)


def test_protocol_hash_is_stable_and_frozen_sources_match_manifest() -> None:
    protocol, manifest = load_frozen_protocol()
    protocol_bytes = Path("results/v31/protocol.json").read_bytes()
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    assert digest == manifest["protocol"]["sha256"]
    assert protocol["v3_search_enabled"] is False
    assert len(manifest["v3_finalist_ast_sha256"]) == 5
    assert json.loads(protocol_bytes)["primary_matrix_representations"]
