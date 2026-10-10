"""Disk-backed feature matrices keyed by exact split and fitted state."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray


class FeatureMatrixCache:
    """Cache finite float32 matrices without mixing splits or PCA fits."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else None
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(
        *,
        dataset_fingerprint: str,
        split_id: str,
        representation_id: str,
        representation_version: str,
        train_size: int,
        seed: int,
        fitted_state_id: str,
        role: str,
    ) -> str:
        payload = {
            "dataset_fingerprint": dataset_fingerprint,
            "split_id": split_id,
            "representation_id": representation_id,
            "representation_version": representation_version,
            "train_size": int(train_size),
            "seed": int(seed),
            "fitted_state_id": fitted_state_id,
            "role": role,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get_or_compute(
        self,
        *,
        key: str,
        compute: Callable[[], NDArray],
        expected_rows: int,
        expected_columns: int,
    ) -> NDArray[np.float32]:
        if self.root is None:
            return self._validate(compute(), expected_rows, expected_columns)
        path = self.root / f"{key}.npy"
        if path.exists():
            cached = np.load(path, allow_pickle=False)
            return self._validate(cached, expected_rows, expected_columns)

        matrix = self._validate(compute(), expected_rows, expected_columns)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(dir=self.root, suffix=".npy", delete=False) as stream:
                temporary = Path(stream.name)
            np.save(temporary, matrix, allow_pickle=False)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return matrix

    @staticmethod
    def _validate(values: NDArray, rows: int, columns: int) -> NDArray[np.float32]:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (rows, columns):
            raise ValueError(f"feature matrix shape {matrix.shape}; expected {(rows, columns)}")
        if not np.isfinite(matrix).all():
            raise ValueError("feature matrix must contain only finite values")
        return np.ascontiguousarray(matrix)
