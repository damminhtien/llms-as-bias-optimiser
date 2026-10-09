"""Atomic, content-addressed storage for deterministic feature matrices."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

_BIAS_HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class FeatureCache:
    """Persist feature matrices under a bias hash and explicit dataset key."""

    root: Path = Path("cache/features")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def get(self, bias_hash: str, dataset_key: str) -> NDArray[np.float32] | None:
        path = self._path(bias_hash, dataset_key)
        if not path.exists():
            return None
        matrix = np.load(path, allow_pickle=False)
        validated = self._validate_matrix(matrix)
        validated.setflags(write=False)
        return validated

    def set(
        self, bias_hash: str, dataset_key: str, matrix: NDArray[np.float32]
    ) -> None:
        path = self._path(bias_hash, dataset_key)
        validated = self._validate_matrix(matrix)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.save(temporary_file, validated, allow_pickle=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _path(self, bias_hash: str, dataset_key: str) -> Path:
        if not isinstance(bias_hash, str) or _BIAS_HASH.fullmatch(bias_hash) is None:
            raise ValueError("bias_hash must be a lowercase SHA-256 hex digest")
        if not isinstance(dataset_key, str) or not dataset_key.strip():
            raise ValueError("dataset_key must be a non-empty string")
        dataset_hash = hashlib.sha256(dataset_key.encode("utf-8")).hexdigest()
        return self.root / bias_hash / f"{dataset_hash}.npy"

    @staticmethod
    def _validate_matrix(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        result = np.asarray(matrix, dtype=np.float32)
        if result.ndim != 2:
            raise ValueError("cached feature matrices must be two-dimensional")
        if not np.isfinite(result).all():
            raise ValueError("cached feature matrices must contain finite values")
        return result
