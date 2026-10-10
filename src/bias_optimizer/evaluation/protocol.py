"""Frozen protocol loading, shared split assignment, and dataset adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from bias_optimizer.data.mnist import MNISTFinalData, MNISTSearchData
from bias_optimizer.data.transfer import TransferDataset
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """Training pool and one fixed evaluation split with explicit provenance."""

    name: str
    train_images: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    evaluation_images: NDArray[np.float32]
    evaluation_labels: NDArray[np.int64]
    evaluation_indices: NDArray[np.int64]
    fingerprint: str
    evaluation_split_id: str
    evaluation_partition: str
    provenance: dict[str, str]

    def __post_init__(self) -> None:
        train_images = _images(self.train_images, "train_images")
        evaluation_images = _images(self.evaluation_images, "evaluation_images")
        train_labels = _labels(self.train_labels, len(train_images), "train_labels")
        evaluation_labels = _labels(
            self.evaluation_labels, len(evaluation_images), "evaluation_labels"
        )
        evaluation_indices = np.asarray(self.evaluation_indices, dtype=np.int64)
        if evaluation_indices.shape != (len(evaluation_images),):
            raise ValueError("evaluation_indices must align with evaluation images")
        if not self.name or not self.fingerprint or not self.evaluation_split_id:
            raise ValueError("dataset identity fields must be non-empty")
        for array in (train_images, train_labels, evaluation_images, evaluation_labels, evaluation_indices):
            array.setflags(write=False)
        object.__setattr__(self, "train_images", train_images)
        object.__setattr__(self, "train_labels", train_labels)
        object.__setattr__(self, "evaluation_images", evaluation_images)
        object.__setattr__(self, "evaluation_labels", evaluation_labels)
        object.__setattr__(self, "evaluation_indices", evaluation_indices)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def from_mnist_search(cls, data: MNISTSearchData) -> EvaluationDataset:
        fingerprint = _array_fingerprint(
            data.train_images, data.train_labels, data.validation_images, data.validation_labels
        )
        return cls(
            name="mnist",
            train_images=data.train_images,
            train_labels=data.train_labels,
            evaluation_images=data.validation_images,
            evaluation_labels=data.validation_labels,
            evaluation_indices=np.arange(len(data.validation_labels), dtype=np.int64),
            fingerprint=fingerprint,
            evaluation_split_id=f"mnist-validation-seed-{data.seed}-{fingerprint[:16]}",
            evaluation_partition="official_train_validation",
            provenance={"source": "OpenML mnist_784 v1", "split_seed": str(data.seed)},
        )

    @classmethod
    def from_transfer(
        cls,
        data: TransferDataset,
        *,
        test_sample_size: int = 2_000,
        test_sample_seed: int = 31_415,
    ) -> EvaluationDataset:
        selected = stratified_indices(data.test_labels, test_sample_size, test_sample_seed)
        fingerprint = hashlib.sha256(
            f"{data.name}:{data.source_sha256}:{data.preprocessing}".encode()
        ).hexdigest()
        return cls(
            name=data.name,
            train_images=data.train_images,
            train_labels=data.train_labels,
            evaluation_images=data.test_images[selected],
            evaluation_labels=data.test_labels[selected],
            evaluation_indices=selected,
            fingerprint=fingerprint,
            evaluation_split_id=f"{data.name}-test-{test_sample_seed}-{test_sample_size}",
            evaluation_partition=data.test_partition,
            provenance={
                "source": data.source,
                "source_sha256": data.source_sha256,
                "preprocessing": data.preprocessing,
            },
        )

    @classmethod
    def from_mnist_official_test(
        cls,
        search_data: MNISTSearchData,
        final_data: MNISTFinalData,
        *,
        test_sample_size: int = 2_000,
        test_sample_seed: int = 27_182,
    ) -> EvaluationDataset:
        selected = stratified_indices(final_data.test_labels, test_sample_size, test_sample_seed)
        fingerprint = _array_fingerprint(
            search_data.train_images, search_data.train_labels
        )
        sample_id = hashlib.sha256(
            fingerprint.encode("ascii") + selected.tobytes()
        ).hexdigest()
        return cls(
            name="mnist",
            train_images=search_data.train_images,
            train_labels=search_data.train_labels,
            evaluation_images=final_data.test_images[selected],
            evaluation_labels=final_data.test_labels[selected],
            evaluation_indices=selected,
            fingerprint=fingerprint,
            evaluation_split_id=f"mnist-official-test-{test_sample_seed}-{sample_id[:16]}",
            evaluation_partition="official_test",
            provenance={
                "source": "OpenML mnist_784 v1",
                "split_seed": str(search_data.seed),
                "test_sample_seed": str(test_sample_seed),
            },
        )


class SplitRegistry:
    """Resolve the same stratified training indices for every model in a run."""

    @staticmethod
    def indices(labels: NDArray, size: int, seed: int) -> NDArray[np.int64]:
        selected = stratified_indices(labels, size, seed)
        return selected

    @staticmethod
    def split_id(dataset: EvaluationDataset, indices: NDArray) -> str:
        digest = hashlib.sha256()
        digest.update(dataset.fingerprint.encode("utf-8"))
        digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        return digest.hexdigest()


def stratified_indices(labels: NDArray, size: int, seed: int) -> NDArray[np.int64]:
    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or size <= 0 or size > len(targets) or seed < 0:
        raise ValueError("invalid stratified split request")
    if size == len(targets):
        return np.arange(len(targets), dtype=np.int64)
    indices = np.arange(len(targets), dtype=np.int64)
    try:
        selected, _, _, _ = train_test_split(
            indices,
            targets,
            train_size=size,
            random_state=seed,
            stratify=targets,
        )
    except ValueError as exc:
        raise ValueError(f"cannot draw a stratified subset of size {size}") from exc
    return np.asarray(selected, dtype=np.int64)


def load_frozen_protocol(root: Path = Path(".")) -> tuple[dict, dict]:
    """Load and verify the immutable protocol and all protected source hashes."""
    root = Path(root)
    protocol_path = root / "results/v31/protocol.json"
    manifest_path = root / "results/v31/manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_protocol_hash = _sha256(protocol_path)
    expected_protocol_hash = manifest.get("protocol", {}).get("sha256")
    if actual_protocol_hash != expected_protocol_hash:
        raise ValueError("V3.1 protocol hash does not match the frozen manifest")
    checksum_line = (root / "results/v31/protocol.sha256").read_text(encoding="ascii").strip()
    if checksum_line != f"{actual_protocol_hash}  protocol.json":
        raise ValueError("protocol.sha256 does not match protocol.json")
    verify_frozen_sources(root, manifest)
    return protocol, manifest


def verify_frozen_sources(root: Path, manifest: dict) -> None:
    """Fail before evaluation if any V2/V3 source or candidate AST has changed."""
    root = Path(root)
    for relative, expected in manifest["protected_source_files"].items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen source changed: {relative}")

    v3 = json.loads((root / "results/v3_frozen_finalists.json").read_text(encoding="utf-8"))
    if v3.get("finalist_selection_frozen") is not True or v3.get("test_set_accessed") is not False:
        raise ValueError("V3 finalist manifest is not a valid pre-test freeze")
    if v3.get("search_archive_sha256") != manifest.get("v3_search_archive_sha256"):
        raise ValueError("V3 search archive provenance does not match the V3.1 freeze")
    from bias_optimizer.dsl.ast import Expr

    for item in v3["finalists"]:
        spec = ProgramBiasSpec.from_dict(item["bias"])
        if program_bias_hash(spec) != item["candidate_id"]:
            raise ValueError(f"frozen V3 candidate hash mismatch: {item['name']}")
        ast_digest = hashlib.sha256(
            json.dumps(item["bias"]["program"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if ast_digest != manifest["v3_finalist_ast_sha256"].get(item["name"]):
            raise ValueError(f"frozen V3 AST changed: {item['name']}")
        Expr.from_dict(item["bias"]["program"])

    v2 = json.loads((root / "results/v2_transfer_finalists.json").read_text(encoding="utf-8"))
    if v2.get("finalist_selection_frozen") is not True or v2.get("test_set_accessed") is not False:
        raise ValueError("V2 finalist manifest is not a valid pre-test freeze")
    selected = manifest["v2_selected_candidate"]
    match = next((item for item in v2["finalists"] if item.get("candidate_id") == selected["candidate_id"]), None)
    if match is None or program_bias_hash(ProgramBiasSpec.from_dict(match["bias"])) != selected["candidate_id"]:
        raise ValueError("frozen V2 comparison candidate changed")
    selected_ast_hash = hashlib.sha256(
        json.dumps(match["bias"]["program"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if selected_ast_hash != selected["ast_sha256"]:
        raise ValueError("frozen V2 comparison AST hash mismatch")


def _images(values: NDArray, name: str) -> NDArray[np.float32]:
    images = np.asarray(values, dtype=np.float32)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or not len(images):
        raise ValueError(f"{name} must have shape (N, 28, 28) with N > 0")
    if not np.isfinite(images).all() or np.any((images < 0) | (images > 1)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return images


def _labels(values: NDArray, expected: int, name: str) -> NDArray[np.int64]:
    labels = np.asarray(values, dtype=np.int64)
    if labels.shape != (expected,):
        raise ValueError(f"{name} must align with its images")
    return labels


def _array_fingerprint(*arrays: NDArray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
