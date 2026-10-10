"""Frozen representation registry, including train-fitted HOG-PCA controls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler, ProgramPipeline
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.baselines import HOGOperator, RawPixelsOperator
from bias_optimizer.features.pipeline import FeaturePipeline


class Representation(Protocol):
    """Image transform that may fit state using training images only."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int | None: ...

    @property
    def version(self) -> str: ...

    @property
    def fitted_state_id(self) -> str: ...

    def fit(self, images: NDArray, labels: NDArray | None = None) -> None: ...

    def transform(self, images: NDArray) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class RepresentationDefinition:
    """Stable metadata and a factory that returns a fresh representation."""

    name: str
    dimension: int
    version: str
    factory: Callable[[], Representation]
    role: str = "control"

    def create(self) -> Representation:
        value = self.factory()
        if value.name != self.name or value.dimension != self.dimension:
            raise ValueError(f"factory for {self.name} returned inconsistent metadata")
        return value


@dataclass(slots=True)
class ProgramRepresentation:
    name: str
    pipeline: ProgramPipeline
    version: str
    _fitted: bool = False

    @property
    def dimension(self) -> int:
        return self.pipeline.feature_dim

    @property
    def fitted_state_id(self) -> str:
        return "stateless"

    def fit(self, images: NDArray, labels: NDArray | None = None) -> None:
        _validate_images(images)
        if labels is not None and len(labels) != len(images):
            raise ValueError("labels must align with fit images")
        self._fitted = True

    def transform(self, images: NDArray) -> NDArray[np.float32]:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        batch = _validate_images(images)
        result = np.empty((len(batch), self.dimension), dtype=np.float32)
        for index, image in enumerate(batch):
            result[index] = self.pipeline.transform(image)
        return result

    def transform_spatial_reassignment(
        self, images: NDArray, *, target_op: str, seed: int
    ) -> NDArray[np.float32]:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        batch = _validate_images(images)
        result = np.empty((len(batch), self.dimension), dtype=np.float32)
        for index, image in enumerate(batch):
            result[index] = self.pipeline.transform_with_spatial_reassignment(
                image, target_op=target_op, seed=seed + index
            )
        return result


@dataclass(slots=True)
class FeatureRepresentation:
    name: str
    pipeline: FeaturePipeline
    version: str
    _fitted: bool = False

    @property
    def dimension(self) -> int:
        return self.pipeline.feature_dim

    @property
    def fitted_state_id(self) -> str:
        return "stateless"

    def fit(self, images: NDArray, labels: NDArray | None = None) -> None:
        _validate_images(images)
        if labels is not None and len(labels) != len(images):
            raise ValueError("labels must align with fit images")
        self._fitted = True

    def transform(self, images: NDArray) -> NDArray[np.float32]:
        if not self._fitted:
            raise RuntimeError("fit must be called before transform")
        batch = _validate_images(images)
        result = np.empty((len(batch), self.dimension), dtype=np.float32)
        for index, image in enumerate(batch):
            result[index] = self.pipeline.transform(image)
        return result


@dataclass(slots=True)
class HOGPCARepresentation:
    """PCA fitted only on the current run's training HOG feature matrix."""

    name: str
    n_components: int
    version: str = "hog-9bin-cell4-block2:pca-full-v1"
    _pca: PCA | None = field(default=None, init=False, repr=False)
    _fit_images: NDArray | None = field(default=None, init=False, repr=False)
    _fit_features: NDArray[np.float32] | None = field(default=None, init=False, repr=False)
    _state_id: str = field(default="unfitted", init=False, repr=False)

    @property
    def dimension(self) -> int:
        return self.n_components

    @property
    def fitted_state_id(self) -> str:
        return self._state_id

    def fit(self, images: NDArray, labels: NDArray | None = None) -> None:
        batch = _validate_images(images)
        if len(batch) < self.n_components:
            raise ValueError("training rows must be at least the PCA dimension")
        if labels is not None and len(labels) != len(batch):
            raise ValueError("labels must align with fit images")
        features = _transform_operator(HOGOperator(), batch)
        pca = PCA(n_components=self.n_components, svd_solver="full")
        pca.fit(features)
        state = hashlib.sha256(pca.components_.tobytes() + pca.mean_.tobytes()).hexdigest()
        self._pca = pca
        self._fit_images = batch
        self._fit_features = features
        self._state_id = state

    def transform(self, images: NDArray) -> NDArray[np.float32]:
        if self._pca is None:
            raise RuntimeError("fit must be called before transform")
        batch = _validate_images(images)
        if batch is self._fit_images and self._fit_features is not None:
            features = self._fit_features
        else:
            features = _transform_operator(HOGOperator(), batch)
        transformed = np.asarray(self._pca.transform(features), dtype=np.float32)
        if transformed.shape != (len(batch), self.n_components):
            raise ValueError("PCA returned an unexpected feature dimension")
        return transformed


@dataclass(slots=True)
class ConcatenatedRepresentation:
    name: str
    parts: tuple[Representation, ...]
    version: str

    @property
    def dimension(self) -> int:
        dimensions = [part.dimension for part in self.parts]
        if any(dimension is None for dimension in dimensions):
            return None
        return sum(int(dimension) for dimension in dimensions)

    @property
    def fitted_state_id(self) -> str:
        payload = ":".join(part.fitted_state_id for part in self.parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fit(self, images: NDArray, labels: NDArray | None = None) -> None:
        for part in self.parts:
            part.fit(images, labels)

    def transform(self, images: NDArray) -> NDArray[np.float32]:
        matrices = [part.transform(images) for part in self.parts]
        if not matrices or len({matrix.shape[0] for matrix in matrices}) != 1:
            raise ValueError("concatenated representations must preserve row alignment")
        return np.concatenate(matrices, axis=1).astype(np.float32, copy=False)


def _validate_images(images: NDArray) -> NDArray[np.float32]:
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1:] != (28, 28):
        raise ValueError("images must have shape (N, 28, 28)")
    if not np.isfinite(batch).all() or np.any((batch < 0) | (batch > 1)):
        raise ValueError("images must be finite and normalized to [0, 1]")
    return batch


def _transform_operator(operator: object, images: NDArray) -> NDArray[np.float32]:
    batch = _validate_images(images)
    dimension = int(operator.feature_dim)
    result = np.empty((len(batch), dimension), dtype=np.float32)
    for index, image in enumerate(batch):
        result[index] = operator.transform(image)  # type: ignore[attr-defined]
    return result


def _ast_sha256(value: object) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _program_pipeline(program: Expr) -> ProgramPipeline:
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    return compiler.compile(program)


def build_frozen_registry(
    root: Path = Path("."),
    *,
    include_all_v3: bool = True,
) -> dict[str, RepresentationDefinition]:
    """Build only representations named and hashed in the frozen V3.1 manifest."""
    root = Path(root)
    manifest_path = root / "results/v31/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v3 = json.loads((root / "results/v3_frozen_finalists.json").read_text(encoding="utf-8"))
    v2 = json.loads((root / "results/v2_transfer_finalists.json").read_text(encoding="utf-8"))
    definitions: dict[str, RepresentationDefinition] = {}

    for item in v3["finalists"]:
        name = str(item["name"])
        ast_hash = _ast_sha256(item["bias"]["program"])
        expected = manifest["v3_finalist_ast_sha256"].get(name)
        if expected != ast_hash or item.get("candidate_id") != _candidate_hash(item["bias"]):
            raise ValueError(f"frozen V3 AST provenance mismatch for {name}")
        if not include_all_v3 and name not in {"angle_centroid_pairwise", "regional_turn_histogram", "horizontal_turn_spatial"}:
            continue
        spec = ProgramBiasSpec.from_dict(item["bias"])
        pipeline = _program_pipeline(spec.program)
        if pipeline.feature_dim != int(item["full_validation"]["feature_dim"]):
            raise ValueError(f"frozen V3 dimension mismatch for {name}")
        definitions[name] = RepresentationDefinition(
            name=name,
            dimension=pipeline.feature_dim,
            version=f"v3:{ast_hash}",
            role="v3_finalist",
            factory=lambda n=name, p=spec.program, v=f"v3:{ast_hash}": ProgramRepresentation(n, _program_pipeline(p), v),
        )

    selected = manifest["v2_selected_candidate"]
    item = next(
        (entry for entry in v2["finalists"] if entry.get("candidate_id") == selected["candidate_id"]),
        None,
    )
    if item is None or item.get("name") != "cycle_angle_hist" or int(item.get("feature_dim", 0)) != 16:
        raise ValueError("the frozen 16D V2 cycle_angle_hist candidate is missing")
    if _ast_sha256(item["bias"]["program"]) != selected["ast_sha256"]:
        raise ValueError("frozen V2 cycle_angle_hist AST hash mismatch")
    spec = ProgramBiasSpec.from_dict(item["bias"])
    pipeline = _program_pipeline(spec.program)
    if pipeline.feature_dim != 16:
        raise ValueError("cycle_angle_hist must have 16 features")
    definitions["cycle_angle_hist"] = RepresentationDefinition(
        name="cycle_angle_hist",
        dimension=16,
        version=f"v2:{selected['ast_sha256']}",
        role="v2_control",
        factory=lambda p=spec.program, v=f"v2:{selected['ast_sha256']}:cycle_angle_hist": ProgramRepresentation("cycle_angle_hist", _program_pipeline(p), v),
    )

    zoning_program = Expr("spatial_split", (Expr("image"),), {"rows": 5, "cols": 6})
    zoning = _program_pipeline(zoning_program)
    if zoning.feature_dim != 30:
        raise ValueError("zoning_30d must have exactly 30 features")
    definitions["zoning_30d"] = RepresentationDefinition(
        name="zoning_30d", dimension=30, version="zoning-5x6-v1", role="dimension_control",
        factory=lambda p=zoning_program: ProgramRepresentation("zoning_30d", _program_pipeline(p), "zoning-5x6-v1"),
    )
    definitions["raw_pixels"] = RepresentationDefinition(
        name="raw_pixels", dimension=784, version="raw-28x28-v1", role="pixel_control",
        factory=lambda: FeatureRepresentation("raw_pixels", FeaturePipeline((RawPixelsOperator(),)), "raw-28x28-v1"),
    )
    hog = HOGOperator()
    definitions["hog"] = RepresentationDefinition(
        name="hog", dimension=hog.feature_dim, version="hog-9bin-cell4-block2-v1", role="classical_reference",
        factory=lambda: FeatureRepresentation("hog", FeaturePipeline((HOGOperator(),)), "hog-9bin-cell4-block2-v1"),
    )
    for dimension in (30, 60):
        name = f"hog_pca_{dimension}d"
        definitions[name] = RepresentationDefinition(
            name=name, dimension=dimension, version="hog-9bin-cell4-block2:pca-full-v1", role="dimension_control",
            factory=lambda n=name, d=dimension: HOGPCARepresentation(n, d),
        )
    return definitions


def build_concatenated_representation(
    registry: Mapping[str, RepresentationDefinition], anchor: str, added: str
) -> RepresentationDefinition:
    """Return a fresh aligned concatenation such as HOG plus a frozen V3 bias."""
    left, right = registry[anchor], registry[added]
    name = f"{anchor}_plus_{added}"
    return RepresentationDefinition(
        name=name,
        dimension=left.dimension + right.dimension,
        version=f"concat:{left.version}:{right.version}",
        role="complementarity",
        factory=lambda: ConcatenatedRepresentation(
            name, (left.create(), right.create()), f"concat:{left.version}:{right.version}"
        ),
    )


def _candidate_hash(value: Mapping[str, object]) -> str:
    spec = ProgramBiasSpec.from_dict(value)
    return program_bias_hash(spec)
