"""Typed, allow-listed image transformations used by invariance experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from bias_optimizer.features._image import validated_image

_STAGE_PARAMS = {
    "translate": frozenset({"dy", "dx"}),
    "rotate": frozenset({"degrees"}),
    "dilate": frozenset({"radius"}),
    "erode": frozenset({"radius"}),
    "shear": frozenset({"amount"}),
    "elastic": frozenset({"amplitude", "sigma"}),
}


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise TypeError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class TransformStep:
    """One Image → Image stage with a validated, immutable parameter map."""

    op: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in _STAGE_PARAMS:
            raise ValueError(f"unknown image-transform primitive {self.op!r}")
        if not isinstance(self.params, Mapping):
            raise TypeError("transform params must be a mapping")
        values = dict(self.params)
        if set(values) != _STAGE_PARAMS[self.op]:
            raise ValueError(
                f"{self.op} requires exactly {sorted(_STAGE_PARAMS[self.op])}"
            )
        if self.op == "translate":
            for name in ("dy", "dx"):
                value = values[name]
                if type(value) is not int or not -2 <= value <= 2:
                    raise ValueError("translation offsets must be integers in [-2, 2]")
            if values["dy"] == 0 and values["dx"] == 0:
                raise ValueError("translation must change the image")
        elif self.op == "rotate":
            angle = _finite(values["degrees"], "degrees")
            if angle == 0 or not -15 <= angle <= 15:
                raise ValueError("rotation degrees must be non-zero and in [-15, 15]")
            values["degrees"] = angle
        elif self.op in {"dilate", "erode"}:
            radius = values["radius"]
            if type(radius) is not int or radius != 1:
                raise ValueError("stroke-width radius must be exactly 1 pixel")
        elif self.op == "shear":
            amount = _finite(values["amount"], "amount")
            if amount == 0 or not -0.2 <= amount <= 0.2:
                raise ValueError("shear amount must be non-zero and in [-0.2, 0.2]")
            values["amount"] = amount
        elif self.op == "elastic":
            amplitude = _finite(values["amplitude"], "amplitude")
            sigma = _finite(values["sigma"], "sigma")
            if not 0 < amplitude <= 2:
                raise ValueError("elastic amplitude must be in (0, 2] pixels")
            if not 1 <= sigma <= 8:
                raise ValueError("elastic sigma must be in [1, 8] pixels")
            values.update(amplitude=amplitude, sigma=sigma)
        object.__setattr__(self, "params", MappingProxyType(values))

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class TransformProgram:
    """A bounded composition of typed Image → Image transformation atoms."""

    steps: tuple[TransformStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= 4:
            raise ValueError("transform programs must contain between 1 and 4 steps")
        if not all(isinstance(step, TransformStep) for step in steps):
            raise TypeError("steps must contain TransformStep values")
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    def to_json(self) -> str:
        import json

        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransformProgram:
        if not isinstance(value, Mapping) or set(value) != {"steps"}:
            raise ValueError("transform program must contain exactly the steps field")
        steps = value["steps"]
        if not isinstance(steps, list):
            raise TypeError("transform program steps must be a list")
        parsed: list[TransformStep] = []
        for item in steps:
            if not isinstance(item, Mapping) or set(item) != {"op", "params"}:
                raise ValueError("each transform step requires exactly op and params")
            parsed.append(TransformStep(op=item["op"], params=item["params"]))
        return cls(tuple(parsed))

    def apply(self, image: NDArray, *, seed: int = 0) -> NDArray[np.float32]:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        result = validated_image(image).copy()
        for index, step in enumerate(self.steps):
            result = _apply_step(result, step, seed=seed + index)
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class InvarianceSpec:
    """A falsifiable label-preserving transformation hypothesis."""

    name: str
    hypothesis: str
    mechanism: str
    transform: TransformProgram
    prediction: str
    falsification: str

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "hypothesis",
            "mechanism",
            "prediction",
            "falsification",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if len(value) > 2_000:
                raise ValueError(f"{field_name} is too long")
        if not isinstance(self.transform, TransformProgram):
            raise TypeError("transform must be a TransformProgram")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "transform": self.transform.to_dict(),
            "prediction": self.prediction,
            "falsification": self.falsification,
        }


def _apply_step(
    image: NDArray[np.float32], step: TransformStep, *, seed: int
) -> NDArray[np.float32]:
    params = step.params
    if step.op == "translate":
        return ndimage.shift(
            image,
            (params["dy"], params["dx"]),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
    if step.op == "rotate":
        return ndimage.rotate(
            image,
            params["degrees"],
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
    if step.op == "dilate":
        return ndimage.maximum_filter(image, size=3, mode="constant").astype(np.float32)
    if step.op == "erode":
        return ndimage.minimum_filter(image, size=3, mode="constant").astype(np.float32)
    if step.op == "shear":
        amount = params["amount"]
        matrix = np.asarray([[1.0, amount], [0.0, 1.0]], dtype=np.float64)
        center = (np.asarray(image.shape, dtype=np.float64) - 1.0) / 2.0
        offset = center - matrix @ center
        return ndimage.affine_transform(
            image,
            matrix,
            offset=offset,
            output_shape=image.shape,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
    if step.op == "elastic":
        rng = np.random.default_rng(seed)
        amplitude = params["amplitude"]
        sigma = params["sigma"]
        displacement = []
        for _ in range(2):
            field = ndimage.gaussian_filter(
                rng.normal(size=image.shape), sigma=sigma, mode="reflect"
            )
            maximum = float(np.max(np.abs(field)))
            displacement.append(field * (amplitude / maximum if maximum else 0.0))
        rows, columns = np.meshgrid(
            np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij"
        )
        coordinates = np.asarray(
            [rows + displacement[0], columns + displacement[1]], dtype=np.float64
        )
        return ndimage.map_coordinates(
            image,
            coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32)
    raise AssertionError(f"unhandled validated transform {step.op}")
