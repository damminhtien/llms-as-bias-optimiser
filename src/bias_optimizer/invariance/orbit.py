"""Deterministic orbit-mean pooling for a frozen representation and transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bias_optimizer.dsl.compiler import ProgramPipeline
from bias_optimizer.features.base import FeatureVector, Image
from bias_optimizer.invariance.transform import TransformProgram


@dataclass(frozen=True, slots=True)
class OrbitPooledPipeline:
    """Average a representation over identity and seeded transformed views."""

    representation: ProgramPipeline
    transform_program: TransformProgram
    seeds: tuple[int, ...] = (11, 23, 47)

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-empty non-negative integers")
        object.__setattr__(self, "seeds", seeds)

    @property
    def feature_dim(self) -> int:
        return self.representation.feature_dim

    def transform_image(self, image: Image) -> FeatureVector:
        views = [self.representation.transform(image)]
        views.extend(
            self.representation.transform(
                self.transform_program.apply(image, seed=seed)
            )
            for seed in self.seeds
        )
        return np.mean(np.stack(views), axis=0, dtype=np.float64).astype(np.float32)

    def transform(self, image: Image) -> FeatureVector:
        return self.transform_image(image)
