"""Feature operator and pipeline contracts."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.float32]
FeatureVector = NDArray[np.float32]


class FeatureOperator(Protocol):
    """Transform one normalized image into a finite feature vector."""

    @property
    def feature_dim(self) -> int:
        """Return the fixed output dimension for one image."""

    def transform(self, image: Image) -> FeatureVector:
        """Return features for a single image."""


class CompiledRepresentation(Protocol):
    """One compiled representation with a fixed dimension and image transform."""

    @property
    def feature_dim(self) -> int: ...

    def transform(self, image: Image) -> FeatureVector: ...
