"""Tests for the allow-listed BiasSpec to FeaturePipeline compiler."""

import numpy as np
import pytest

from bias_optimizer.compiler import BiasCompilationError, BiasCompiler
from bias_optimizer.domain.bias import BiasSpec, OperatorSpec
from bias_optimizer.features.pipeline import FeaturePipeline


def _bias(operators: tuple[OperatorSpec, ...]) -> BiasSpec:
    return BiasSpec(
        name="compiled_candidate",
        hypothesis="Approved structural features may complement one another.",
        operators=operators,
        prediction="Should improve validation accuracy.",
        falsification="Reject if validation accuracy does not improve.",
    )


def test_compiler_builds_pipeline_from_registered_operator_specs() -> None:
    spec = _bias((OperatorSpec("topology"), OperatorSpec("spatial")))

    pipeline = BiasCompiler().compile(spec)
    features = pipeline.transform(np.zeros((28, 28), dtype=np.float32))

    assert isinstance(pipeline, FeaturePipeline)
    assert pipeline.feature_dim == 7
    assert features.shape == (7,)
    assert np.isfinite(features).all()


def test_compiler_preserves_allowed_operator_parameters() -> None:
    spec = _bias((OperatorSpec("topology", {"threshold": 0.6}),))

    pipeline = BiasCompiler().compile(spec)

    assert pipeline.operators[0].threshold == 0.6  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("operator", "error"),
    [
        (OperatorSpec("unregistered_operator"), "unknown operator"),
        (OperatorSpec("topology", {"threshold": 1.0}), "strictly between 0 and 1"),
        (OperatorSpec("symmetry", {"threshold": 0.5}), "does not accept parameters"),
    ],
)
def test_compiler_reports_invalid_specifications_cleanly(
    operator: OperatorSpec, error: str
) -> None:
    spec = _bias((operator,))

    with pytest.raises(BiasCompilationError, match=error):
        BiasCompiler().compile(spec)


def test_compiler_requires_a_bias_spec() -> None:
    with pytest.raises(TypeError, match="BiasSpec"):
        BiasCompiler().compile("raw_pixels")  # type: ignore[arg-type]
