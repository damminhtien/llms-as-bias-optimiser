"""Compact zoning controls are sized to match discovered feature programs."""

import pytest

from bias_optimizer.dsl.compiler import ProgramCompiler
from experiments.evaluate_v3_compact_baseline import zoning_bias


@pytest.mark.parametrize(
    ("rows", "columns", "expected_dim"),
    ((4, 4, 16), (5, 6, 30)),
)
def test_zoning_bias_has_declared_feature_dimension(
    rows: int,
    columns: int,
    expected_dim: int,
) -> None:
    bias = zoning_bias(rows, columns)

    assert bias.name == f"zoning_{rows}x{columns}"
    assert ProgramCompiler().compile(bias.program).feature_dim == expected_dim


def test_zoning_bias_rejects_feature_width_above_search_limit() -> None:
    with pytest.raises(ValueError, match="cannot exceed 128"):
        zoning_bias(12, 12)
