"""Evaluate compact 16- and 30-dimensional zoning controls on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.ml.evaluator import ProgramEvaluator


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def zoning_bias(rows: int = 4, columns: int = 4) -> ProgramBiasSpec:
    if type(rows) is not int or type(columns) is not int or rows <= 0 or columns <= 0:
        raise ValueError("zoning dimensions must be positive integers")
    if rows * columns > 128:
        raise ValueError("zoning feature dimension cannot exceed 128")
    name = f"zoning_{rows}x{columns}"
    dimension = rows * columns
    return ProgramBiasSpec(
        name=name,
        hypothesis="Coarse local ink occupancy is a compact classical shape descriptor.",
        mechanism=f"{dimension} normalized image zones preserve coarse spatial layout.",
        program=_node("spatial_split", _node("image"), rows=rows, cols=columns),
        prediction=f"The {dimension}-dimensional zoning baseline provides a compact shape control.",
        falsification=f"Reject the control if its dimension is not {dimension} or validation fails.",
    )


def evaluate_compact_baseline(output_path: Path) -> dict[str, object]:
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    evaluator = ProgramEvaluator(compiler=compiler, train_sizes=(500, 5_000))
    baselines = []
    for rows, columns in ((4, 4), (5, 6)):
        bias = zoning_bias(rows, columns)
        pipeline = compiler.compile(bias.program)
        expected_dimension = rows * columns
        if pipeline.feature_dim != expected_dimension:
            raise ValueError(f"{bias.name} baseline has an unexpected feature width")
        result = evaluator.evaluate(bias)
        if result.accuracy_5000 is None:
            raise ValueError("compact baseline evaluation requires both train sizes")
        baselines.append(
            {
                "name": bias.name,
                "bias": bias.to_dict(),
                "feature_dim": result.feature_dim,
                "evaluated_train_sizes": [500, 5_000],
                "mnist_validation_accuracy_500": result.accuracy_500,
                "mnist_validation_accuracy_5000": result.accuracy_5000,
                "ranking_score": result.ranking_score,
                "feature_runtime_ms": result.feature_runtime_ms,
                "training_runtime_ms": result.training_runtime_ms,
                "inference_runtime_ms": result.inference_runtime_ms,
                "test_set_accessed": False,
            }
        )
    report: dict[str, object] = {
        "baselines": baselines,
        "test_set_accessed": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/v3_compact_baseline.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_compact_baseline(args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
