"""Evaluate frozen V3 finalists with mechanism-level representation ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bias_optimizer.data.mnist import MNISTDataConfig
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.ml.evaluator import Evaluator, ProgramEvaluator
from bias_optimizer.ml.learner import LearnerConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ablate_program(program: Expr, mechanism: str) -> tuple[Expr, int]:
    """Remove one named mechanism while retaining the surrounding program."""
    if mechanism not in {"no_spatial", "no_order", "no_curvature", "no_relation"}:
        raise ValueError(f"unknown ablation {mechanism!r}")
    removed = 0

    def rewrite(node: Expr) -> Expr:
        nonlocal removed
        args = tuple(rewrite(child) for child in node.args)
        params = node.parameter_values
        if mechanism == "no_spatial" and node.op == "spatial_condition":
            removed += 1
            return Expr(
                "histogram",
                (args[0],),
                {
                    "bins": params.get("bins", 8),
                    "low": params.get("low", -3.141593),
                    "high": params.get("high", 3.141593),
                },
            )
        if mechanism == "no_order":
            if node.op == "autocorrelation":
                removed += 1
                moment_count = max(1, min(8, len(params.get("lags", (1, 2, 4)))))
                return Expr(
                    "moments", (args[0],), {"orders": list(range(1, moment_count + 1))}
                )
            if node.op in {"run_length_encode", "delta", "delta_angle"}:
                removed += 1
                return args[0]
        if mechanism == "no_curvature" and node.op == "delta_angle":
            removed += 1
            return args[0]
        if mechanism == "no_relation":
            if node.op == "pairwise_difference":
                removed += 1
                return Expr("concat", args)
            if node.op == "cross_histogram":
                removed += 1
                return Expr(
                    "concat",
                    (
                        Expr(
                            "histogram",
                            (args[0],),
                            {
                                "bins": params.get("bins_x", 6),
                                "low": params.get("low_x", 0.0),
                                "high": params.get("high_x", 1.0),
                            },
                        ),
                        Expr(
                            "histogram",
                            (args[1],),
                            {
                                "bins": params.get("bins_y", 6),
                                "low": params.get("low_y", 0.0),
                                "high": params.get("high_y", 1.0),
                            },
                        ),
                    ),
                )
            if node.op == "spatial_condition":
                removed += 1
                return Expr(
                    "histogram",
                    (args[0],),
                    {
                        "bins": params.get("bins", 8),
                        "low": params.get("low", -3.141593),
                        "high": params.get("high", 3.141593),
                    },
                )
        return Expr(node.op, args, params)

    result = rewrite(program)
    if not removed:
        raise ValueError(f"program has no {mechanism.removeprefix('no_')} mechanism")
    return result, removed


def evaluate_v3_ablations(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    finalist_count: int = 5,
    mechanisms: tuple[str, ...] = (
        "no_spatial",
        "no_order",
        "no_curvature",
        "no_relation",
    ),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("ablation evaluation requires frozen finalists")
    archive = Path(manifest["search_archive"])
    if not archive.is_absolute() and not archive.exists():
        archive = manifest_path.parent.parent / archive
    archive_hash = _sha256(archive)
    if archive_hash != manifest.get("search_archive_sha256"):
        raise ValueError("search archive changed after finalists were frozen")

    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.DISCOVERY,
            max_feature_dim=128,
            max_depth=8,
        )
    )
    evaluator = ProgramEvaluator(
        evaluator=Evaluator(
            learner_config=LearnerConfig(seed=42),
            data_config=MNISTDataConfig(data_dir=Path(data_dir)),
        ),
        compiler=compiler,
        train_sizes=(500, 5_000),
    )
    results = []
    for finalist in manifest["finalists"][:finalist_count]:
        parent = ProgramBiasSpec.from_dict(finalist["bias"])
        compiler.compile(parent.program)
        parent_evaluation = finalist["full_validation"]
        ablations = []
        for mechanism in mechanisms:
            try:
                program, node_count = ablate_program(parent.program, mechanism)
                pipeline = compiler.compile(program)
            except ValueError as exc:
                if "has no" in str(exc):
                    continue
                ablations.append(
                    {"mechanism": mechanism, "status": "invalid", "reason": str(exc)}
                )
                continue
            ablated = ProgramBiasSpec(
                name=f"{parent.name[:100]}_{mechanism}",
                hypothesis=parent.hypothesis,
                mechanism=f"Mechanism ablation: {mechanism}.",
                program=program,
                prediction=parent.prediction,
                falsification=parent.falsification,
            )
            evaluation = evaluator.evaluate(ablated)
            if evaluation.accuracy_5000 is None:
                raise ValueError("ablation evaluation omitted the 5,000-sample result")
            ablations.append(
                {
                    "mechanism": mechanism,
                    "status": "evaluated",
                    "removed_node_count": node_count,
                    "feature_dim": pipeline.feature_dim,
                    "accuracy_500": evaluation.accuracy_500,
                    "accuracy_5000": evaluation.accuracy_5000,
                    "delta_accuracy_500": (
                        evaluation.accuracy_500 - parent_evaluation["accuracy_500"]
                    ),
                    "delta_accuracy_5000": (
                        evaluation.accuracy_5000 - parent_evaluation["accuracy_5000"]
                    ),
                    "program": program.to_dict(),
                }
            )
        results.append(
            {
                "candidate_id": finalist["candidate_id"],
                "name": parent.name,
                "mechanism": parent.mechanism,
                "parent_validation": parent_evaluation,
                "ablations": ablations,
            }
        )

    report = {
        "schema_version": 1,
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": _sha256(manifest_path),
        "search_archive_sha256": archive_hash,
        "mnist_test_accessed": False,
        "train_sizes": [500, 5_000],
        "train_seed": 42,
        "results": results,
        "claim_boundary": "Validation ablations isolate the named DSL mechanism by rewriting only its corresponding AST operators; dimension changes are reported and results are not test-set estimates.",
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
        "--manifest", type=Path, default=Path("results/v3_frozen_finalists.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v3_ablation_report.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--finalist-count", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_v3_ablations(
        args.manifest,
        args.output,
        data_dir=args.data_dir,
        finalist_count=args.finalist_count,
    )
    print(
        json.dumps(
            {"results": len(report["results"]), "output": str(args.output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
