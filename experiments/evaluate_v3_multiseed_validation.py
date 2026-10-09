"""Evaluate frozen V3 finalists on MNIST validation across fixed seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from bias_optimizer.data.mnist import MNISTDataConfig
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.ml.evaluator import Evaluator, ProgramEvaluator
from bias_optimizer.ml.learner import LearnerConfig


class _ProgramEvaluator(Protocol):
    def evaluate(self, bias: ProgramBiasSpec) -> Evaluation: ...


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_multiseed_validation(
    manifest_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    seeds: tuple[int, ...] = (11, 23, 47),
    finalist_count: int = 5,
    evaluator_factory: Callable[[int], _ProgramEvaluator] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("multi-seed evaluation requires frozen finalists")
    archive = Path(manifest["search_archive"])
    if not archive.is_absolute() and not archive.exists():
        archive = manifest_path.parent.parent / archive
    archive_hash = _sha256(archive)
    if archive_hash != manifest.get("search_archive_sha256"):
        raise ValueError("search archive changed after finalists were frozen")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be a non-empty tuple of non-negative integers")

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )

    def create_evaluator(seed: int) -> _ProgramEvaluator:
        if evaluator_factory is not None:
            return evaluator_factory(seed)
        return ProgramEvaluator(
            evaluator=Evaluator(
                learner_config=LearnerConfig(seed=seed),
                data_config=MNISTDataConfig(data_dir=Path(data_dir)),
            ),
            compiler=compiler,
            train_sizes=(500, 5_000),
        )

    results = []
    for finalist in manifest["finalists"][:finalist_count]:
        bias = ProgramBiasSpec.from_dict(finalist["bias"])
        if program_bias_hash(bias) != finalist.get("candidate_id"):
            raise ValueError("finalist hash does not match its frozen AST")
        compiler.compile(bias.program)
        evaluations = []
        for seed in seeds:
            evaluation = create_evaluator(seed).evaluate(bias)
            if evaluation.accuracy_5000 is None:
                raise ValueError("multi-seed evaluation requires both train sizes")
            evaluations.append(
                {
                    "seed": seed,
                    "accuracy_500": evaluation.accuracy_500,
                    "accuracy_5000": evaluation.accuracy_5000,
                    "feature_dim": evaluation.feature_dim,
                    "feature_runtime_ms": evaluation.feature_runtime_ms,
                }
            )
        results.append(
            {
                "candidate_id": finalist["candidate_id"],
                "name": bias.name,
                "evaluations": evaluations,
                "summary": {
                    "accuracy_500_mean": float(
                        np.mean([item["accuracy_500"] for item in evaluations])
                    ),
                    "accuracy_500_std": float(
                        np.std([item["accuracy_500"] for item in evaluations])
                    ),
                    "accuracy_5000_mean": float(
                        np.mean([item["accuracy_5000"] for item in evaluations])
                    ),
                    "accuracy_5000_std": float(
                        np.std([item["accuracy_5000"] for item in evaluations])
                    ),
                },
            }
        )

    report = {
        "schema_version": 1,
        "selection_manifest": str(manifest_path),
        "selection_manifest_sha256": _sha256(manifest_path),
        "search_archive_sha256": archive_hash,
        "mnist_test_accessed": False,
        "train_sizes": [500, 5_000],
        "seeds": list(seeds),
        "results": results,
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
        "--output", type=Path, default=Path("results/v3_multiseed_validation.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    report = evaluate_multiseed_validation(
        args.manifest, args.output, data_dir=args.data_dir
    )
    print(
        json.dumps(
            {"finalist_count": len(report["results"]), "output": str(args.output)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
