"""Re-evaluate the top V3 stage-one programs with 500 and 5,000 samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.ml.evaluator import ProgramEvaluator


class _Evaluator(Protocol):
    def evaluate(self, bias) -> Evaluation: ...


def _archive_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_finalists(
    archive_path: Path,
    output_path: Path,
    *,
    count: int = 20,
    evaluator: _Evaluator | None = None,
) -> dict[str, object]:
    """Select by 500-sample validation fitness and evaluate only those at 5k."""
    archive_path = Path(archive_path)
    output_path = Path(output_path)
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    records = tuple(
        ProgramSearchRecord.from_json(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not records:
        raise ValueError("search archive is empty")
    if any(record.track != SearchTrack.DISCOVERY.value for record in records):
        raise ValueError("search archive contains a non-discovery record")

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    unique: dict[str, ProgramSearchRecord] = {}
    for record in records:
        compiler.compile(record.bias.program)
        unique.setdefault(record.bias.program.to_json(), record)
    candidates = sorted(
        unique.values(),
        key=lambda item: (
            -item.evaluation.ranking_score,
            -item.evaluation.accuracy_500,
            item.candidate_id,
        ),
    )[:count]
    if len(candidates) < count:
        raise ValueError(f"need {count} unique candidates; found {len(candidates)}")

    program_evaluator = evaluator or ProgramEvaluator(
        compiler=compiler,
        train_sizes=(500, 5_000),
    )
    evaluated = []
    for rank, record in enumerate(candidates, start=1):
        result = program_evaluator.evaluate(record.bias)
        if result.accuracy_5000 is None:
            raise ValueError("finalist evaluator did not evaluate 5,000 samples")
        evaluated.append(
            {
                "rank": rank,
                "candidate_id": record.candidate_id,
                "bias": record.bias.to_dict(),
                "descriptor": record.descriptor,
                "novelty": record.novelty,
                "stage1": {
                    "accuracy_500": record.evaluation.accuracy_500,
                    "ranking_score": record.evaluation.ranking_score,
                    "feature_dim": record.evaluation.feature_dim,
                    "runtime_ms": record.evaluation.runtime_ms,
                },
                "full_validation": {
                    "accuracy_500": result.accuracy_500,
                    "accuracy_5000": result.accuracy_5000,
                    "ranking_score": result.ranking_score,
                    "feature_dim": result.feature_dim,
                    "feature_runtime_ms": result.feature_runtime_ms,
                    "training_runtime_ms": result.training_runtime_ms,
                    "inference_runtime_ms": result.inference_runtime_ms,
                    "confusion_matrix": result.confusion_matrix.tolist(),
                },
            }
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "search_archive": str(archive_path),
        "search_archive_sha256": _archive_sha256(archive_path),
        "selection_rule": "top stage-one ranking score at n=500, then accuracy_500, then candidate_id",
        "candidate_count": len(unique),
        "evaluated_train_sizes": [500, 5_000],
        "finalist_count": len(evaluated),
        "evaluations": evaluated,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("results/v3_program_search_discovery.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v3_finalist_evaluations.json"),
    )
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    summary = evaluate_finalists(args.archive, args.output, count=args.count)
    print(
        json.dumps(
            {
                "finalist_count": summary["finalist_count"],
                "archive_sha256": summary["search_archive_sha256"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
