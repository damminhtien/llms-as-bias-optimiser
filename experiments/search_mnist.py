"""Run a measured, validation-only Qwen-guided MNIST search."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

from bias_optimizer.domain.bias import BiasSpec
from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.search.archive import SearchArchive
from bias_optimizer.search.engine import SearchEngine


class _TimedEvaluator:
    """Accumulate CPU and elapsed time spent evaluating representations."""

    def __init__(self, evaluator: Evaluator) -> None:
        self._evaluator = evaluator
        self.cpu_seconds = 0.0
        self.wall_seconds = 0.0
        self.calls = 0

    def evaluate(self, bias: BiasSpec) -> Evaluation:
        cpu_start = process_time()
        wall_start = perf_counter()
        result = self._evaluator.evaluate(bias)
        self.cpu_seconds += process_time() - cpu_start
        self.wall_seconds += perf_counter() - wall_start
        self.calls += 1
        return result


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _read_appended_jsonl(path: Path, start_line: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        lines = [line for line in source if line.strip()]
    return [json.loads(line) for line in lines[start_line:]]


def run_search_experiment(
    *,
    generations: int = 10,
    candidates_per_generation: int = 8,
    archive_path: Path = Path("results/search.jsonl"),
    response_archive_path: Path = Path("results/llm_responses.jsonl"),
    output_path: Path = Path("results/search_metadata.json"),
) -> dict[str, Any]:
    """Run or resume a search and save candidate, token, CPU, and wall metrics."""
    started_at = datetime.now(UTC)
    wall_start = perf_counter()
    response_start_line = _line_count(response_archive_path)
    archive = SearchArchive(archive_path)
    prior_ids = {record.candidate_id for record in archive.records}
    timed_evaluator = _TimedEvaluator(Evaluator())

    completed = SearchEngine(
        evaluator=timed_evaluator,
        archive=archive,
    ).run(
        generations=generations,
        candidates_per_generation=candidates_per_generation,
    )

    appended_responses = _read_appended_jsonl(
        response_archive_path,
        response_start_line,
    )
    measured_usage = [
        item
        for item in appended_responses
        if item.get("prompt_tokens") is not None
        and item.get("response_tokens") is not None
    ]
    new_records = [
        record for record in completed.records if record.candidate_id not in prior_ids
    ]
    new_candidates = [
        record for record in new_records if record.record_type == "candidate"
    ]
    new_ablations = [
        record for record in new_records if record.record_type == "ablation"
    ]
    generation_counts = Counter(
        record.generation
        for record in completed.records
        if record.record_type == "candidate"
    )
    ended_at = datetime.now(UTC)

    metadata: dict[str, Any] = {
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "generations_requested": generations,
        "candidates_per_generation_requested": candidates_per_generation,
        "search_data_only": True,
        "archive_path": str(archive_path),
        "response_archive_path": str(response_archive_path),
        "candidate_records_total": sum(
            record.record_type == "candidate" for record in completed.records
        ),
        "candidate_records_added": len(new_candidates),
        "ablation_records_added": len(new_ablations),
        "candidate_records_by_generation": {
            str(generation): generation_counts[generation]
            for generation in sorted(generation_counts)
        },
        "llm_attempts": len(appended_responses),
        "llm_accepted_attempts": sum(
            bool(item.get("accepted")) for item in appended_responses
        ),
        "llm_token_usage_records": len(measured_usage),
        "llm_token_usage_complete": len(measured_usage) == len(appended_responses),
        "llm_prompt_tokens": sum(item["prompt_tokens"] for item in measured_usage),
        "llm_response_tokens": sum(item["response_tokens"] for item in measured_usage),
        "llm_total_tokens": sum(
            item["prompt_tokens"] + item["response_tokens"] for item in measured_usage
        ),
        "llm_models": sorted(
            {item["model"] for item in appended_responses if item.get("model")}
        ),
        "evaluation_calls_added": timed_evaluator.calls,
        "evaluation_cpu_seconds": timed_evaluator.cpu_seconds,
        "evaluation_wall_seconds": timed_evaluator.wall_seconds,
        "search_wall_seconds": perf_counter() - wall_start,
        "best_candidates": [
            {
                "candidate_id": record.candidate_id,
                "name": record.bias.name,
                "accuracy_500": record.evaluation.accuracy_500,
                "accuracy_5000": record.evaluation.accuracy_5000,
                "feature_dim": record.evaluation.feature_dim,
                "ranking_score": record.evaluation.ranking_score,
            }
            for record in completed.top(5)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--candidates-per-generation", type=int, default=8)
    parser.add_argument("--archive", type=Path, default=Path("results/search.jsonl"))
    parser.add_argument(
        "--response-archive",
        type=Path,
        default=Path("results/llm_responses.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/search_metadata.json"),
    )
    args = parser.parse_args()
    metadata = run_search_experiment(
        generations=args.generations,
        candidates_per_generation=args.candidates_per_generation,
        archive_path=args.archive,
        response_archive_path=args.response_archive,
        output_path=args.output,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
