"""Benchmark bounded cross-candidate subtree reuse on archived programs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.data.mnist import load_mnist_search_data
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.features.pipeline import BatchFeatureExtractor

_EXPENSIVE_OPS = frozenset(
    {"skeletonize", "graph", "paths", "edge_lengths", "angles", "delta", "delta_angle"}
)


def _expensive_subtrees(program: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    stack = [program]
    while stack:
        node = stack.pop()
        if node.op in _EXPENSIVE_OPS:
            result[node.to_json()] = node.op
        stack.extend(node.args)
    return result


def profile_cache(
    archive_path: Path,
    output_path: Path,
    *,
    image_count: int = 96,
    candidate_count: int = 16,
    max_cache_mb: int = 256,
) -> dict[str, Any]:
    records = [
        ProgramSearchRecord.from_json(line)
        for line in Path(archive_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = {
        record.candidate_id: record
        for record in records
        if record.track == SearchTrack.DISCOVERY.value
    }
    subtrees = {
        candidate_id: _expensive_subtrees(record.bias.program)
        for candidate_id, record in candidates.items()
    }
    selected: list[ProgramSearchRecord] = []
    available = set(candidates)
    seen_subtrees: set[str] = set()
    while available and len(selected) < candidate_count:
        next_id = max(
            available,
            key=lambda candidate_id: (
                len(set(subtrees[candidate_id]) & seen_subtrees),
                candidates[candidate_id].evaluation.ranking_score,
                candidate_id,
            ),
        )
        available.remove(next_id)
        selected.append(candidates[next_id])
        seen_subtrees.update(subtrees[next_id])
    if len(selected) < 2:
        raise ValueError("profile needs at least two discovery programs")
    data = load_mnist_search_data()
    images, _ = data.sample_training_data(image_count, seed=42)
    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    pipelines = tuple(compiler.compile(item.bias.program) for item in selected)
    baseline = BatchFeatureExtractor(cache=None, subexpression_cache=None)
    baseline_start = perf_counter()
    baseline_matrices = [
        baseline.transform(
            pipeline,
            images,
            bias_hash=item.candidate_id,
            dataset_key="cache-profile/sample-seed-42/count-96",
        )
        for index, (item, pipeline) in enumerate(zip(selected, pipelines, strict=True))
    ]
    baseline_seconds = perf_counter() - baseline_start

    shared_cache = SubexpressionCache(max_cache_mb * 1024 * 1024)
    cached = BatchFeatureExtractor(cache=None, subexpression_cache=shared_cache)
    cache_start = perf_counter()
    cached_matrices = [
        cached.transform(
            pipeline,
            images,
            bias_hash=item.candidate_id,
            dataset_key="cache-profile/sample-seed-42/count-96",
        )
        for index, (item, pipeline) in enumerate(zip(selected, pipelines, strict=True))
    ]
    cache_seconds = perf_counter() - cache_start
    equivalent = all(
        np.array_equal(left, right)
        for left, right in zip(baseline_matrices, cached_matrices, strict=True)
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "archive": str(archive_path),
        "candidate_count": len(selected),
        "candidate_ids": [item.candidate_id for item in selected],
        "image_count": len(images),
        "baseline_seconds": baseline_seconds,
        "cached_seconds": cache_seconds,
        "speedup": baseline_seconds / cache_seconds if cache_seconds else None,
        "feature_matrices_exactly_equal": equivalent,
        "cache_metrics": shared_cache.metrics,
        "claim_boundary": "Timings cover this local sample and candidate order; they are a cache smoke profile, not a general speed guarantee.",
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
        "--archive", type=Path, default=Path("results/program_search_discovery.jsonl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/subexpression_cache_profile.json"),
    )
    parser.add_argument("--images", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--cache-mb", type=int, default=256)
    args = parser.parse_args()
    report = profile_cache(
        args.archive,
        args.output,
        image_count=args.images,
        candidate_count=args.candidates,
        max_cache_mb=args.cache_mb,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
