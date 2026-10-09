"""Measure behavioral novelty on a fixed MNIST validation probe set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bias_optimizer.cache.subexpression_cache import SubexpressionCache
from bias_optimizer.data.mnist import MNISTDataConfig, load_mnist_search_data
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.novelty.behavior import (
    behavioral_novelty,
    pairwise_distance_profile,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_behavioral_novelty(
    archive_path: Path,
    output_path: Path,
    *,
    data_dir: Path = Path("data"),
    probe_count: int = 128,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    records = tuple(
        ProgramSearchRecord.from_json(line)
        for line in archive_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    unique_records = {record.candidate_id: record for record in records}
    if not unique_records:
        raise ValueError("search archive is empty")
    data = load_mnist_search_data(MNISTDataConfig(data_dir=Path(data_dir)))
    if type(probe_count) is not int or probe_count < 2:
        raise ValueError("probe_count must be an integer of at least two")
    if probe_count > len(data.validation_images):
        raise ValueError("probe_count exceeds the available validation probes")
    probes = data.validation_images[:probe_count]

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    subexpression_cache = SubexpressionCache()
    profiles: dict[str, np.ndarray] = {}
    for candidate_id, record in unique_records.items():
        pipeline = compiler.compile(record.bias.program)
        features = np.stack(
            [
                pipeline.transform_with_cache(
                    image,
                    cache=subexpression_cache,
                    sample_key=f"v3_behavior_probe:{index}",
                )
                for index, image in enumerate(probes)
            ]
        )
        profiles[candidate_id] = pairwise_distance_profile(features)

    results = []
    for candidate_id, record in unique_records.items():
        novelty, nearest_id, similarity = behavioral_novelty(
            profiles[candidate_id], profiles, candidate_id=candidate_id
        )
        nearest_name = (
            unique_records[nearest_id].bias.name if nearest_id is not None else None
        )
        results.append(
            {
                "candidate_id": candidate_id,
                "name": record.bias.name,
                "structural_novelty": record.novelty,
                "behavioral_novelty": novelty,
                "nearest_behavior_candidate_id": nearest_id,
                "nearest_behavior_name": nearest_name,
                "nearest_behavior_similarity": similarity,
            }
        )
    report = {
        "schema_version": 1,
        "search_archive": str(archive_path),
        "search_archive_sha256": _sha256(archive_path),
        "probe_split": "MNIST validation partition; labels are not used",
        "probe_count": probe_count,
        "distance_metric": "Pearson correlation of pairwise Euclidean distances between probe representations",
        "candidate_count": len(results),
        "subexpression_cache": subexpression_cache.metrics,
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
        "--archive",
        type=Path,
        default=Path("results/v3_program_search_discovery.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/v3_behavioral_novelty.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--probe-count", type=int, default=128)
    args = parser.parse_args()
    summary = evaluate_behavioral_novelty(
        args.archive,
        args.output,
        data_dir=args.data_dir,
        probe_count=args.probe_count,
    )
    print(
        json.dumps(
            {
                "candidate_count": summary["candidate_count"],
                "probe_count": summary["probe_count"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
