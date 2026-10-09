"""Run bounded V3 program search without accessing MNIST test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.seeds import generate_v3_pilot_seed_programs
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.llm.program_proposer import ProgramProposer
from bias_optimizer.llm.response_archive import JsonlResponseArchive
from bias_optimizer.ml.evaluator import ProgramEvaluator
from bias_optimizer.novelty import DESCRIPTOR_TARGET_CELLS, MapElitesArchive
from bias_optimizer.search.program_engine import ProgramSearchEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=tuple(track.value for track in SearchTrack),
        default=SearchTrack.DISCOVERY.value,
    )
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--candidates-per-generation", type=int, default=18)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--v3-pilot-seeds",
        action="store_true",
        help="use the ten explicit, mechanism-balanced V3 pilot programs",
    )
    parser.add_argument(
        "--proposal-rounds",
        type=int,
        default=36,
        help="maximum LLM refill batches per generation",
    )
    parser.add_argument("--max-feature-dim", type=int)
    parser.add_argument(
        "--stage1-500-only",
        action="store_true",
        help="screen discovery candidates using 500 training examples only",
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--llm-responses", type=Path)
    parser.add_argument("--failures", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    track = SearchTrack(args.track)
    if args.v3_pilot_seeds and track is not SearchTrack.DISCOVERY:
        raise SystemExit("--v3-pilot-seeds is supported only for discovery")
    if args.stage1_500_only and track is not SearchTrack.DISCOVERY:
        raise SystemExit("--stage1-500-only is supported only for discovery")
    default_dim = 128
    max_feature_dim = (
        args.max_feature_dim if args.max_feature_dim is not None else default_dim
    )
    archive_path = args.archive or Path(
        f"results/v3_program_search_{track.value}.jsonl"
    )
    response_path = args.llm_responses or Path(
        f"results/v3_program_llm_responses_{track.value}.jsonl"
    )
    program_evaluator = ProgramEvaluator(
        compiler=ProgramCompiler(
            ProgramConstraints(
                track=track,
                max_feature_dim=max_feature_dim,
                forbid_raw_pixels=track is SearchTrack.AUGMENTATION,
                require_vector_root=track is SearchTrack.AUGMENTATION,
            )
        ),
        raw_pixel_anchor=track is SearchTrack.AUGMENTATION,
        train_sizes=(500,) if args.stage1_500_only else (500, 5_000),
    )
    engine = ProgramSearchEngine(
        track=track,
        max_feature_dim=max_feature_dim,
        archive=MapElitesArchive(archive_path),
        failure_path=args.failures,
        proposer=ProgramProposer(response_archive=JsonlResponseArchive(response_path)),
        evaluator=program_evaluator,
        seed_count=args.seed_count,
        seed=args.seed,
        seed_programs=(
            generate_v3_pilot_seed_programs() if args.v3_pilot_seeds else None
        ),
        max_proposal_rounds=args.proposal_rounds,
        require_complete=True,
    )
    archive = engine.run(
        generations=args.generations,
        candidates_per_generation=args.candidates_per_generation,
    )
    print(
        json.dumps(
            {
                "track": track.value,
                "evaluated_candidates": len(archive.records),
                "occupied_cells": [list(cell) for cell in archive.occupied_cells],
                "underexplored_cells": [
                    list(cell)
                    for cell in archive.underexplored_cells(
                        tuple(
                            cell
                            for cell in DESCRIPTOR_TARGET_CELLS
                            if track is SearchTrack.AUGMENTATION or cell[0] != "pixel"
                        )
                    )
                ],
                "archive": str(archive_path),
                "failure_count": engine.failure_count,
                "failure_log": str(
                    args.failures
                    if args.failures is not None
                    else archive_path.with_name(f"{archive_path.stem}_failures.jsonl")
                ),
                "subexpression_cache": program_evaluator.subexpression_cache_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
