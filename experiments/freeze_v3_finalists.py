"""Freeze V3 finalists using validation only before any transfer test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(expr: Expr):
    yield expr
    for child in expr.args:
        yield from _walk(child)


def _counterfactual_expectation(program: Expr) -> dict[str, str]:
    operations = {node.op for node in _walk(program)}
    measures = {
        node.params.get("measure")
        for node in _walk(program)
        if node.op == "path_summary"
    }
    if "cross_histogram" in operations:
        if measures.intersection({"centroid_x", "centroid_y"}):
            intervention = "Reassign path centroids among summaries while preserving path counts and graph-summary marginals."
            mechanism = "graph and location interaction"
        else:
            intervention = "Shuffle one input sequence while preserving both input value marginals and event locations."
            mechanism = "cross-variable relation"
    elif "spatial_condition" in operations:
        intervention = "Reassign events across spatial regions while preserving the global event-value histogram."
        mechanism = "spatial conditioning"
    elif operations.intersection({"autocorrelation", "run_length_encode"}):
        intervention = "Shuffle event order within each path while preserving the per-path event-value multiset."
        mechanism = "ordered stroke or turn persistence"
    elif "cycle_rank" in operations or "connected_components" in operations:
        intervention = "Apply a topology-changing skeleton edit while matching total foreground mass."
        mechanism = "topological structure"
    else:
        intervention = "Permute within-path geometry events while preserving their global marginal distribution."
        mechanism = "global path geometry"
    return {
        "mechanism": mechanism,
        "intervention": intervention,
        "expected_effect": "Validation accuracy and prediction agreement should decrease if this mechanism is causal.",
    }


def freeze_finalists(
    archive_path: Path,
    evaluations_path: Path,
    output_path: Path,
    *,
    count: int = 5,
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    evaluations_path = Path(evaluations_path)
    output_path = Path(output_path)
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if not archive_path.exists() or not evaluations_path.exists():
        raise FileNotFoundError("search archive and finalist evaluations are required")

    evaluations = json.loads(evaluations_path.read_text(encoding="utf-8"))
    expected_archive_hash = evaluations.get("search_archive_sha256")
    archive_hash = _sha256(archive_path)
    if expected_archive_hash != archive_hash:
        raise ValueError("search archive changed after finalist re-evaluation")
    candidates = evaluations.get("evaluations")
    if not isinstance(candidates, list):
        raise TypeError("finalist evaluation report has no evaluations list")
    selected = sorted(
        candidates,
        key=lambda item: (
            -item["full_validation"]["ranking_score"],
            -item["full_validation"]["accuracy_500"],
            item["candidate_id"],
        ),
    )[:count]
    if len(selected) < count:
        raise ValueError(f"need {count} full candidates; found {len(selected)}")

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
    )
    finalists = []
    for item in selected:
        bias = ProgramBiasSpec.from_dict(item["bias"])
        compiler.compile(bias.program)
        finalists.append(
            {
                "candidate_id": item["candidate_id"],
                "name": bias.name,
                "hypothesis": bias.hypothesis,
                "mechanism": bias.mechanism,
                "prediction": bias.prediction,
                "falsification": bias.falsification,
                "expected_transfer_behavior": {
                    "emnist_digits": "Positive transfer relative to compact structural controls is predicted.",
                    "emnist_letters": "Positive but weaker transfer is predicted because the domain has more classes.",
                    "kmnist": "Positive transfer is predicted if the learned glyph structure generalizes.",
                    "fashion_mnist": "Negative-control domain; no handwriting-specific advantage is predicted.",
                },
                "expected_counterfactual": _counterfactual_expectation(bias.program),
                "descriptor": item["descriptor"],
                "novelty": item["novelty"],
                "full_validation": item["full_validation"],
                "bias": bias.to_dict(),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "finalist_selection_frozen": True,
        "test_set_accessed": False,
        "selection_rule": "top five by 0.6 validation accuracy at 500 plus 0.4 at 5000, then accuracy_500 and candidate_id",
        "search_archive": str(archive_path),
        "search_archive_sha256": archive_hash,
        "finalist_evaluations": str(evaluations_path),
        "finalist_evaluations_sha256": _sha256(evaluations_path),
        "candidate_count": evaluations.get("candidate_count"),
        "finalist_count": len(finalists),
        "transfer_protocol": {
            "datasets": [
                "emnist_digits",
                "emnist_letters",
                "kmnist",
                "fashion_mnist",
            ],
            "representation_frozen": True,
            "classifier_refit_per_domain": True,
            "train_sizes": [500, 5_000],
            "train_seeds": [11, 23, 47],
            "transfer_test_sample_size": 2_000,
        },
        "finalists": finalists,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("results/v3_program_search_discovery.jsonl"),
    )
    parser.add_argument(
        "--evaluations",
        type=Path,
        default=Path("results/v3_finalist_evaluations.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v3_frozen_finalists.json"),
    )
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    manifest = freeze_finalists(
        args.archive,
        args.evaluations,
        args.output,
        count=args.count,
    )
    print(
        json.dumps(
            {
                "finalist_count": manifest["finalist_count"],
                "candidate_ids": [
                    item["candidate_id"] for item in manifest["finalists"]
                ],
                "search_archive_sha256": manifest["search_archive_sha256"],
                "test_set_accessed": manifest["test_set_accessed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
