"""Measure raw-pixel controls plus frozen discovery programs on MNIST validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import (
    ProgramConstraints,
    SearchTrack,
    validate_program,
)
from bias_optimizer.ml.evaluator import Evaluator, ProgramEvaluator
from bias_optimizer.ml.learner import LearnerConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_pixel_augmentations(
    manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compare raw pixels with raw pixels plus each frozen finalist AST."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("finalist_selection_frozen") is not True:
        raise ValueError("augmentation controls require a frozen finalist manifest")
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("finalists must be frozen before test-set access")
    archive = Path(manifest["search_archive"])
    if not archive.is_absolute() and not archive.exists():
        archive = manifest_path.parent.parent / archive
    archive_sha256 = _sha256(archive)
    if archive_sha256 != manifest.get("search_archive_sha256"):
        raise ValueError("discovery archive changed after finalist freeze")

    compiler = ProgramCompiler(
        ProgramConstraints(track=SearchTrack.AUGMENTATION, max_feature_dim=1_024)
    )
    evaluator = ProgramEvaluator(
        compiler=compiler,
        evaluator=Evaluator(learner_config=LearnerConfig()),
    )
    image = Expr(op="image")
    raw_pixels = Expr(op="flatten_pixels", args=(image,))
    controls = [
        (
            "raw_pixels",
            "raw_pixels_baseline",
            None,
            raw_pixels,
        )
    ]
    discovery_constraints = ProgramConstraints(
        track=SearchTrack.DISCOVERY,
        max_feature_dim=128,
    )
    for finalist in manifest["finalists"]:
        bias = ProgramBiasSpec.from_dict(finalist["bias"])
        info = validate_program(bias.program, discovery_constraints)
        if info.depth >= 6:
            continue
        combined = Expr(op="concat", args=(raw_pixels, bias.program))
        name = f"raw_plus_{bias.name}"
        controls.append(
            (
                name,
                "raw_pixels_plus_frozen_discovery_program",
                finalist["candidate_id"],
                combined,
            )
        )
    spatial = Expr(
        op="spatial_split",
        args=(image,),
        params={"rows": 2, "cols": 3},
    )
    controls.append(
        (
            "raw_plus_spatial_2x3",
            "raw_pixels_plus_spatial_control",
            None,
            Expr(op="concat", args=(raw_pixels, spatial)),
        )
    )

    results: list[dict[str, Any]] = []
    for name, kind, source_candidate_id, program in controls:
        bias = ProgramBiasSpec(
            name=name,
            hypothesis="A fixed raw-pixel representation provides an augmentation control.",
            mechanism="Use raw pixels alone or concatenate them with a frozen structural feature vector.",
            program=program,
            prediction="Pixel augmentation may improve low-data validation accuracy.",
            falsification="Reject the augmentation claim if it does not beat the raw-pixel control.",
        )
        evaluation = evaluator.evaluate(bias)
        results.append(
            {
                "name": name,
                "kind": kind,
                "candidate_id": program_bias_hash(bias),
                "source_discovery_candidate_id": source_candidate_id,
                "feature_dim": evaluation.feature_dim,
                "accuracy_500": evaluation.accuracy_500,
                "accuracy_5000": evaluation.accuracy_5000,
                "feature_runtime_ms": evaluation.feature_runtime_ms,
            }
        )

    baseline = results[0]
    for result in results:
        result["delta_accuracy_500_vs_raw_pixels"] = (
            result["accuracy_500"] - baseline["accuracy_500"]
        )
        result["delta_accuracy_5000_vs_raw_pixels"] = (
            result["accuracy_5000"] - baseline["accuracy_5000"]
        )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": _sha256(manifest_path),
            "search_archive_sha256": archive_sha256,
            "official_test_accessed": False,
            "selection_data": "MNIST validation split only",
            "train_sizes": [500, 5_000],
            "train_seed": LearnerConfig().seed,
            "learner": "StandardScaler + LogisticRegression(C=1, lbfgs, max_iter=1000)",
            "compositions_are_llm_proposals": False,
            "claim_boundary": "These post-search controls compose raw pixels with depth-compatible frozen finalists plus a fixed spatial control; they are not new LLM proposals and use the same validation split that selected the finalists.",
        },
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
        "--manifest", type=Path, default=Path("results/v2_transfer_finalists.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v2_pixel_augmentation_controls.json"),
    )
    args = parser.parse_args()
    report = evaluate_pixel_augmentations(args.manifest, args.output)
    print(f"Saved raw-pixel augmentation controls to {args.output}")
    for result in report["results"]:
        print(
            f"{result['name']}: @500 {result['accuracy_500']:.4f} "
            f"(Δ {result['delta_accuracy_500_vs_raw_pixels']:+.4f}); "
            f"@5000 {result['accuracy_5000']:.4f} "
            f"(Δ {result['delta_accuracy_5000_vs_raw_pixels']:+.4f})"
        )


if __name__ == "__main__":
    main()
