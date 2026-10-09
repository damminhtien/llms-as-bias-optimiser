"""Evaluate the five human seed biases on MNIST validation data without an LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bias_optimizer.cache import FeatureCache
from bias_optimizer.data.mnist import MNISTDataConfig
from bias_optimizer.domain.bias import bias_spec_hash
from bias_optimizer.domain.seed_biases import initial_human_biases
from bias_optimizer.ml.evaluator import Evaluator
from bias_optimizer.ml.learner import LearnerConfig


def run_seed_evaluations(
    *,
    seed: int = 42,
    validation_size: int = 2_000,
    data_dir: Path = Path("data"),
    cache_dir: Path = Path("cache/features"),
) -> dict[str, object]:
    """Evaluate each fixed seed and return a reproducible validation report."""
    evaluator = Evaluator(
        LearnerConfig(seed=seed),
        data_config=MNISTDataConfig(
            seed=seed,
            validation_size=validation_size,
            data_dir=data_dir,
        ),
        feature_cache=FeatureCache(cache_dir),
    )
    records: list[dict[str, object]] = []
    for bias in initial_human_biases():
        evaluation = evaluator.evaluate(bias)
        records.append(
            {
                "candidate_id": bias_spec_hash(bias),
                "bias": bias.to_dict(),
                "accuracy_500": evaluation.accuracy_500,
                "accuracy_5000": evaluation.accuracy_5000,
                "feature_dim": evaluation.feature_dim,
                "feature_runtime_ms": evaluation.feature_runtime_ms,
                "training_runtime_ms": evaluation.training_runtime_ms,
                "inference_runtime_ms": evaluation.inference_runtime_ms,
                "ranking_score": evaluation.ranking_score,
                "confusion_matrix_5000": evaluation.confusion_matrix.tolist(),
            }
        )
        print(
            f"{bias.name}: accuracy@500={evaluation.accuracy_500:.4f}, "
            f"accuracy@5000={evaluation.accuracy_5000:.4f}, "
            f"ranking_score={evaluation.ranking_score:.5f}"
        )

    return {
        "dataset": "OpenML mnist_784 version 1",
        "split_seed": seed,
        "validation_size": validation_size,
        "training_sizes": [500, 5_000],
        "validation_is_the_only_search_evaluation_set": True,
        "llm_used": False,
        "results": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=int, default=2_000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/features"))
    parser.add_argument("--output", type=Path, default=Path("results/seed_biases.json"))
    args = parser.parse_args()

    report = run_seed_evaluations(
        seed=args.seed,
        validation_size=args.validation_size,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved seed evaluation results to {args.output}")


if __name__ == "__main__":
    main()
