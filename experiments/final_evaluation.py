"""Evaluate frozen MNIST finalists on the official test partition."""

from __future__ import annotations

import argparse
from pathlib import Path

from bias_optimizer.ml.final_evaluation import run_final_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalists", type=Path, default=Path("results/finalists.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/final_evaluation.json")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/final_evaluation")
    )
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[250, 500, 1_000, 5_000, 60_000],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    args = parser.parse_args()
    report = run_final_evaluation(
        finalists_path=args.finalists,
        output_path=args.output,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        train_sizes=tuple(args.train_sizes),
        seeds=tuple(args.seeds),
    )
    print(f"Saved final evaluation to {args.output}")
    for result in report["results"]:
        metrics = result["summaries"].get("500")
        if metrics is not None:
            print(
                f"{result['name']}: accuracy@500="
                f"{metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}"
            )


if __name__ == "__main__":
    main()
