"""Paired uncertainty and exact disagreement tests for shared examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest


def paired_bootstrap_delta(
    targets: NDArray,
    baseline_predictions: NDArray,
    candidate_predictions: NDArray,
    *,
    replicates: int = 2_000,
    seed: int = 20_261_010,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Bootstrap accuracy(candidate)-accuracy(baseline) on paired example IDs."""
    truth = np.asarray(targets, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    if baseline.ndim == 1:
        baseline = baseline[None, :]
    if candidate.ndim == 1:
        candidate = candidate[None, :]
    if baseline.shape != candidate.shape or baseline.ndim != 2:
        raise ValueError("paired predictions must have equal (seeds, examples) shapes")
    if baseline.shape[1] != len(truth) or truth.ndim != 1 or not len(truth):
        raise ValueError("targets must align with prediction example columns")
    if replicates <= 0 or not 0 < confidence < 1 or seed < 0:
        raise ValueError("invalid bootstrap configuration")

    observed = float(np.mean(candidate == truth) - np.mean(baseline == truth))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(truth), size=len(truth))
        base_accuracy = np.mean(baseline[:, sampled] == truth[sampled])
        candidate_accuracy = np.mean(candidate[:, sampled] == truth[sampled])
        draws[index] = candidate_accuracy - base_accuracy
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail))
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "replicates": replicates,
        "seed": seed,
    }


def mcnemar_exact(
    targets: NDArray,
    baseline_predictions: NDArray,
    candidate_predictions: NDArray,
) -> dict[str, float | int]:
    """Compute exact two-sided McNemar p-value from paired disagreements."""
    truth = np.asarray(targets, dtype=np.int64)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    if truth.ndim != 1 or baseline.shape != truth.shape or candidate.shape != truth.shape:
        raise ValueError("McNemar inputs must be aligned 1D arrays")
    b = int(np.count_nonzero((baseline != truth) & (candidate == truth)))
    c = int(np.count_nonzero((baseline == truth) & (candidate != truth)))
    total = b + c
    p_value = 1.0 if total == 0 else float(binomtest(min(b, c), total, p=0.5, alternative="two-sided").pvalue)
    return {"baseline_wrong_candidate_correct": b, "baseline_correct_candidate_wrong": c, "p_value": p_value}
