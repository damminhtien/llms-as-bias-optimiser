from __future__ import annotations

import numpy as np

from bias_optimizer.evaluation.metrics import learning_curve_aulc, pareto_frontier
from bias_optimizer.evaluation.statistics import mcnemar_exact, paired_bootstrap_delta


def test_bootstrap_resamples_paired_examples_and_is_repeatable() -> None:
    targets = np.asarray([0, 1, 0, 1], dtype=np.int64)
    baseline = np.asarray([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=np.int64)
    candidate = np.asarray([[0, 1, 0, 1], [0, 1, 0, 1]], dtype=np.int64)
    result = paired_bootstrap_delta(targets, baseline, candidate, replicates=200, seed=9)
    repeated = paired_bootstrap_delta(targets, baseline, candidate, replicates=200, seed=9)
    assert result == repeated
    assert result["delta"] > 0
    assert result["ci_low"] <= result["delta"] <= result["ci_high"]


def test_exact_mcnemar_counts_only_disagreements() -> None:
    targets = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
    baseline = np.asarray([1, 1, 0, 0, 1], dtype=np.int64)
    candidate = np.asarray([0, 0, 1, 1, 0], dtype=np.int64)
    result = mcnemar_exact(targets, baseline, candidate)
    assert result["baseline_wrong_candidate_correct"] == 3
    assert result["baseline_correct_candidate_wrong"] == 2
    assert 0 <= result["p_value"] <= 1


def test_aulc_integrates_over_log_train_size_and_frontier_removes_dominated_points() -> None:
    value = learning_curve_aulc({100: 0.5, 1_000: 0.7})
    assert np.isclose(value, 0.6 * np.log(10))
    assert pareto_frontier([(30, 0.8), (60, 0.79), (90, 0.9)]) == [(30, 0.8), (90, 0.9)]
