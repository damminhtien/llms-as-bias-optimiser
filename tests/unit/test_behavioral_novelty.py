"""Behavioral novelty compares program effects across fixed probe images."""

import numpy as np

from bias_optimizer.novelty.behavior import (
    behavioral_novelty,
    distance_profile_similarity,
    pairwise_distance_profile,
)


def test_distance_profiles_detect_equivalent_program_behavior_across_widths() -> None:
    base = np.asarray([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
    equivalent = np.column_stack((base[:, 0] * 2.0, base[:, 0] * -3.0))

    left = pairwise_distance_profile(base)
    right = pairwise_distance_profile(equivalent)

    assert distance_profile_similarity(left, right) == 1.0
    novelty, nearest_id, similarity = behavioral_novelty(
        right, {"base": left}, candidate_id="candidate"
    )
    assert novelty == 0.0
    assert nearest_id == "base"
    assert similarity == 1.0


def test_behaviorally_distinct_profile_gets_novelty_against_archive() -> None:
    base = np.asarray([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
    distinct = np.asarray([[0.0], [9.0], [2.0], [4.0]], dtype=np.float32)
    left = pairwise_distance_profile(base)
    right = pairwise_distance_profile(distinct)

    novelty, nearest_id, similarity = behavioral_novelty(
        right, {"base": left}, candidate_id="candidate"
    )

    assert nearest_id == "base"
    assert novelty == 1.0 - max(0.0, similarity)
    assert novelty > 0.0
