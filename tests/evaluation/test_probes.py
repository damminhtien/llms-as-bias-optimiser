from __future__ import annotations

import numpy as np

from bias_optimizer.evaluation.probes import PROBE_DEFINITIONS


def test_probe_factories_are_fresh_and_predict_aligned_rows() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(80, 6)).astype(np.float32)
    labels = np.asarray([index % 2 for index in range(80)], dtype=np.int64)
    for name in ("linear_logreg", "rbf_svm", "small_mlp", "knn"):
        first = PROBE_DEFINITIONS[name].create(seed=11)
        second = PROBE_DEFINITIONS[name].create(seed=11)
        assert first is not second
        first.fit(features, labels)
        assert first.predict(features[:9]).shape == (9,)


def test_scaler_is_fitted_on_training_features() -> None:
    features = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probe = PROBE_DEFINITIONS["linear_logreg"].create(seed=23)
    probe.fit(features, labels)
    scaler = probe._model[0]
    np.testing.assert_allclose(scaler.mean_, features.mean(axis=0))
