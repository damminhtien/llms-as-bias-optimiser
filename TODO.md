# V3.1 Bias Evaluation Checklist

V3.1 evaluates the already frozen V3 representations. It does not propose or
search for new biases. `PLAN.md`, `reports/V3_SEARCH.md`,
`reports/V3_EVIDENCE.md`, the V3 finalist manifest, and candidate ASTs remain
the V3 record and must not be rewritten by this study.

## Freeze and protocol

- [x] Start `v3.1-bias-evaluation` from `v3-relational-bias-search`.
- [x] Update `ARCHITECTURE.md` with the V3.1 evaluation boundary and design.
- [x] Create this V3.1 checklist without rewriting the frozen V3 `PLAN.md`.
- [x] Verify the V3 report, evidence, finalist manifest, candidate ASTs, and
  source branch are unchanged; record their hashes in the V3.1 manifest.
- [x] Write `V3_1_PROTOCOL.md` and canonical `results/v31/protocol.json` before
  computing any V3.1 result. Freeze hypotheses, representations, probes,
  datasets, sizes, seeds, transformations, metrics, and primary comparisons.
- [x] Hash the canonical protocol and verify repeatable serialization.

## Evaluation package

- [x] Add `src/bias_optimizer/evaluation/` with schema, representation registry,
  probes, protocol, runner, metrics, statistics, complementarity, robustness,
  and artifact modules.
- [x] Define stateful `Representation.fit/transform`; wrap all five V3
  finalists, V2 `cycle_angle_hist`, raw pixels, HOG, zoning30, and train-fitted
  HOG-PCA30/HOG-PCA60.
- [x] Define fresh `Probe` instances for fixed logistic regression, RBF-SVM,
  and small MLP configurations; keep kNN secondary. Do not change the V3
  `Learner` class or tune probes against test performance.
- [x] Add deterministic stratified split IDs shared across representations for
  each dataset, size, and seed. Preserve train/evaluation alignment.
- [x] Add split-aware feature caching. Cache stateless full-dataset features
  and HOG inputs; never cache PCA outputs across training splits.
- [x] Add a generic runner that fits each representation and probe fresh on
  training data only, retains predictions, and records feature, fit, and
  inference timings, accuracy, error, and confusion matrix.
- [x] Add learning curves at 100, 250, 500, 1,000, 2,500, and 5,000 samples
  with seeds 11, 23, and 47; report trapezoidal AULC over log sample size.
- [x] Add dimension-matched HOG-PCA comparisons and accuracy/dimension Pareto
  frontiers. Do not divide accuracy by dimension into a scalar score.
- [x] Add HOG+V3 and raw+V3 complementarity for the top three V3
  representations at sizes 500 and 5,000, with linear and MLP probes.
- [x] Add targeted mechanism destruction with a matched nuisance control;
  report both accuracy drops and intervention coverage.
- [x] Add clean-trained robustness evaluations for frozen translation,
  rotation, dilation, erosion, stroke-width, and noise transformations.
- [x] Add same-matrix transfer evaluation for MNIST, EMNIST Digits, EMNIST
  Letters, KMNIST, and Fashion-MNIST, preserving each dataset's provenance and
  the frozen V3 test-access boundary.
- [x] Add paired bootstrap confidence intervals and McNemar tests for primary
  comparisons. Report mean and SD over subset seeds; do not use a t-test with
  three seeds.
- [x] Add artifact manifests, per-run predictions, summaries, and profile
  generation under `results/v31/`.

## Leakage and validation gates

- [x] Test shared split indices, repeatable and distinct seeds, PCA train-only
  fitting, scaler train-only fitting, fresh fitted objects, label isolation,
  concatenation alignment, prediction length, paired bootstrap alignment, and
  stable protocol hashing.
- [x] Run the narrow evaluation test suite and repository-required checks.
- [x] Run a smoke matrix with one representation, two probes, and two sample
  sizes; inspect artifacts before starting larger runs.
- [x] Run Phase A core matrix: nine representations × three primary probes ×
  four sample sizes (250, 500, 1,000, 5,000) × three seeds.
- [x] Run the two additional frozen sizes (100 and 2,500) to complete all six
  learning-curve points.
- [x] Analyze learning curves, learner-capacity rank changes, and the
  dimension/accuracy frontier.
- [x] Run Phase B complementarity for the top three V3 representations.
- [x] Run Phase C mechanism and robustness evaluations for the top two or three
  V3 representations.
- [x] Run Phase D transfer for representations supported by Phase A/B.
- [x] Generate `reports/V3_1_BIAS_EVALUATION.md` with the five predeclared
  research answers, evidence limits, and a multi-axis profile per bias.
- [x] Confirm frozen V3 files remain unchanged; inspect status and diff, run
  `git diff --check`, and commit the completed V3.1 work as one focused commit.
