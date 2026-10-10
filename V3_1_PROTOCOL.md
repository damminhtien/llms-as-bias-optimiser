# V3.1 Bias Evaluation Protocol

**Frozen before V3.1 experiments.** This study evaluates the existing V3
finalists; it does not search for or add bias candidates. V3 reports, source
archives, candidate ASTs, and finalist manifests remain unchanged. The
machine-readable protocol and provenance hashes are in `results/v31/`.

## Research questions

1. Does V3 reduce sample requirements compared with dimension-matched controls?
2. Does nonlinear probing recover task performance unavailable to a linear
   probe, and does representation ranking change with learner capacity?
3. How do V3 representations compare with HOG-PCA at the same feature width?
4. Does V3 add predictive signal to HOG and raw pixels?
5. Do targeted interventions, nuisance controls, robustness tests, and related
   domain transfer behave as the proposed bias mechanisms predict?

The estimand is accuracy as a function of representation, probe, sample size,
and dataset. No single downstream accuracy is interpreted as absolute retained
information or a universal bias ranking.

## Frozen representations

The exact V3 and V2 candidate IDs, AST hashes, source manifest hashes, and
archives are recorded in `results/v31/manifest.json`.

| ID | Representation | Dimension | Role |
|---|---|---:|---|
| `angle_centroid_pairwise` | Frozen V3 finalist | 30 | Core V3; complementarity/mechanism |
| `regional_turn_histogram` | Frozen V3 finalist | 60 | Core V3; complementarity/mechanism |
| `horizontal_turn_spatial` | Frozen V3 finalist | 30 | Core V3; complementarity/mechanism |
| `regional_turn_profile` | Frozen V3 finalist | 36 | Frozen registry; eligible for transfer |
| `vertical_turn_gradient` | Frozen V3 finalist | 42 | Frozen registry; eligible for transfer |
| `cycle_angle_hist` | Frozen V2 finalist, first 16D candidate | 16 | Core V2 control |
| `zoning_30d` | 5 × 6 zoning | 30 | Compact dimension-matched control |
| `hog_pca_30d` | HOG followed by PCA(30) | 30 | Dimension-matched classical control |
| `hog_pca_60d` | HOG followed by PCA(60) | 60 | Dimension-matched classical control |
| `hog` | 9 orientations, 4 × 4 cells, 2 × 2 blocks | 1,296 | Strong classical reference |
| `raw_pixels` | Flattened normalized 28 × 28 image | 784 | Pixel control |

The Phase A matrix uses the first three V3 representations plus V2 cycle-angle,
zoning, both HOG-PCA controls, HOG, and raw pixels (nine total). Registry order
and stable IDs are stored in the JSON protocol.

## Frozen probes

Every probe is wrapped in `StandardScaler` fitted on training features only.
Probe estimators are fresh per run and never tuned against evaluation results.

| Probe | Fixed settings |
|---|---|
| `linear_logreg` | LogisticRegression(C=1, solver=`lbfgs`, max_iter=1000, random_state=seed) |
| `rbf_svm` | SVC(kernel=`rbf`, C=10, gamma=`scale`, random_state=seed) |
| `small_mlp` | MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-4, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, max_iter=300, random_state=seed) |
| `knn` (secondary) | KNeighborsClassifier(n_neighbors=5, weights=`distance`) |

Only the first three probes are primary. Any optional sensitivity analysis must
use inner cross-validation on that run's training data only.

## Splits and sample sizes

- MNIST core work uses `load_mnist_search_data`: the official 60,000-image
  training partition split into 58,000 training-pool images and a fixed,
  stratified 2,000-image validation set (split seed 42). No official test
  labels enter Phase A or B.
- Each `(dataset, train_size, seed)` resolves one stratified training-index
  vector from the dataset training pool. Every representation and probe shares
  that vector. Training seeds are 11, 23, and 47.
- The core matrix uses train sizes 250, 500, 1,000, and 5,000: 9 × 3 × 4 × 3 =
  324 fits. The learning-curve extension adds sizes 100 and 2,500 for the same
  matrix, yielding six-point curves without changing the core matrix.
- AULC is trapezoidal integration of mean accuracy against `log(train_size)`;
  its units and integration bounds are reported. Every accuracy point is also
  reported.
- Transfer datasets use their pinned train partitions and one fixed
  stratified 2,000-image official-test sample (seed 31,415). Source checksums,
  preprocessing, and partition provenance are retained.

## Evaluation phases

### Phase A: core matrix and learning curves

Run the nine core representations against the three primary probes on MNIST
validation at the frozen sizes and seeds above. Report accuracy, error,
confusion matrix, fit/feature/inference time, feature width, and all per-example
predictions. Compare learner rankings and report an accuracy-versus-dimension
Pareto frontier. Do not divide accuracy by dimension into a scalar score.

### Phase B: complementarity

For each of the three core V3 representations, compare HOG and HOG+V3, and raw
pixels and raw+V3, using `linear_logreg` and `small_mlp` at 500 and 5,000
training examples with all three seeds. Report the paired accuracy change on the
same MNIST validation examples.

### Phase C: mechanism and robustness

For the three core V3 representations, train once per seed on 5,000 MNIST
training-pool examples using `linear_logreg` and evaluate on a fixed stratified
2,000-image sample from the official MNIST test partition. Compare the original model with a
targeted event-location reassignment that preserves the event-value and
location marginals. The nuisance control adds deterministic low-amplitude
Gaussian pixel noise; its scale is calibrated on training images only to match
the target intervention's mean feature-space displacement. Keep the clean-fit
representation and probe fixed for both interventions.

The same clean-trained models are tested on horizontal and vertical
translations of ±1/±2 pixels, rotations of ±5°/±10°, one-pixel dilation,
one-pixel erosion, and Gaussian pixel noise (sigma 0.05 on normalized [0,1]
images). No transformed examples are used for fitting. Report accuracy change
and prediction agreement for every transform.

### Phase D: transfer

After A/B, choose the two V3 representations with the highest mean linear
validation accuracy at 5,000 samples among those whose mean linear validation
accuracy at 500 exceeds zoning30's mean at 500, or that show positive
`linear_logreg` HOG complementarity at 500 or 5,000 in at least two of three
seeds. This is a validation-only selection rule. Evaluate these
two plus HOG, raw pixels, zoning30, and HOG-PCA at the matching 30D/60D widths
on EMNIST Digits, EMNIST Letters, KMNIST, and Fashion-MNIST. Use the primary
probe matrix, training sizes 500 and 5,000, seeds 11/23/47, and fixed test
samples. Report dataset provenance and interpret transfer by domain similarity.

## Metrics and paired statistics

- Accuracy and error rate, confusion matrix in sorted label order, feature
  dimension, feature-extraction time, fit time, and inference time.
- Learning-curve AULC, learner-dependent accuracy gain (nonlinear minus
  linear), Pareto frontier, complementarity delta, transformed accuracy drop,
  and prediction agreement.
- Paired bootstrap resamples the same evaluation-example indices for both
  models, with replacement, 2,000 replicates, percentile 95% confidence
  intervals, and RNG seed 20,261,010. When combining training seeds, each
  resampled example index is shared across seeds.
- McNemar uses exact two-sided binomial testing on paired model disagreements.
  Primary pairs are V3 versus zoning30, V3 versus HOG-PCA30, and HOG+V3 versus
  HOG. Preserve per-example predictions for every pair.
- Report mean and SD over the three training-subset seeds as subset variability;
  do not run a t-test over three seeds. Do not label probe performance as
  absolute information retention.

## Leakage, caching, and reproducibility

Every run creates a new representation, fitted scaler, and classifier. The
representation sees training images and optional training labels only. PCA is
fit on sampled training HOG features and then applied to train/evaluation
features. No global PCA transform is cached. Stateless features may be cached
by dataset fingerprint, split ID, representation ID/version, and input-index
set. Fitted feature cache keys also include train size, seed, and fitted-state
identity. Protocol serialization is canonical JSON and its SHA-256 is frozen in
the V3.1 manifest before experiment artifacts are created.

Required gates include same indices across representations, deterministic and
distinct seeds, train-only PCA/scaling, fresh fitted objects, no evaluation
labels in `Representation.fit`, aligned concatenation and predictions, paired
bootstrap indices, and stable protocol hashing. Run a one-representation ×
two-probe × two-size smoke matrix before Phase A. Do not change hypotheses or
representations after viewing results; any deviation is documented as
exploratory and kept out of the primary analysis.
