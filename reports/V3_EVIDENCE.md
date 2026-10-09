# V3 Relational Bias Search — Evidence and Claim Boundary

## Protocol and provenance

The search used 20 deterministic seed programs and 10 generations of 18 Qwen
proposals. All 200 candidates were evaluated on MNIST validation at 500 training
examples. The archive occupies 21 MAP-Elites descriptor cells. Proposal batches
were balanced across order-sensitive, spatial-relational, graph-relational, and
free-exploration mechanisms; archive program counts, including the 20 seed
programs, were 61, 48, 40, and 51 respectively. The archive records 185 LLM
response batches and 27 proposals
rejected for mechanism-family mismatch. The V3 search report contains the
per-cell ASTs, scores, feature widths, runtime, and novelty measures:
[`V3_SEARCH.md`](V3_SEARCH.md).

The archive SHA-256 is
`452917a23a9e1be2b252e9739445e514bb119450c99c21020790d05e0cb2e686`.
The frozen-finalist manifest SHA-256 is
`a2a2215e6d25ddb1e8d4e35ae084453117994b3e07341b71fdf522274ed33435`.
The manifest records `test_set_accessed: false`. The top candidate,
`angle_centroid_pairwise`, was proposed in generation 7 rather than supplied as
a complete seed feature. Its AST subtracts the spatially conditioned centroid
histogram from the spatially conditioned stroke-angle histogram. All five
finalists are orderless and use localized path geometry; no order-sensitive
program survived into the finalist set.

## MNIST validation

The top 20 were re-evaluated at 500 and 5,000 training examples. The top five
were then evaluated with training seeds 11, 23, and 47; entries below are mean
accuracy ± standard deviation across those seeds. Evaluation uses the frozen
learner and validation partition only.

| Finalist | Dimension | 500 examples | 5,000 examples |
|---|---:|---:|---:|
| `angle_centroid_pairwise` | 30 | 0.8730 ± 0.0057 | 0.8945 ± 0.0029 |
| `regional_turn_histogram` | 60 | 0.8660 ± 0.0014 | 0.9063 ± 0.0014 |
| `horizontal_turn_spatial` | 30 | 0.8603 ± 0.0049 | 0.8853 ± 0.0049 |
| `regional_turn_profile` | 36 | 0.8585 ± 0.0063 | 0.8845 ± 0.0050 |
| `vertical_turn_gradient` | 42 | 0.8587 ± 0.0063 | 0.8843 ± 0.0040 |

For comparison, the 16-dimensional zoning control scored 0.6910 / 0.7265 and
the dimension-matched 30-dimensional zoning control scored 0.7615 / 0.8245 at
500 / 5,000 examples on MNIST validation. These controls use the same learner
and do not access the test partition. The V2 raw-free `cycle_angle_hist`
reference scored 0.7180 at 500 examples; V2's `raw + angle_hist` result used
raw pixels and is not a raw-free comparison.

Fixed-probe behavioral novelty was computed on 128 MNIST validation images
without labels. `angle_centroid_pairwise` has structural novelty 0.056 and
behavioral novelty 0.371 against the archive. This is a useful, measurable
composition, but its novelty score is moderate rather than evidence that its
behavior is wholly unlike other searched programs.

## Mechanism evidence

The finalists were frozen before the held-out MNIST interventions. Each
counterfactual reassigned angle events to locations while preserving the
per-image event-value multiset and location multiset. The classifier was trained
on 5,000 examples and evaluated on the same stratified 2,000-image official-test
sample; three intervention seeds were used. The table shows the original score
and mean counterfactual score across intervention seeds.

| Finalist | Original accuracy | Counterfactual accuracy | Mean change | Changed feature rows / 2,000 |
|---|---:|---:|---:|---:|
| `angle_centroid_pairwise` | 0.9060 | 0.6585 | -0.2475 | 1,900–1,906 |
| `regional_turn_histogram` | 0.9110 | 0.5612 | -0.3498 | 1,966–1,973 |
| `horizontal_turn_spatial` | 0.8910 | 0.5832 | -0.3078 | 1,900–1,906 |
| `regional_turn_profile` | 0.8885 | 0.5825 | -0.3060 | 1,899–1,904 |
| `vertical_turn_gradient` | 0.8900 | 0.5790 | -0.3110 | 1,900–1,906 |

No intervention failed on these samples. The large, consistent drops support the
claim that the mapping between stroke-angle events and image location matters
for these representations on MNIST.

Validation-only mechanism ablations support the same conclusion. Removing
spatial conditioning from the finalists reduced accuracy by 0.2275–0.2395 at
500 examples and 0.2320–0.2490 at 5,000 examples. For
`angle_centroid_pairwise`, removing spatial conditioning yielded 0.6310 / 0.6465
at 500 / 5,000. Replacing its pairwise difference with concatenated global
marginals produced the same scores (20D), so this run does not isolate a further
benefit from the subtraction itself. Order and curvature ablations were not
applicable to these five ASTs because none contains an order-sensitive operator
or `delta_angle`.

## Transfer evaluation

Finalists and baselines were frozen before transfer tests. Each entry below is
mean accuracy across training seeds 11, 23, and 47 on a stratified 2,000-image
sample from the dataset's official test split. The same
`StandardScaler + LogisticRegression(C=1)` learner was refit per domain. V3
ranges show the five frozen programs; the 30D zoning result is the compact,
dimension-matched classical control. HOG is a high-dimensional reference.

| Dataset | V3 range, 500 | V3 range, 5,000 | Zoning 30D, 500 / 5,000 | Raw, 5,000 | HOG, 5,000 | V2 best, 5,000 |
|---|---:|---:|---:|---:|---:|---:|
| EMNIST Digits | 0.8902–0.9048 | 0.9242–0.9302 | 0.8145 / 0.8573 | 0.8952 | 0.9705 | 0.8120 |
| EMNIST Letters | 0.4842–0.5233 | 0.5800–0.6187 | 0.5053 / 0.5967 | 0.5643 | 0.8072 | 0.4993 |
| KMNIST | 0.4027–0.4303 | 0.4555–0.5167 | 0.5345 / 0.6030 | 0.6012 | 0.7948 | 0.4538 |
| Fashion-MNIST | 0.4158–0.4328 | 0.4670–0.5253 | 0.7292 / 0.7648 | 0.7838 | 0.8200 | 0.7080 |

All five V3 programs beat the 30D zoning control on EMNIST Digits at both train
sizes. On EMNIST Letters, two of five beat zoning at 5,000 examples and one of
five does so at 500. None beats zoning, raw pixels, or HOG on KMNIST or
Fashion-MNIST. HOG remains stronger than V3 on every transfer domain. Thus,
these results show useful transfer to EMNIST Digits and limited transfer to
EMNIST Letters, but they do not meet the predeclared strong-success condition
for EMNIST and KMNIST together and do not establish a universal handwriting
prior.

## Outcome against the predeclared criteria

- **Minimum success: met.** A 30D V3 relational program exceeds V2's 71.8%
  raw-free 500-example reference and the 30D zoning control on MNIST validation.
- **Strong success: partial.** V3 exceeds compact controls on MNIST and transfers
  to EMNIST Digits, with limited EMNIST Letters gains, but fails the KMNIST
  transfer comparison.
- **Very strong result: not met.** The proposed spatial event-location
  mechanism is supported by ablation and counterfactual evidence, but transfer
  is domain-limited and V3 does not beat HOG. The claim should remain that the
  search found a compact, useful spatial structural prior for MNIST/EMNIST
  Digits, not a generally transferable handwriting law.

The V2 synthesis report remains unchanged. The detailed machine-readable
artifacts are `results/v3_search_summary.json`,
`results/v3_multiseed_validation.json`, `results/v3_ablation_report.json`,
`results/v3_counterfactual_report.json`, and
`results/v3_transfer_report.json`.
