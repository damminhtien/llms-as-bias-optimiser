# Architecture — V3 Relational Bias Search

## V3 active architecture

### Research boundary

V3 asks whether LLM-guided typed program synthesis can discover compact,
relational, transferable structural priors. V2 is frozen at `v2-final`; all V3
archives use separate names and the V2 final report remains untouched. The LLM
proposes falsifiable hypotheses and bounded ASTs. Local deterministic code owns
type checking, execution, feature extraction, scoring, archiving, and evaluation.

### Search and evidence flow

```text
LLM hypothesis + typed representation AST + mechanism focus
                         │
                         ▼
              typed validator and compiler
                         │
                         ▼
             bounded relational DSL runtime
                         │
                         ▼
              fixed train/validation learner
                         │
                         ▼
        five-axis MAP-Elites + structural novelty
              │          │           │
              │          │           └── behavioral novelty on frozen probes
              │          └────────────── mechanism-specific falsification
              └───────────────────────── frozen transfer and matched baselines
```

No generated Python is executed. Discovery rejects raw-pixel primitives and
caps feature width at 128. Augmentation is a separate track where a deterministic
raw-pixel anchor is concatenated with each structural program. The augmentation
track measures added information; it does not decide whether raw pixels belong
in the representation.

### Behavioral descriptor and archive

Every program receives the descriptor

```text
source × order × spatial × composition × complexity
```

with `source ∈ {topology, graph, path_geometry, pixel, mixed}`,
`order ∈ {orderless, first_order, higher_order}`, `spatial ∈ {global, localized}`,
`composition ∈ {single, composite}`, and
`complexity ∈ {compact, moderate, deep}`. The archive retains the best quality
candidate for each five-axis cell. Source describes the output signal: an angle
histogram is `path_geometry` even though its implementation traverses a graph
and paths. Multiple output signal types are `mixed`. The handcrafted descriptor
suite fills 12 distinct cells, and descriptor tests pass before V3 search.

The proposer receives underexplored descriptor cells rather than a single
primary niche. V3 proposal batches are balanced across order-sensitive,
spatial-relational, graph-relational, and free exploration mechanisms. The
archive remains open to other valid cells; the target-cell list guides prompts
but does not reject novel descriptor combinations.

### Relational DSL contract

The four V3 additions are deliberately relational rather than a larger list of
complete feature operators:

```text
spatial_condition(Sequence) -> Vector
pairwise_difference(Vector, Vector) -> Vector
cross_histogram(Sequence, Sequence) -> Vector
path_summary(PathSet) -> Sequence
```

`spatial_condition` summarizes sequence events by their carried image
coordinates. Sequence transformations preserve or aggregate event locations so
that conditioning remains well-defined after first and higher differences.
`pairwise_difference` requires equal static vector widths. `cross_histogram`
aligns events by nearest image location when both sequences carry coordinates,
and otherwise interpolates along normalized event progress before building a
bounded joint histogram. `path_summary` returns one selected deterministic
statistic per path with its centroid attached as location metadata. These contracts let programs
express curvature by height, turning persistence, loop-position interaction,
and branch-location interaction without naming those complete features.

### Evaluation gates

The 40-candidate pilot (ten explicit seeds balanced across order-sensitive,
spatial-relational, graph-relational, and free-exploration mechanisms plus
2 × 15 proposals) precedes the 200-candidate search. Before scaling up, review cell occupancy, invalid and
duplicate rates, feature dimensions, order-sensitive coverage, and relational
candidate survival. At least 30% of valid LLM proposals, excluding the
hand-built seeds, must fall outside the global-histogram family. A failed gate
stops the run for archive or proposer repair.

Only a passing pilot proceeds to 20 seeds plus 10 × 18 proposals. The
`--stage1-500-only` search mode records `accuracy_5000: null` and ranks programs
with validation accuracy at 500 examples plus the size/runtime penalties. The
separate finalist evaluator refits only the top 20 at 500 and 5,000 examples.
The default evaluator still uses both sizes, preserving the V2 protocol. The top
five are frozen with their ASTs and archive hash before any official test
partition is accessed. Transfer, counterfactual, and finalist-selection code
validates that frozen hash.

The frozen top five have a separate validation-only multi-seed runner using
training seeds 11, 23, and 47. It reports means and standard deviations at 500
and 5,000 examples without reopening finalist selection.

Behavioral novelty uses 128 fixed MNIST validation images without labels. Each
program is represented by pairwise Euclidean distances between probe images;
Pearson correlation compares programs across different output widths. Novelty is
`1 - max(0, similarity)` to the nearest other archived behavior, with
near-identical behavior receiving zero novelty.

Mechanism-specific counterfactuals preserve the nuisance statistic they name:

| Mechanism under test | Intervention | Preserved statistic |
| --- | --- | --- |
| Stroke order | Shuffle values within each path sequence | Per-path event-value multiset and value-location pairing |
| Spatial conditioning | Reassign event locations within each path | Global value and location marginals |
| Cross-variable relation | Reassign path centroids among summaries, or shuffle one input at `cross_histogram` | Path counts and graph-summary marginals; centroid marginals when multiple paths exist |
| Topology | Relocate a foreground pixel until connectivity or cycle rank changes | Foreground pixel count |
| Global path summary | Permute feature vectors between test images | Exact batch distribution of feature vectors |

The runner records intervention coverage, changed feature rows, prediction
agreement, and accuracy change. A low changed-row count reveals when a candidate
contains too few within-image relation events for a strong falsification.

Validation-only mechanism ablations rewrite the AST: `no_spatial` collapses a
conditional histogram into a global one; `no_order` replaces sequence-order
summaries with moments or the underlying sequence; `no_curvature` removes angle
differences; and `no_relation` replaces joint histograms with separate
marginal histograms. Reports include the resulting feature widths and accuracy
changes. The compact classical control is a 16-dimensional zoning descriptor.

### Current implementation status

- Five-axis behavioral descriptors and MAP-Elites cells are implemented.
- Handcrafted descriptor tests verify 12 distinct cells, including the
  orderless, first-order, higher-order, localized, and composite behaviors.
- The four typed relational primitives carry sequence locations, reject invalid
  alignments or widths, and have tests for spatial curvature, loop position,
  turning persistence, and branch location.
- Search requests are split across four mechanism families. The augmentation
  controller adds the raw-pixel anchor to each structural vector automatically.
- The pilot CLI can use ten explicit, statically validated seeds distributed
  across its four mechanism families; the regular search retains the broader
  deterministic seed generator.
- The 40-program pilot passed its predeclared gate: 29/30 valid LLM programs
  were outside the global-histogram family, with 17 occupied cells and balanced
  accepted family counts. Exact rates and failures are in
  [`reports/V3_PILOT.md`](reports/V3_PILOT.md).
- Order, spatial-location, cross-relation, and topology interventions are
  implemented and unit-tested. Validation-only mechanism ablations,
  fixed-probe behavioral novelty, and top-five multi-seed evaluation have
  dedicated runners and tests.
- The full discovery search completed 200 candidates across 21 descriptor
  cells. The Qwen archive contains 180 proposals and 27 mechanism-family
  mismatches. The best generated relational program,
  `angle_centroid_pairwise`, is a 30D orderless spatial composition proposed in
  generation 7.
- The top five scored 0.8585–0.8730 mean accuracy at 500 and 0.8843–0.9063 at
  5,000 MNIST validation examples across seeds 11/23/47. A matched 30D zoning
  baseline scored 0.7615 / 0.8245. See
  [`reports/V3_SEARCH.md`](reports/V3_SEARCH.md).
- The frozen finalists' event-to-location counterfactuals reduced accuracy by
  24.8–35.0 points on average on stratified 2,000-image MNIST test samples.
  Removing spatial conditioning reduced validation accuracy by 22.8–24.9
  points at 5,000 examples.
- Transfer evaluation used 2,000-image official-test samples and matched
  train sizes/seeds for EMNIST Digits/Letters, KMNIST, and Fashion-MNIST. All
  five V3 programs beat 30D zoning on EMNIST Digits; only two beat it on EMNIST
  Letters at 5,000 examples. None beat compact controls on KMNIST or
  Fashion-MNIST; HOG remained stronger on all four domains. This supports a
  useful MNIST/EMNIST Digits spatial prior, not a universal handwriting rule.
  Full protocols and scores are in [`reports/V3_EVIDENCE.md`](reports/V3_EVIDENCE.md).

## V3.1 frozen-bias evaluation architecture

### Research boundary

V3.1 does not search for new biases or change the V3 finalist set. It evaluates
the frozen V3 programs alongside the predeclared V2 and classical controls to
answer how an inductive bias should be evaluated. Performance is the joint
quantity

```text
performance = f(representation, learner, training size, domain)
```

so a low score with one learner is evidence about that representation-learner
pair, not proof that the representation is poor. V3's search reports, evidence,
finalist manifest, and candidate ASTs remain frozen and unchanged.

There is no LLM or search loop in V3.1. The evaluation code consumes a
predeclared registry and immutable protocol, then records evidence across
sample efficiency, learner capacity, dimension-matched controls,
complementarity, mechanism specificity, robustness, and transfer.

### Evaluation package and data flow

```text
frozen V3 programs + V2/classical controls
                    │
                    ▼
      stateful Representation registry
                    │
                    ▼
       split-aware feature matrix cache
                    │
                    ▼
          fixed Probe configuration
                    │
                    ▼
            evaluation runner
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  learning curves complementarity robustness/transfer
        └───────────┼───────────┘
                    ▼
      paired statistics and artifacts
                    │
                    ▼
       bias evaluation profile/report
```

Implementation belongs in `src/bias_optimizer/evaluation/`; experiment files
are thin command-line entry points. `Representation` has `fit(images, labels)`
and `transform(images)` methods because HOG-PCA is stateful. Every run creates
a fresh representation and probe, fits representations only on that run's
training images, then transforms train and evaluation images. The runner must
never expose evaluation labels to representation fitting. Stateless full-dataset
features may be cached; fitted PCA outputs are split- and seed-specific.

The frozen registry includes the five V3 finalists, the V2 `cycle_angle_hist`,
raw pixels, HOG, 30D zoning, and train-fitted HOG-PCA at 30D and 60D. Fixed
probes are standardized logistic regression (the V3 learner settings), RBF
SVM, and a small MLP; kNN is secondary. Probe settings are frozen in
`results/v31/protocol.json`; test performance must not tune them. Optional
sensitivity analysis uses inner cross-validation on training data only.

### Protocol and analysis

`V3_1_PROTOCOL.md` and its canonical JSON manifest freeze primary hypotheses,
representations, probes, datasets, sample sizes, seeds, transformations,
metrics, and comparisons before any V3.1 results are produced. For a fixed
`(dataset, size, seed)`, all representations use the same stratified training
indices. The primary MNIST learning curve uses sizes 100, 250, 500, 1,000,
2,500, and 5,000 with seeds 11, 23, and 47. It reports each point and the
trapezoidal accuracy area under the curve over log training size.

Core analysis crosses representations and probes; compares V3 with HOG-PCA at
the same dimensions; and reports a Pareto frontier over feature dimension and
accuracy. Complementarity compares HOG and raw pixels against their
concatenations with the three leading V3 representations. Mechanism analysis
compares a targeted counterfactual with a magnitude-matched nuisance control.
Robustness evaluates clean-trained models on frozen image transformations.
Transfer reuses the same representation/probe matrix on MNIST, EMNIST Digits,
EMNIST Letters, KMNIST, and Fashion-MNIST, with interpretation tied to domain
similarity rather than a universal-transfer claim.

Every result retains per-example predictions for paired bootstrap confidence
intervals and McNemar comparisons. Means and standard deviations across the
three training-subset seeds describe subset variability; they are not used as
a three-sample t-test. The final report presents a multi-axis profile for each
bias rather than a single aggregate score. Result files live under
`results/v31/` and the report is `reports/V3_1_BIAS_EVALUATION.md`.

### Leakage and reproducibility gates

The evaluation split registry is deterministic and independent of
representation identity. PCA and every scaler fit only on training data.
Every seed/run uses fresh fitted objects. Protocol and artifact manifests are
hashed, predictions align exactly with evaluation indices, and paired
statistics resample the same example indices for both models. Unit tests for
these conditions must pass before starting the core matrix. The prescribed
execution order and acceptance checklist are tracked in [`TODO.md`](TODO.md).

## V2 frozen architecture reference

### 1. Research boundary

V1 searched over subsets of a fixed seven-operator registry. Its results are the
frozen baseline. V2 adds a typed representation DSL so the LLM can compose
atomic operations into programs that were not written as complete feature
operators.

\[
\boxed{\text{LLM proposes a hypothesis and program; deterministic local code validates and evaluates it.}}
\]

The LLM never receives images or labels, accesses the official test set, changes
the learner, assigns fitness, or emits executable code. V1 models and archives
remain in place for reproduction; its architecture notes are preserved in
[`ARCHITECTURE_V1.md`](ARCHITECTURE_V1.md). V2 proposals use `ProgramBiasSpec`
and the separate program-search path.

### 2. V2 data flow

```text
LLMClient
   │ JSON: hypothesis + mechanism + Expr AST + prediction + falsification
   ▼
ProgramProposer
   │ parse, deduplicate, retry once on invalid output
   ▼
ProgramCompiler ─── ProgramConstraints
   │ static type checks, parameter bounds, track limits
   ▼
ProgramPipeline
   │ deterministic interpreter; no eval/import/source generation
   ▼
ProgramEvaluator ─── frozen StandardScaler + LogisticRegression
   │ train/validation only; feature cache isolated by V2 namespace
   ▼
ProgramSearchRecord
   │ score + AST descriptor + structural novelty + provenance
   ▼
MapElitesArchive
   │ best record per (primitive family, complexity) cell
   └──────────────────────────────► next bounded LLM context
```

The V1 `Evaluator`, fixed learner, and data split implementation are reused.
`Evaluator.evaluate_pipeline()` is the shared boundary for any compiled
fixed-width representation. V2 uses its own cache namespace and candidate
hash, so it cannot collide with V1 feature matrices.

### 3. Program domain and JSON contract

`Expr` is an immutable tree with three fields:

```json
{
  "op": "spatial_split",
  "args": [{"op": "image", "args": [], "params": {}}],
  "params": {"rows": 2, "cols": 3}
}
```
Each node has exactly `op`, `args`, and `params`; parameters must be finite JSON
values. `ProgramBiasSpec` contains:

```text
name, hypothesis, mechanism, program, prediction, falsification
```

Canonical JSON is the basis for SHA-256 identity and cache keys. AST identity,
rather than the prose explanation, is used to deduplicate representations.

### 4. Typed primitive language

The V2 allow-list is implemented in `dsl/primitives.py`. `AngleSequence` is a
typed `Sequence` specialization, so angle wrapping cannot be applied to graph
degree or edge-length sequences by mistake. Static signatures are:

```text
image                 () -> Image
threshold             Image -> BinaryImage
skeletonize           Image | BinaryImage -> Skeleton
graph                 Skeleton -> Graph
connected_components  BinaryImage -> Scalar
cycle_rank            Graph -> Scalar
paths                 Graph -> PathSet
degree_sequence       Graph -> Sequence
edge_lengths          PathSet -> Sequence
angles                PathSet -> AngleSequence
delta                 Sequence -> Sequence
delta_angle           AngleSequence -> AngleSequence
sign                  Sequence | AngleSequence -> Sequence
run_length_encode     Sequence -> Sequence
histogram             Sequence | AngleSequence -> Vector
moments               Sequence | AngleSequence -> Vector
quantiles             Sequence | AngleSequence -> Vector
autocorrelation       Sequence | AngleSequence -> Vector
normalize             Vector -> Vector
spatial_split         Image -> Vector
flatten_pixels        Image -> Vector
ratio                 Vector x Vector -> Vector
count                 Graph | PathSet | Sequence | AngleSequence -> Scalar
concat                Vector x Vector ... -> Vector
```

Parameters have operation-specific bounds. Vector widths are inferred statically
for histograms, moments, quantiles, autocorrelation, spatial splits, concatenation,
and ratios. A program root must be `Vector` or `Scalar`. Every other intermediate
type has to match its primitive signature before compilation succeeds.

The image input is depth zero. Default limits are operation depth 6 and 127 AST
nodes. The interpreter validates finite output and the statically inferred output
width for every image. Repeated identical subtrees are evaluated once per image;
a bounded cross-candidate cache reuses expensive validated subtrees when the
canonical AST and exact dataset-row key match.

### 5. Independent search tracks

| Track | Raw pixels | Default maximum width | Role |
| --- | --- | ---: | --- |
| `augmentation` | `flatten_pixels` allowed | 1,024 | Measure structural additions to the pixel baseline |
| `discovery` | `flatten_pixels` rejected anywhere in the tree | 128 | Search compact structural representations |

Each run has a distinct JSONL archive and feature-cache namespace. The discovery
track rejects raw pixels statically, before invoking the evaluator. Both tracks
use the same train/validation protocol and fixed learner as V1.

The existing validation ranking is retained for this MVP:

\[
F=0.6A_{500}+0.4A_{5000}-0.001\log(1+d)-0.0001\log(1+t),
\]

where \(d\) is feature width and \(t\) is measured feature, training, and
inference runtime. The official test split remains outside the search process.

### 6. Quality-diversity archive and novelty

The archive derives a primary niche deterministically from primitive families:

```text
topology, graph_structure, geometry, stroke_dynamics,
spatial_relations, frequency_scale, compositional, hybrid, raw_pixels
```

Programs using multiple semantic families enter `hybrid`; programs without a
family marker enter `compositional`. Complexity is binned by AST node count as
`compact` (up to 5), `moderate` (6–10), or `deep` (over 10). The archive stores
all evaluated candidates and retains the highest existing validation ranking in
each `(niche, complexity)` cell. Proposer context reports occupied elites and
underexplored implemented niches.

Structural novelty is the normalized ordered-tree edit cost to the nearest
different archived AST. Node-operation changes and parameter changes have
explicit costs; subtree insertions and deletions cost their node count. Novelty
does not depend on natural-language explanations or embeddings. The score is
stored as evidence and does not replace the cell's quality ranking.

Transfer, sequence counterfactuals, and invariance search are isolated
post-search workflows described below. Pareto selection is still a future
option; the current archive retains one quality elite per descriptor cell.

### 7. Proposal and search lifecycle

```text
20 deterministic typed seed programs
          ↓
validate for selected track → evaluate on train/validation
          ↓
initialize MAP-Elites cells
          ↓
10 generations × 18 evaluated proposals (configurable)
          ↓
LLM batches of at most 4 ASTs → type/parameter/width/depth validation → deduplicate
          ↓
evaluate valid programs → calculate descriptor and novelty
          ↓
append candidate and response JSONL records → update cell elite
```

Ollama receives a JSON Schema built from the selected track's typed primitive
signatures. It constrains child types, operation-owned parameter names and
bounds, a maximum depth of 6, at most 127 nodes, the raw-pixel rule, and exact
proposal count. Vector-width rules still receive final static validation before
feature extraction. Invalid LLM responses or duplicate-only batches receive one
bounded repair attempt. Search archives can resume by loading their JSONL
records. Every proposal batch is numbered from the durable response archive, so
refill prompts and seeded sampling change across retries and process restarts.
The program proposer uses temperature 0.65, disables Qwen thinking with
`/no_think`, and derives each request seed from the archived batch number and
repair attempt. Each structured batch has a 6,000-token output cap and
600-second request timeout. The CLI permits up to 36 refill batches per
generation; smaller proposal batches reduce structured-output truncation.
Proposal text is limited to 25 words and 240
characters per field. V1's ordinary JSON generation remains greedy.
Provider errors are archived, and if a batch still fails validation or
contains only duplicates after repair, the engine records it and continues
with the next bounded refill batch. The prompt includes a compact typed-
expression signature for every archived AST in its strict duplicate ban list;
this avoids both recent-history gaps and repeated JSON field overhead. The raw
prompts and model responses are saved separately for audit.

### 8. Package layout

```text
src/bias_optimizer/
├── domain/
│   ├── bias.py                 # frozen V1 OperatorSpec/BiasSpec
│   ├── program.py              # V2 ProgramBiasSpec and canonical hash
│   └── program_search.py       # V2 durable search records
├── dsl/
│   ├── ast.py                  # immutable Expr tree
│   ├── types.py                # Image through Scalar type vocabulary
│   ├── primitives.py           # bounded atom signatures and interpreter
│   ├── validator.py            # static typing and track constraints
│   ├── compiler.py             # validated ProgramPipeline
│   └── seeds.py                # deterministic typed initial programs
├── novelty/
│   ├── descriptors.py          # niche and complexity descriptors
│   ├── tree_distance.py        # structural distance and novelty
│   └── map_elites.py           # V2 quality-diversity archive
├── llm/
│   └── program_proposer.py     # V2 prompt, parser, repair, response archive
├── ml/
│   └── evaluator.py            # frozen V1 evaluator + V2 adapter
├── data/
│   └── transfer.py             # checksum-aware handwriting transfer loaders
├── cache/
│   ├── feature_cache.py        # persistent whole-matrix cache
│   └── subexpression_cache.py  # bounded cross-candidate subtree LRU
├── invariance/
│   ├── transform.py            # bounded Image -> Image transform DSL
│   ├── proposer.py              # LLM transformation-hypothesis proposals
│   └── orbit.py                 # deterministic orbit-mean pooling
└── search/
    └── program_engine.py       # two-track LLM/MAP-Elites search loop

experiments/search_programs.py  # V2 command-line entry point
experiments/freeze_transfer_candidates.py
experiments/evaluate_pixel_augmentations.py
experiments/evaluate_transfer.py
experiments/evaluate_counterfactuals.py
experiments/search_invariances.py
experiments/profile_subexpression_cache.py
```

V1's operator compiler, search engine, finalist selection, final evaluation,
and report remain available. The experiment branches share the learner and data
protocol but keep candidate schemas and search archives separate.

### 9. Running V2

```sh
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_programs.py --track discovery --proposal-rounds 36
python experiments/search_programs.py --track augmentation --proposal-rounds 36
```

Defaults use 20 seed programs, 10 generations, and 18 evaluated proposals per
generation. Report the two tracks separately. A candidate program alone is not
evidence of a discovered inductive bias; transfer and mechanism-falsification
results bound that claim.

### 10. Frozen transfer evaluation

`freeze_transfer_candidates.py` ranks raw-free discovery records by MNIST
validation accuracy at 500 samples, then by the stored validation score and
candidate ID. It writes each complete candidate AST plus the discovery archive
SHA-256 and `test_set_accessed: false`. The transfer runner verifies this
manifest and archive before loading any destination test partition.

`evaluate_pixel_augmentations.py` compares raw pixels with depth-compatible
frozen finalist ASTs concatenated to pixels, plus a fixed spatial control. It
uses MNIST train/validation only. These post-search compositions are not new
LLM proposals and reuse the validation split that selected the finalists.

`data/transfer.py` validates the downloaded gzip/IDX headers, dimensions, sample
counts, label bounds, and checksums. EMNIST is downloaded from the NIST archive;
its labels use zero-based IDs and its image axes are transposed to the documented
orientation. KMNIST uses the CODH files when available and a checksum-verified
OpenML dataset mirror if the original file host cannot be reached. The fallback
builds a deterministic stratified 60k/10k split and the report labels that split
explicitly. Fashion-MNIST uses publisher files and their published MD5 values.

The transfer runner freezes feature programs but refits the same
StandardScaler/logistic-regression learner in each domain. It uses 500 and 5,000
training examples, three train seeds, and one stratified 2,000-example sample
from each source test split. The 2,000-example test sample is reported as such;
it is not presented as the full test-set score. Three human structural controls
are measured in each domain. This design answers whether MNIST-selected
representations carry over, while Fashion-MNIST is the negative-control task.

### 11. Sequence counterfactuals

`ProgramPipeline.transform_with_sequence_shuffle()` applies a deterministic
permutation to each path group after a selected sequence-valued primitive. It
keeps the multiset and path boundaries intact, then recomputes downstream AST
nodes. The runner fits on original MNIST features and measures predictions on
original and counterfactual views from a stratified sample of the official MNIST
test partition, after the candidate manifest has been verified. It records
accuracy change and prediction agreement over three seeds. This is direct
evidence about order dependence in the frozen program; a small or zero change
does not support an order-sensitive mechanism.

### 12. Invariance programs and orbit pooling

`TransformProgram` is a separate typed sequence of at most four `Image -> Image`
atoms. Its allow-list includes integer translations, small rotations, one-pixel
dilation and erosion, bounded shear, and seeded smooth elastic displacement.
All parameters are validated before execution; no generated code runs. The LLM
must provide a mechanism, prediction, and falsifier in `InvarianceSpec`.

`search_invariances.py` proposes transformations using only MNIST train and
validation data. For each one it measures transformed-view label accuracy,
prediction agreement, and changed-pixel fraction against a frozen discovery
representation. A predeclared validation rule selects candidates for
`OrbitPooledPipeline`, which averages the representation over the original view
and three seeded transformed views. The script cannot load the official test
split, so its result is validation evidence and must be followed by an
independent held-out evaluation before a strong invariance claim.

### 13. Cross-candidate subexpression cache

The in-memory LRU key is `(exact split key + row index, canonical subtree AST)`.
Only expensive typed operations (skeletonization, graph/path construction, and
ordered sequence derivations) are retained. A default 256 MiB limit bounds
memory. Evicted entries are recomputed; there is no disk persistence or effect on
feature values. The search CLI records hits, misses, evictions, occupancy, and
hit rate. `profile_subexpression_cache.py` compares the uncached and cached
feature matrices byte-for-byte and reports sample-specific elapsed time. The
profile is a local smoke measurement, not a universal speed claim.

### 14. Post-search commands

After both tracks complete, run the stages in this order:

```sh
python experiments/freeze_transfer_candidates.py
python experiments/evaluate_pixel_augmentations.py
python experiments/evaluate_transfer.py
python experiments/evaluate_counterfactuals.py
python experiments/search_invariances.py
python experiments/profile_subexpression_cache.py
python experiments/summarize_v2.py
```

The final report should cite the archive and manifest hashes, exact candidate
IDs, dataset checksums/splits, sample counts, metrics, and cache equality. It
must distinguish MNIST validation search results from post-freeze test results.

### 15. Measured V2 run (2026-10-09)

Both tracks completed 20 seed programs plus 10 × 18 generation records. The
discovery archive's best `cycle_angle_hist` candidate reached 0.7180 validation
accuracy at 500 training examples; the augmentation archive's best candidate
reached 0.6445. Each archive occupied four implemented niche families, with
`hybrid` dominant. Geometry, stroke-dynamics, frequency-scale, and compositional
niches were empty in both; raw pixels were allowed but absent from the
augmentation archive. The QD archive kept some descriptor-cell variation but
did not produce broad niche coverage.

Both tracks evaluated 200 candidates, but the augmentation run switched to
request batches of at most four after a large structured response was truncated.
Their top scores are descriptive and should not be interpreted as a controlled
comparison between the two search tracks.

Because the augmentation archive contained no raw-pixel program, a separate
validation-only control composed raw pixels with the depth-compatible
`angle_hist_12` finalist. Raw pixels scored 0.8170 / 0.8765 at 500 / 5,000
examples; raw pixels plus `angle_hist_12` scored 0.8525 / 0.9170, while the
fixed raw-plus-spatial control scored 0.8170 / 0.8785. These are post-search
compositions on the same validation split used for finalist selection, not new
LLM proposals or independent generalization estimates.

The five raw-free discovery finalists were frozen before transfer evaluation.
At 5,000 training examples, `spatial_cycle_hist` scored 0.8120 on a stratified
2,000-image EMNIST Digits test sample, 0.4993 on EMNIST Letters, 0.4538 on
KMNIST, and 0.7080 on Fashion-MNIST. The strong Fashion-MNIST result means the
run does not establish handwriting-specific transfer. All source splits,
checksums, and test counts are recorded in `reports/V2_PROGRAM_SYNTHESIS.md` and
the JSON transfer artifact.

Shuffling sequence values at the selected AST nodes produced zero accuracy
change and 1.000 prediction agreement for every frozen finalist. The measured
programs therefore provide no evidence for sequence-order dependence; the
counterfactual falsifies an order-sensitive explanation for these finalists.
The LLM-proposed two-pixel horizontal translation passed the predeclared
validation rule, but orbit-mean pooling reduced holdout accuracy by 0.0020.
Finally, the cache profile preserved exact features and measured a 3.14× speedup
on 16 programs × 96 images. That timing is a local smoke measurement, not a
general performance guarantee. The linked research report contains all scores
and claim boundaries.
