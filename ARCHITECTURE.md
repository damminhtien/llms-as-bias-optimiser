# Architecture — V2 Representation Program Synthesis

## 1. Research boundary

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

## 2. V2 data flow

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

## 3. Program domain and JSON contract

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

## 4. Typed primitive language

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

## 5. Independent search tracks

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

## 6. Quality-diversity archive and novelty

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

## 7. Proposal and search lifecycle

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

## 8. Package layout

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

## 9. Running V2

```sh
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_programs.py --track discovery --proposal-rounds 36
python experiments/search_programs.py --track augmentation --proposal-rounds 36
```

Defaults use 20 seed programs, 10 generations, and 18 evaluated proposals per
generation. Report the two tracks separately. A candidate program alone is not
evidence of a discovered inductive bias; transfer and mechanism-falsification
results bound that claim.

## 10. Frozen transfer evaluation

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

## 11. Sequence counterfactuals

`ProgramPipeline.transform_with_sequence_shuffle()` applies a deterministic
permutation to each path group after a selected sequence-valued primitive. It
keeps the multiset and path boundaries intact, then recomputes downstream AST
nodes. The runner fits on original MNIST features and measures predictions on
original and counterfactual views from a stratified sample of the official MNIST
test partition, after the candidate manifest has been verified. It records
accuracy change and prediction agreement over three seeds. This is direct
evidence about order dependence in the frozen program; a small or zero change
does not support an order-sensitive mechanism.

## 12. Invariance programs and orbit pooling

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

## 13. Cross-candidate subexpression cache

The in-memory LRU key is `(exact split key + row index, canonical subtree AST)`.
Only expensive typed operations (skeletonization, graph/path construction, and
ordered sequence derivations) are retained. A default 256 MiB limit bounds
memory. Evicted entries are recomputed; there is no disk persistence or effect on
feature values. The search CLI records hits, misses, evictions, occupancy, and
hit rate. `profile_subexpression_cache.py` compares the uncached and cached
feature matrices byte-for-byte and reports sample-specific elapsed time. The
profile is a local smoke measurement, not a universal speed claim.

## 14. Post-search commands

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

## 15. Measured V2 run (2026-10-09)

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
