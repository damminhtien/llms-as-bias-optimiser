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
cross-candidate subtree caching is deferred until the initial V2 run is profiled.

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

Symmetry/invariance transformations, counterfactual experiments, cross-dataset
transfer, and Pareto selection are follow-on work; they are not claimed as
implemented by this first V2 slice.

## 7. Proposal and search lifecycle

```text
20 deterministic typed seed programs
          ↓
validate for selected track → evaluate on train/validation
          ↓
initialize MAP-Elites cells
          ↓
10 generations × 18 LLM proposals (configurable)
          ↓
parse JSON → type/parameter/width/depth validation → deduplicate AST
          ↓
evaluate valid programs → calculate descriptor and novelty
          ↓
append candidate and response JSONL records → update cell elite
```

Invalid LLM responses receive one bounded repair attempt. Invalid programs never
reach feature extraction. Search archives can resume by loading their JSONL
records. The raw prompts and model responses are saved separately for audit.

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
└── search/
    └── program_engine.py       # two-track LLM/MAP-Elites search loop

experiments/search_programs.py  # V2 command-line entry point
```

V1's operator compiler, search engine, finalist selection, final evaluation,
and report remain available. The experiment branches share the learner and data
protocol but keep candidate schemas and search archives separate.

## 9. Running V2

```sh
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_programs.py --track discovery
python experiments/search_programs.py --track augmentation
```

Defaults use 20 seed programs, 10 generations, and 18 proposals per generation.
The two tracks should be reported separately. No result should be described as a
discovered inductive bias until it has survived representative reruns and the
planned transfer and mechanism-falsification stages.
