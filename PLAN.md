# PLAN — V2 Representation Program Synthesis

## Research direction

Move the project from selecting and combining a fixed menu of seven feature
operators to discovering typed representation programs:

\[
\text{operator selection}
\rightarrow
\text{representation program synthesis}
\rightarrow
\text{inductive-bias discovery}
\]

The V1 MNIST experiment is frozen as the comparison baseline. Its `OperatorSpec`,
seven-operator allow-list, evaluator protocol, archived results, and final report
remain available on the V1 path. V2 adds a separate typed program path so the
baseline can still be reproduced and compared without mixing its search history.

## V2 scope and status

- [x] Replace V2 candidate representations with immutable typed `Expr` trees.
- [x] Add static type, primitive-parameter, AST-depth, node-count, and feature-dimension validation.
- [x] Compile only allow-listed DSL nodes; never execute generated Python.
- [x] Add raw-pixel augmentation and a separate discovery track that rejects `flatten_pixels` and caps features at 128 by default.
- [x] Add deterministic seed-program generation, LLM program proposals, bounded repair, and an independent evaluator/archive path.
- [x] Constrain Ollama proposals with track-specific typed AST schemas, operation-owned parameter bounds, depth/node limits, and exact proposal counts.
- [x] Number refill prompts and sampling seeds from the durable response archive; continue after failed or duplicate-only batches.
- [x] Add structural AST novelty and a MAP-Elites archive indexed by primitive family and complexity.
- [x] Persist V2 candidates and raw proposal exchanges in separate JSONL files.
- [x] Add `experiments/search_programs.py` as the V2 entry point.
- [x] Add a checksum-verified freeze manifest for raw-free MNIST-validation finalists.
- [x] Add validation-only raw-pixel baseline and raw-pixel-plus-frozen-program controls.
- [x] Add reproducible transfer loaders for EMNIST Digits/Letters, KMNIST, and Fashion-MNIST.
- [x] Add a bounded sequence-shuffle counterfactual over typed sequence values.
- [x] Add a typed image-transformation DSL, LLM proposal boundary, and orbit-mean pooling.
- [x] Add a 256 MiB LRU cache for reusable AST subexpressions across candidates.
- [x] Add `experiments/summarize_v2.py` to rebuild a provenance-linked V2 report from local artifacts.
- [x] Run the planned 20-seed + 10-generation search on both tracks and report results.
- [x] Freeze five finalists from the raw-free discovery track before evaluating transfer-test samples.
- [x] Run sequence-shuffle counterfactuals on the frozen MNIST finalists and report the null order-dependence result.
- [x] Run LLM-guided invariance search on train/validation data and evaluate orbit pooling without loading MNIST test data.
- [x] Profile cross-candidate caching against uncached transforms, check exact feature equality, and report measured hit rate, memory, and speedup.
- [x] Write the V2 research report, separating validation search, held-out transfer, mechanism falsification, and smoke/profile evidence.

## V2 run results

The 2026-10-09 run completed 200 candidates per track (20 seeds plus 10 × 18
generation records). Search details, archived proposal counts, finalist IDs,
source checksums, and all reported metrics are in
[`reports/V2_PROGRAM_SYNTHESIS.md`](reports/V2_PROGRAM_SYNTHESIS.md). The raw-free
track's top program, `cycle_angle_hist`, scored 0.7180 validation accuracy at
500 examples; the augmentation track's top candidate scored 0.6445. Both
quality-diversity archives occupied only four of their implemented niches, with
`hybrid` dominant. Geometry, stroke-dynamics, frequency-scale, and compositional
niches remained empty; raw-pixel candidates also remained absent in the
augmentation archive. MAP-Elites therefore preserved some cell diversity but
did not achieve broad niche coverage in this run.

Both tracks evaluated the same 200-candidate budget, but the augmentation run
switched to batches of at most four after a large structured response was
truncated. The per-track score maxima are therefore descriptive and should not
be treated as a controlled comparison between search tracks.

The frozen transfer finalist `spatial_cycle_hist` reached 0.8120 on the
2,000-image EMNIST Digits sample, 0.4993 on EMNIST Letters, 0.4538 on KMNIST,
and 0.7080 on Fashion-MNIST with 5,000 training examples. Its strong Fashion
score means these results do not establish handwriting-specific transfer.
Every tested sequence-shuffle intervention left predictions unchanged
(accuracy delta 0, agreement 1); the finalists provide no evidence that
sequence order matters. A two-pixel horizontal translation passed the
validation invariance rule, while orbit-mean pooling reduced the held-out
validation accuracy by 0.0020. The cache profile preserved exact features and
measured a 3.14× speedup on a local 16-program, 96-image sample; this is a smoke
measurement, not a general performance guarantee.

The V2 augmentation archive permits raw pixels but contained no LLM candidate
using that primitive. A separate validation-only control scored raw pixels at
0.8170 / 0.8765 for 500 / 5,000 examples. Concatenating the depth-compatible
`angle_hist_12` finalist with pixels scored 0.8525 / 0.9170 (+0.0355 / +0.0405);
a fixed raw-plus-spatial control scored 0.8170 / 0.8785. These post-search
compositions reuse the finalist-selection validation split, so the gains are
exploratory rather than an independent estimate.

## V2 search contract

Each proposal contains:

```text
name
hypothesis
mechanism
program: typed expression tree
prediction
falsification
```

The DSL starts with atoms such as image thresholding, skeletonization, graph and
path construction, sequence statistics, normalization, spatial splits, vector
composition, ratios, and counts. An LLM can compose these atoms into new
representations. Candidate validation is static and bounded before feature
extraction.

The two search tracks are isolated:

| Track | Pixel primitive | Default dimension cap | Purpose |
| --- | --- | ---: | --- |
| Augmentation | Allowed | 1,024 | Compare structural additions to raw pixels |
| Discovery | Rejected anywhere in the AST | 128 | Search compact structural representations |

Programs have operation depth at most 6 and at most 127 AST nodes by default.
The image input leaf is depth zero. Discovery programs are scored with the
existing fixed learner on validation accuracy at 500 and 5,000 examples, with
the existing feature-size and runtime penalties. No test data is available to
the search engine.

The archive keeps the best candidate per `(primary primitive family, complexity
bin)` cell. The deterministic primary-family descriptor uses AST primitive
families; multi-family programs enter the `hybrid` niche. Tree novelty compares
program structure and parameters, not explanations or LLM embeddings.

## Post-search evaluation protocol

`experiments/freeze_transfer_candidates.py` selects five candidates using only
MNIST validation accuracy at 500 training examples, with the recorded ranking
score and candidate hash as deterministic tie-breakers. The manifest stores the
full ASTs and SHA-256 of the discovery archive before any official test partition
is loaded. Transfer evaluation verifies both before loading another dataset.

`experiments/evaluate_pixel_augmentations.py` compares raw pixels with the
depth-compatible frozen finalist concatenated to pixels, plus a fixed spatial
control. It uses MNIST train/validation only; these post-search compositions
are not new LLM proposals and share the finalist-selection validation split.

The reusable loader validates IDX magic values, image dimensions, row counts,
label ranges, and source checksums. It supports NIST EMNIST Digits and Letters,
the CODH KMNIST files (with a checksummed OpenML mirror fallback), and the
Fashion-MNIST publisher files. EMNIST labels are mapped to zero-based classes;
the published EMNIST axis orientation is corrected by transposing image axes.
Transfer uses 500 and 5,000 training examples, three fixed train seeds, and a
seeded stratified sample of 2,000 held-out test images per domain. The learner is
refit per domain; the discovered representation AST is frozen. Fashion-MNIST is
the negative-control domain. Results identify the exact source split and
checksum, so the OpenML KMNIST fallback is never described as the publisher's
official test partition.

`experiments/evaluate_counterfactuals.py` fits the frozen candidate on 5,000
MNIST training images, then shuffles values within each path sequence at a
selected sequence-valued AST node on a stratified official-test sample. This
preserves each sequence's multiset while destroying order; downstream features
are recomputed. The report includes accuracy change and prediction agreement
for three deterministic shuffles. It tests sequence mechanisms that cannot be
identified by feature ablation alone.

The invariance DSL composes up to four bounded `translate`, `rotate`, `dilate`,
`erode`, `shear`, and `elastic` image primitives. LLM proposals state a label-
preservation hypothesis and falsifier. `experiments/search_invariances.py`
measures label accuracy, model agreement, and image change on the MNIST
train/validation protocol; it has no test-set loader. A candidate that passes a
predeclared validation rule can be converted to an `OrbitPooledPipeline`, which
averages the frozen representation over identity and transformed views.

The shared subexpression cache is memory-only, scoped by an exact split key,
sample index, and canonical AST JSON. It keeps expensive skeleton, graph, path,
and sequence intermediates under a configurable byte bound (256 MiB by default)
and exposes hits, misses, evictions, and occupancy. The standalone profile
compares exact feature matrices and local wall time against the uncached path.

## V2 MVP run

```sh
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_programs.py --track discovery \
  --generations 10 --candidates-per-generation 18 --seed-count 20 --proposal-rounds 36
python experiments/search_programs.py --track augmentation \
  --generations 10 --candidates-per-generation 18 --seed-count 20 --proposal-rounds 36
```

Each track writes an independent candidate archive and LLM-response archive.
LLM calls propose at most four ASTs at a time; up to 36 refill batches per
generation are allowed, which reduces response truncation and makes the search
resumable from its JSONL candidate archive. Report tracks separately; do not
pool them into one leaderboard.

## V1 baseline plan (frozen historical record)

The completed V1 checklist below records the original operator-selection
experiment and is retained for reproducibility. It does not describe the V2
program-synthesis search.

---

## TODO — `llm-as-bias-optimizer`

- [x] **0. Bootstrap repository**
  - [x] Create repo `llm-as-bias-optimizer`
  - [x] Use Python 3.14
  - [x] Create `pyproject.toml`
  - [x] Add dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `networkx`, `pydantic`, `pytest`
  - [x] Create initial structure:
    ```text
    src/bias_optimizer/
        domain/{bias,evaluation,search}.py
        features/base.py
        compiler/bias_compiler.py
        ml/{learner,evaluator}.py
        llm/{client,proposer,prompts}.py
        search/{controller,archive}.py
        data/mnist.py
        cache/feature_cache.py
    tests/{unit,integration}/
    experiments/
    results/
    cache/
    ```

- [x] **1. Build deterministic MNIST experiment**
  - [x] Load MNIST
  - [x] Create fixed train / validation / test splits
  - [x] Fix random seed
  - [x] Add subset sampling for:
    \[
    n\in\{500,5000,60000\}
    \]
  - [x] Ensure test set is inaccessible during search
  - [x] Write dataset/split reproducibility tests

- [x] **2. Implement fixed downstream learner**
  - [x] Use `StandardScaler`
  - [x] Use multinomial logistic regression
  - [x] Fix classifier hyperparameters
  - [x] Create `train_and_predict(features, labels)`
  - [x] Measure:
    - accuracy
    - confusion matrix
    - training time
    - inference time
  - [x] Verify same features + same seed ⇒ same result

- [x] **3. Implement baseline representations**
  - [x] `RawPixelsOperator`
  - [x] Simple downsampled-pixel baseline
  - [x] HOG / gradient baseline
  - [x] Record baseline accuracy at \(n=500\) and \(n=5000\)
  - [x] Save baseline results before introducing the LLM

- [x] **4. Implement the first structural operators**
  - [x] Skeletonization utility
  - [x] `TopologyOperator`
    - [x] connected components
    - [x] holes
    - [x] endpoints
    - [x] junctions
  - [x] `SpatialOperator`
    - [x] top / middle / bottom regions
  - [x] `SymmetryOperator`
  - [x] Unit-test each operator independently

- [x] **5. Implement handwriting-dynamics operators**
  - [x] Convert skeleton into graph \(G=(V,E)\)
  - [x] Detect endpoints and junctions
  - [x] Extract graph paths
  - [x] Implement local stroke direction
    \[
    \theta_t=\operatorname{atan2}(\Delta y,\Delta x)
    \]
  - [x] Quantize directions into 8 bins
  - [x] `StrokeDirectionOperator`
  - [x] Implement curvature
    \[
    \Delta\theta_t=\theta_{t+1}-\theta_t
    \]
  - [x] `CurvatureOperator`
  - [x] Implement direction-transition matrix
  - [x] Resolve optional trajectory inference: use orientation-neutral path features instead of assigning pen order
  - [x] Test on manually selected digits `0, 1, 6, 8, 9`

- [x] **6. Define the bias domain model**
  - [x] Implement `OperatorSpec`
  - [x] Implement `BiasSpec`
  - [x] Fields:
    ```text
    name
    hypothesis
    operators
    prediction
    falsification
    ```
  - [x] Make specs immutable where practical
  - [x] Add JSON serialization/deserialization
  - [x] Validate identifier syntax and JSON-safe parameter values (operator allow-list and bounds are step 7)

- [x] **7. Implement operator registry**
  - [x] Register only allowed operators:
    ```text
    raw_pixels
    topology
    spatial
    symmetry
    stroke_direction
    curvature
    direction_transition
    ```
  - [x] Reject arbitrary code from LLM
  - [x] Validate parameter bounds
  - [x] Add registry tests

- [x] **8. Implement `FeaturePipeline`**
  - [x] Compose several `FeatureOperator`s
  - [x] Concatenate output vectors
  - [x] Guarantee finite numeric output
  - [x] Expose `feature_dim`
  - [x] Batch-transform images
  - [x] Cache feature matrices by bias hash

- [x] **9. Implement `BiasCompiler`**
  - [x] Input:
    ```text
    BiasSpec
    ```
  - [x] Output:
    ```text
    FeaturePipeline
    ```
  - [x] Compile operator specs through registry
  - [x] Reject invalid bias specifications cleanly
  - [x] Ensure LLM never directly edits evaluator/classifier code

- [x] **10. Implement `Evaluator`**
  - [x] Input: `BiasSpec`
  - [x] Compile representation
  - [x] Extract/cache features
  - [x] Train fixed logistic regression
  - [x] Evaluate at \(n=500\)
  - [x] Evaluate at \(n=5000\)
  - [x] Return `Evaluation`
  - [x] Include:
    ```text
    accuracy_500
    accuracy_5000
    feature_dim
    feature_runtime_ms
    inference_runtime_ms
    confusion_matrix
    ```
  - [x] Define initial ranking score with \(\lambda_d=0.001\), \(\lambda_t=0.0001\), and \(t\) as total measured runtime in milliseconds:
    \[
    F=
    0.6A_{500}+0.4A_{5000}
    -\lambda_d\log(1+d)
    -\lambda_t\log(1+t)
    \]

- [x] **11. Create initial human-designed biases**
  - [x] `B0 = raw_pixels`
  - [x] `B1 = topology`
  - [x] `B2 = topology + spatial`
  - [x] `B3 = topology + curvature`
  - [x] `B4 = topology + stroke_direction + curvature`
  - [x] Evaluate all five
  - [x] Confirm the whole pipeline works without any LLM

- [x] **12. Define the LLM abstraction**
  - [x] Create `LLMClient` protocol
    ```python
    class LLMClient(Protocol):
        def generate(self, prompt: str) -> str: ...
    ```
  - [x] Implement local Ollama provider using `qwen3.5:35b-mlx`
  - [x] Keep provider-specific code isolated
  - [x] Add mock LLM client for tests

- [x] **13. Implement structured LLM output**
  - [x] Require JSON only
  - [x] Parse JSON → `BiasSpec`
  - [x] Reject malformed output
  - [x] Retry once on schema failure
  - [x] Never execute LLM-generated Python
  - [x] Record raw LLM response for reproducibility

- [x] **14. Design proposer prompt**
  - [x] Include research objective
  - [x] Include allowed operators
  - [x] State fixed classifier constraint
  - [x] Include top-performing hypotheses
  - [x] Include confusion pairs
  - [x] Include ablation evidence when available
  - [x] Require for every proposal:
    ```text
    hypothesis
    representation
    expected effect
    prediction
    falsification condition
    ```
  - [x] Explicitly tell LLM:
    > Prefer conceptual changes over simply adding more features.

- [x] **15. Implement `LLMProposer`**
  - [x] Input: bounded search-history evidence
  - [x] Generate four candidate types:
    ```text
    exploitation
    failure-driven
    simplification
    exploration
    ```
  - [x] Convert output to validated `BiasSpec`
  - [x] Remove duplicate representations within the batch and against history
  - [x] Limit context to relevant history

- [x] **16. Implement search records**
  - [x] `SearchRecord`
    ```text
    generation
    BiasSpec
    Evaluation
    parent IDs
    prompt/model metadata
    ```
  - [x] Persist every record to JSONL
  - [x] Generate deterministic candidate hashes

- [x] **17. Implement MVP `SearchEngine`**
  - [x] Seed with five human biases
  - [x] Evaluate seeds
  - [x] Keep top 5
  - [x] Ask LLM for 5 new candidates per generation, with bounded refill after deduplication
  - [x] Evaluate candidates
  - [x] Merge + rank
  - [x] Repeat for 5 generations
  - [x] Target 30–40 evaluations; the Qwen run archived 26 unique records after duplicate removal
    \[
    30\text{–}40\text{ evaluations}
    \]

- [x] **18. Add failure-driven feedback**
  - [x] Extract major confusion pairs
  - [x] Use observed directed pairs such as 4 → 9 and 5 → 3
  - [x] Feed these back to LLM
  - [x] Ask for hypotheses specifically addressing those failures
  - [x] Track whether proposed fixes actually improve those pairs

- [x] **19. Add ablation for elite candidates**
  - [x] For each of the top five biases, evaluate distinct one-operator removals
    \[
    B=\{b_1,\dots,b_k\}
    \]
  - [x] Evaluate
    \[
    B\setminus\{b_i\}
    \]
  - [x] Compute and plot
    \[
    \Delta_i=A(B)-A(B\setminus\{b_i\})
    \]
  - [x] Return ablation evidence to LLM
  - [x] Promote an ablated representation when removing its operator improves accuracy at 500 samples; the smoke run found no unsupported operators among the seed elites

- [x] **20. Run the first serious experiment**
  - [x] 10 generations
  - [x] Approximately 5–8 new candidates/generation; 62 candidate records were archived after duplicate removal
  - [x] Target 60–100 candidates
    \[
    60\text{–}100\text{ candidates}
    \]
  - [x] Search using validation data only
  - [x] Track total LLM tokens: 88,414
  - [x] Track CPU evaluation time: 119.5 seconds
  - [x] Track total search wall-clock time: 545.7 seconds

- [x] **21. Select finalists**
  - [x] Keep top 5 by validation accuracy at 500 training examples
  - [x] Include one smallest representation
  - [x] Include one fastest representation
  - [x] Include original stroke-flow hypothesis even if it loses
  - [x] Freeze search before touching test set; the frozen artifact records the search archive SHA-256 and `test_set_accessed: false`
  - [x] The frozen set contains 8 unique finalists; no test data was loaded during selection

- [x] **22. Final evaluation**
  - [x] Evaluate finalists at:
    \[
    n\in\{250,500,1000,5000,60000\}
    \]
  - [x] Run multiple seeds
    \[
    \{11,23,47\}
    \]
  - [x] Report mean ± standard deviation
  - [x] Evaluate the official test set only after verifying the frozen finalist and archive checksums
  - [x] Compare against raw pixels and HOG
  - [x] Evaluate 8 finalists plus 2 baselines; at 500 examples, the best Qwen bias scored 86.41%, raw pixels 84.62%, and HOG 93.55% test accuracy

- [x] **23. Produce final plots**
  - [x] Learning curve:
    \[
    x=\log N_{\text{train}},\quad y=\text{accuracy}
    \]
  - [x] Bias evolution across generations, using search validation results only
  - [x] Feature dimension vs accuracy
  - [x] Runtime vs accuracy
  - [x] Confusion matrices for finalists, aggregated over the three 500-example seeds
  - [x] Ablation contribution plot
  - [x] Save all six PNGs under `results/figures/`; generation validates the frozen archive checksum

- [x] **24. Answer the research question**
  - [x] Best Qwen representation beat raw pixels by 1.79 percentage points at 500 examples; HOG remained stronger by 7.14 points
  - [x] Best Qwen representation beat the human stroke-flow seed by 24.30 points at 500 examples
  - [x] Summarize mixed failure-feedback results: 53 of 95 targeted confusion counts improved
  - [x] All 20 archived operator-removal ablations reduced validation accuracy; report contribution examples
  - [x] Explicit stroke direction helped when combined with raw pixels; removing it cost 2.45 validation points for the best candidate
  - [x] State that pen order was not inferred, so curvature-versus-pen-order was not tested
  - [x] The best bias was only 8 features larger than raw pixels; smaller structural-only candidates performed poorly
  - [x] Record full findings, limits, protocol, and reproducibility in `reports/FINAL_REPORT.md`

- [x] **25. Stop condition**
  - [x] Stop at 62 candidates because later generations did not improve materially
  - [x] Best-so-far \(A_{500}\) did not improve after generation 5; generations 8–10 added 0.00 percentage points
  - [x] Keep the project bounded; do not expand into unrestricted AutoML
  - [x] Keep the central claim focused on:
    \[
    \boxed{\text{LLM-guided inductive-bias discovery}}
    \]
