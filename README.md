# LLM as Bias Optimizer

An experiment in discovering useful image representations through LLM-proposed
hypotheses and deterministic local evaluation. The classifier and evaluator stay
under local control. The frozen V1 experiment selects from named `BiasSpec`
operators; the new V2 path proposes typed representation programs. Neither path
executes generated code or gives the LLM access to fitness assignment.

## Development setup

Requires Python 3.14.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots]'
```

Run the test suite with `python -m pytest`.

## Reproduce the experiment

The proposer uses the local Ollama model `qwen3.5:35b-mlx`. Start Ollama and
install that model if it is not already available, then run the search and
freeze its finalists before any test-set evaluation:

```sh
# If needed, install the local model first:
ollama pull qwen3.5:35b-mlx
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_mnist.py --generations 10 --candidates-per-generation 8
python experiments/select_finalists.py
python experiments/final_evaluation.py
python experiments/plot_results.py
```

Final evaluation verifies that the search archive still matches the frozen
selection checksum. Run it before extending the search. Generated JSON/JSONL,
feature-cache, and PNG files are written under `results/` and `cache/`; Git
ignores these local artifacts. The final analysis is in
[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md).

## V2 typed program search

V2 adds a typed expression DSL, static validation, a raw-pixel-free discovery
track, and a MAP-Elites archive. It keeps the V1 run as a separate baseline and
uses the same fixed learner and train/validation split.

```sh
export OLLAMA_MODEL=qwen3.5:35b-mlx
python experiments/search_programs.py --track discovery
python experiments/search_programs.py --track augmentation
```

The defaults evaluate 20 deterministic seed programs, then request 18 proposals
for each of 10 generations, using batches of at most four ASTs and up to 36
refill batches per generation. Discovery rejects raw pixels and caps the feature
width at 128; augmentation allows raw pixels and uses a 1,024-feature cap. Each
track has its own candidate and LLM-response JSONL archives. See
[`PLAN.md`](PLAN.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md) for the V2 scope
and post-search evaluation protocol.

After both search tracks finish, freeze the raw-free discovery finalists before
loading external test partitions. The follow-on commands evaluate transfer,
sequence counterfactuals, LLM-proposed invariances, orbit pooling, and the
bounded shared-subexpression cache:

```sh
python experiments/freeze_transfer_candidates.py
python experiments/evaluate_pixel_augmentations.py
python experiments/evaluate_transfer.py
python experiments/evaluate_counterfactuals.py
python experiments/search_invariances.py
python experiments/profile_subexpression_cache.py
python experiments/summarize_v2.py
```

Transfer uses EMNIST Digits/Letters and KMNIST as handwriting domains, with
Fashion-MNIST as a negative control. It reports the exact sampled test protocol
and source checksums. The completed V2 run and its limits are summarized in
[`reports/V2_PROGRAM_SYNTHESIS.md`](reports/V2_PROGRAM_SYNTHESIS.md); MAP-Elites
left several niches empty, sequence shuffles did not change finalist predictions,
and Fashion-MNIST transfer prevents a handwriting-specific claim.

## Project layout

- `src/bias_optimizer/domain/` — immutable experiment and search data models.
- `src/bias_optimizer/dsl/` — typed V2 representation AST, validator, compiler, and seed grammar.
- `src/bias_optimizer/novelty/` — structural novelty descriptors and MAP-Elites archive.
- `src/bias_optimizer/features/` — approved feature operators and pipelines.
- `src/bias_optimizer/compiler/` — compile declarative biases into pipelines.
- `src/bias_optimizer/ml/` — fixed learner and deterministic evaluator.
- `src/bias_optimizer/llm/` — provider-neutral proposal boundary.
- `src/bias_optimizer/search/` — V1 controller and V2 program-search engine.
- `src/bias_optimizer/data/` — reproducible MNIST splits and transfer loaders.
- `src/bias_optimizer/invariance/` — bounded transformation programs and orbit pooling.
- `src/bias_optimizer/cache/` — feature and cross-candidate subexpression caches.
- `experiments/` — baseline, search, finalist, evaluation, and plot entry points.
- `reports/` — tracked research conclusions.
- `tests/` — unit and integration tests.
- `results/`, `cache/` — local generated artifacts; contents are ignored.
