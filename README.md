# LLM as Bias Optimizer

An experiment in discovering useful image representations through LLM-proposed
hypotheses and deterministic local evaluation. The classifier and evaluator stay
under local control; the LLM proposes `BiasSpec` values and never executes code
or assigns fitness.

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

## Project layout

- `src/bias_optimizer/domain/` — immutable experiment and search data models.
- `src/bias_optimizer/features/` — approved feature operators and pipelines.
- `src/bias_optimizer/compiler/` — compile declarative biases into pipelines.
- `src/bias_optimizer/ml/` — fixed learner and deterministic evaluator.
- `src/bias_optimizer/llm/` — provider-neutral proposal boundary.
- `src/bias_optimizer/search/` — search controller and JSONL archive.
- `src/bias_optimizer/data/` — reproducible MNIST splits.
- `src/bias_optimizer/cache/` — feature cache interface.
- `experiments/` — baseline, search, finalist, evaluation, and plot entry points.
- `reports/` — tracked research conclusions.
- `tests/` — unit and integration tests.
- `results/`, `cache/` — local generated artifacts; contents are ignored.
