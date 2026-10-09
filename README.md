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
python -m pip install -e '.[dev]'
```

Run the test suite with `python -m pytest`.

## Project layout

- `src/bias_optimizer/domain/` — immutable experiment and search data models.
- `src/bias_optimizer/features/` — approved feature operators and pipelines.
- `src/bias_optimizer/compiler/` — compile declarative biases into pipelines.
- `src/bias_optimizer/ml/` — fixed learner and deterministic evaluator.
- `src/bias_optimizer/llm/` — provider-neutral proposal boundary.
- `src/bias_optimizer/search/` — search controller and JSONL archive.
- `src/bias_optimizer/data/` — reproducible MNIST splits.
- `src/bias_optimizer/cache/` — feature cache interface.
- `experiments/` — baseline and search entry points.
- `tests/` — unit and integration tests.
- `results/`, `cache/` — local generated artifacts; contents are ignored.

This commit establishes the package and its boundaries. The experiment, feature
operators, provider integration, and search loop are later implementation steps
in `PLAN.md`.
