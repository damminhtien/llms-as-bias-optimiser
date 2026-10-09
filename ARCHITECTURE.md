## 1. Design goals

The system searches over **inductive biases**, not model parameters.

The main invariant is:

\[
\boxed{\text{LLM proposes hypotheses; deterministic local code evaluates them.}}
\]

The LLM must never:

- access the test set;
- modify the evaluator;
- modify the downstream classifier;
- execute arbitrary generated code;
- directly assign fitness scores.

This keeps the experiment reproducible and makes performance changes attributable to the proposed representation.

---

# 2. System architecture

```text
                         ┌──────────────────────┐
                         │      LLM Provider    │
                         │ OpenAI / local model │
                         └──────────┬───────────┘
                                    │
                              structured JSON
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Proposer        │
                         │ evidence → BiasSpec  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Validator       │
                         │ schema + constraints │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     BiasCompiler     │
                         │ BiasSpec → Pipeline  │
                         └──────────┬───────────┘
                                    │
                                    ▼
        MNIST ───────────────► FeaturePipeline
                                    │
                             X ∈ R^(N × d)
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Fixed ML Learner   │
                         │ Logistic Regression  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Evaluator       │
                         │ accuracy / runtime   │
                         │ confusion / size     │
                         └──────────┬───────────┘
                                    │
                              Evaluation
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   SearchController   │
                         │ archive / selection  │
                         └──────────┬───────────┘
                                    │
                             summarized evidence
                                    │
                                    └──────────────► Proposer
```

The architecture has three distinct planes:

```text
Reasoning plane:
    LLM proposer

Execution plane:
    feature extraction
    model training
    evaluation

Control plane:
    search loop
    history
    selection
    caching
```

Do not mix them.

---

# 3. Runtime architecture

Use Python 3.14.

```text
Local machine
│
├── Python 3.14 process
│
├── MNIST dataset
├── feature extraction
├── sklearn training
├── evaluation
├── cache
├── search state
└── experiment artifacts
        │
        │ HTTPS or local IPC
        ▼
LLM provider
├── OpenAI API
or
└── local llama.cpp / Ollama
```

Only the proposer requires an LLM.

Everything else runs locally.

This gives a clean cost model:

\[
C_{\text{total}}
=
C_{\text{LLM}}
+
C_{\text{feature}}
+
C_{\text{training}}
+
C_{\text{evaluation}}.
\]

---

# 4. Core data model

Prefer immutable value objects.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BiasSpec:
    name: str
    hypothesis: str
    operators: tuple[OperatorSpec, ...]
    prediction: str
    falsification: str
```

A `BiasSpec` is the contract between the LLM and the deterministic system.

Example:

```python
BiasSpec(
    name="localized_curvature",
    hypothesis=(
        "Digit identity depends on topology and "
        "where curvature occurs vertically."
    ),
    operators=(
        OperatorSpec("topology", {}),
        OperatorSpec(
            "curvature",
            {"vertical_bins": 3},
        ),
    ),
    prediction="Should improve 3-vs-5 discrimination.",
    falsification="Reject if accuracy@500 improves by <0.3%.",
)
```

---

# 5. Feature abstraction

Use a small protocol.

```python
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


Image = NDArray[np.float32]
FeatureVector = NDArray[np.float32]


class FeatureOperator(Protocol):
    def transform(self, image: Image) -> FeatureVector:
        ...
```

Concrete operators:

```text
RawPixelsOperator
TopologyOperator
SpatialOperator
SymmetryOperator
CurvatureOperator
StrokeDirectionOperator
DirectionTransitionOperator
```

Do not let the LLM define new Python classes in the MVP.

The LLM only composes pre-approved operators.

---

# 6. Operator registry

Use an explicit registry rather than reflection.

```python
type OperatorFactory = callable


_OPERATOR_REGISTRY = {
    "raw_pixels": RawPixelsOperator,
    "topology": TopologyOperator,
    "spatial": SpatialOperator,
    "symmetry": SymmetryOperator,
    "curvature": CurvatureOperator,
    "stroke_direction": StrokeDirectionOperator,
    "direction_transition": DirectionTransitionOperator,
}
```

The registry is a security and reproducibility boundary.

Unknown operator:

```text
LLM output
    ↓
unknown operator
    ↓
validation error
    ↓
candidate rejected
```

No `eval()`, dynamic imports, or generated source execution.

---

# 7. Feature pipeline

```python
@dataclass(frozen=True, slots=True)
class FeaturePipeline:
    operators: tuple[FeatureOperator, ...]

    def transform(self, image: Image) -> FeatureVector:
        parts = [
            operator.transform(image)
            for operator in self.operators
        ]
        return np.concatenate(parts)
```

Batch transformation should be implemented separately:

```python
class BatchFeatureExtractor:
    def transform(
        self,
        pipeline: FeaturePipeline,
        images: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        ...
```

This separation makes caching and parallelization easier.

---

# 8. Bias compiler

The compiler converts declarative hypotheses into executable feature pipelines.

```python
class BiasCompiler:
    def compile(
        self,
        spec: BiasSpec,
    ) -> FeaturePipeline:
        operators = tuple(
            self._build_operator(operator_spec)
            for operator_spec in spec.operators
        )

        return FeaturePipeline(
            operators=operators,
        )

    def _build_operator(
        self,
        spec: OperatorSpec,
    ) -> FeatureOperator:
        ...
```

Responsibilities:

```text
BiasCompiler
├── resolve operator names
├── validate parameters
├── instantiate operators
└── build FeaturePipeline
```

It should not:

```text
train models
evaluate candidates
call LLMs
manage search state
```

---

# 9. Fixed learner

Wrap sklearn behind a very small interface.

```python
class Learner:
    def fit_predict(
        self,
        train_x: NDArray[np.float32],
        train_y: NDArray[np.int64],
        eval_x: NDArray[np.float32],
    ) -> NDArray[np.int64]:
        ...
```

Implementation:

```text
StandardScaler
    ↓
LogisticRegression
```

Keep all hyperparameters fixed in configuration.

For example:

```python
@dataclass(frozen=True, slots=True)
class LearnerConfig:
    regularization_c: float = 1.0
    max_iter: int = 1_000
```

The LLM never changes this object.

---

# 10. Evaluation model

```python
@dataclass(frozen=True, slots=True)
class Evaluation:
    accuracy_500: float
    accuracy_5000: float

    feature_dim: int
    feature_runtime_ms: float
    inference_runtime_ms: float

    confusion_matrix: NDArray[np.int64]
```

Evaluator:

```python
class Evaluator:
    def evaluate(
        self,
        bias: BiasSpec,
    ) -> Evaluation:
        ...
```

Internal flow:

```text
BiasSpec
   ↓
compile
   ↓
FeaturePipeline
   ↓
feature extraction
   ↓
cache lookup/write
   ↓
fixed learner
   ↓
metrics
   ↓
Evaluation
```

The evaluator must be deterministic for:

\[
(\text{BiasSpec}, \text{dataset split}, \text{seed})
\]

---

# 11. LLM abstraction

Use dependency inversion.

```python
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
    ) -> str:
        ...
```

Concrete implementations:

```text
OpenAIClient
LocalLlamaClient
MockLLMClient
```

The rest of the codebase must not depend on provider-specific APIs.

---

# 12. LLM proposer

```python
class BiasProposer:
    def __init__(
        self,
        client: LLMClient,
    ) -> None:
        self._client = client

    def propose(
        self,
        context: SearchContext,
        count: int,
    ) -> list[BiasSpec]:
        ...
```

The proposer performs:

```text
SearchContext
    ↓
prompt construction
    ↓
LLM request
    ↓
JSON response
    ↓
schema validation
    ↓
BiasSpec[]
```

No evaluation logic belongs here.

---

# 13. What the LLM sees

The LLM should receive compressed evidence.

Example:

```text
Current best:
topology + localized curvature

accuracy@500:
86.3%

accuracy@5000:
94.5%

Ablation:
topology      +3.1%
curvature     +1.8%
direction     +0.1%

Main errors:
3 ↔ 5
4 ↔ 9

Available operators:
topology
curvature
stroke_direction
direction_transition
symmetry
spatial

Generate:
1 exploitation proposal
1 failure-driven proposal
1 simplification proposal
1 exploratory proposal
```

This is enough information for reasoning.

Do not dump the entire experiment history into the prompt.

---

# 14. Search context

```python
@dataclass(frozen=True, slots=True)
class SearchContext:
    generation: int
    top_candidates: tuple[SearchRecord, ...]
    failure_cases: tuple[str, ...]
    ablations: tuple[AblationResult, ...]
```

This creates a stable boundary between search state and LLM prompting.

---

# 15. Search record

Every evaluated candidate must be persisted.

```python
@dataclass(frozen=True, slots=True)
class SearchRecord:
    candidate_id: str
    generation: int
    bias: BiasSpec
    evaluation: Evaluation
    parent_ids: tuple[str, ...]
```

Candidate ID:

\[
\text{id}
=
\operatorname{SHA256}
(
\text{canonical BiasSpec JSON}
)
\]

This gives reproducible caching.

---

# 16. Search controller

The controller owns the experiment.

```python
class SearchController:
    def __init__(
        self,
        proposer: BiasProposer,
        evaluator: Evaluator,
        archive: SearchArchive,
    ) -> None:
        ...

    def run(
        self,
        generations: int,
        candidates_per_generation: int,
    ) -> SearchArchive:
        ...
```

Main loop:

```python
for generation in range(generations):
    context = archive.build_context()

    candidates = proposer.propose(
        context=context,
        count=candidates_per_generation,
    )

    for bias in candidates:
        evaluation = evaluator.evaluate(bias)

        archive.add(
            SearchRecord(
                candidate_id=compute_id(bias),
                generation=generation,
                bias=bias,
                evaluation=evaluation,
                parent_ids=(),
            )
        )
```

Keep version 1 this simple.

Do not introduce distributed queues or async workers yet.

---

# 17. Search archive

```python
class SearchArchive:
    def add(
        self,
        record: SearchRecord,
    ) -> None:
        ...

    def top(
        self,
        k: int,
    ) -> list[SearchRecord]:
        ...

    def contains(
        self,
        candidate_id: str,
    ) -> bool:
        ...
```

Storage initially:

```text
JSONL + .npy cache
```

No database required.

Example:

```text
results/
├── search.jsonl
├── metadata.json
└── finalists.json

cache/
├── 011fa2.../
│   ├── train_500.npy
│   └── val.npy
└── ...
```

---

# 18. Package structure

I would use:

```text
llm-as-bias-optimizer/
│
├── pyproject.toml
├── README.md
│
├── src/
│   └── bias_optimizer/
│       │
│       ├── domain/
│       │   ├── bias.py
│       │   ├── evaluation.py
│       │   └── search.py
│       │
│       ├── features/
│       │   ├── base.py
│       │   ├── topology.py
│       │   ├── curvature.py
│       │   ├── stroke.py
│       │   ├── symmetry.py
│       │   └── spatial.py
│       │
│       ├── compiler/
│       │   └── bias_compiler.py
│       │
│       ├── ml/
│       │   ├── learner.py
│       │   └── evaluator.py
│       │
│       ├── llm/
│       │   ├── client.py
│       │   ├── proposer.py
│       │   └── prompts.py
│       │
│       ├── search/
│       │   ├── controller.py
│       │   └── archive.py
│       │
│       ├── data/
│       │   └── mnist.py
│       │
│       └── cache/
│           └── feature_cache.py
│
├── experiments/
│   ├── baselines.py
│   └── search_mnist.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── results/
```

This is enough separation without turning the project into an enterprise framework.

---

# 19. Dependency direction

Keep dependencies one-way.

```text
domain
  ↑
features
  ↑
compiler

domain
  ↑
ml

domain
  ↑
llm

domain
  ↑
search
```

More concretely:

```text
SearchController
    ├── BiasProposer
    ├── Evaluator
    └── SearchArchive

BiasProposer
    └── LLMClient

Evaluator
    ├── BiasCompiler
    ├── FeatureCache
    ├── Learner
    └── Dataset

BiasCompiler
    └── FeatureOperator registry
```

Avoid circular dependencies.

---

# 20. Configuration

Use immutable configuration objects.

```python
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int
    train_sizes: tuple[int, ...]
    validation_size: int
    generations: int
    candidates_per_generation: int
```

Example:

```python
ExperimentConfig(
    seed=42,
    train_sizes=(500, 5_000),
    validation_size=2_000,
    generations=5,
    candidates_per_generation=5,
)
```

Do not scatter magic numbers through the code.

---

# 21. Failure handling

Treat every candidate as untrusted input.

Candidate lifecycle:

```text
LLM output
   ↓
JSON parse
   ↓
schema validation
   ↓
semantic validation
   ↓
compile
   ↓
feature extraction
   ↓
finite-value validation
   ↓
evaluation
```

Possible states:

```text
VALID
INVALID_SCHEMA
INVALID_OPERATOR
INVALID_PARAMETER
COMPILE_FAILED
FEATURE_FAILED
EVALUATION_FAILED
```

A bad candidate must not terminate the search.

---

# 22. Testing strategy

Minimum tests:

### Unit

```text
TopologyOperator
CurvatureOperator
StrokeDirectionOperator
BiasCompiler
BiasSpec validation
candidate hashing
```

### Integration

```text
BiasSpec
→ compile
→ feature extraction
→ logistic regression
→ Evaluation
```

### LLM contract

Use `MockLLMClient`.

```python
def test_proposer_parses_valid_candidate() -> None:
    ...
```

Do not use a real API in CI tests.

---

# 23. MVP execution flow

Start with five human-defined candidates:

```text
raw pixels
topology
topology + spatial
topology + curvature
topology + direction + curvature
```

Then:

```text
evaluate seed candidates
        ↓
build SearchContext
        ↓
LLM proposes 5
        ↓
validate
        ↓
evaluate locally
        ↓
archive
        ↓
select best evidence
        ↓
repeat
```

For the first implementation:

\[
5\text{ generations}
\times
5\text{ candidates}
+
5\text{ seeds}
=
30
\]

evaluations.

That is enough to validate the architecture.

---

# 24. Non-negotiable engineering boundaries

I would enforce these explicitly.

```text
LLM
    can:
        propose BiasSpec
        reason from experiment summaries

    cannot:
        read MNIST directly
        inspect test labels
        train models
        modify learner config
        modify evaluator
        execute Python
        define fitness
```

And:

```text
Evaluator
    can:
        compile biases
        extract features
        train fixed learner
        compute metrics

    cannot:
        call LLM
        mutate search strategy
```

This separation makes the experiment auditable.

---

# 25. Final minimal class graph

```text
                    ┌─────────────┐
                    │  LLMClient  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ BiasProposer│
                    └──────┬──────┘
                           │
                        BiasSpec
                           │
             ┌─────────────▼─────────────┐
             │       BiasCompiler        │
             └─────────────┬─────────────┘
                           │
                    FeaturePipeline
                           │
             ┌─────────────▼─────────────┐
             │        Evaluator          │
             │                           │
             │ Dataset                   │
             │ FeatureCache              │
             │ Learner                   │
             └─────────────┬─────────────┘
                           │
                      Evaluation
                           │
             ┌─────────────▼─────────────┐
             │     SearchController      │
             └─────────────┬─────────────┘
                           │
                    SearchArchive
```

For version 1, I would stop here.

The core architecture should remain approximately:

\[
\boxed{
\texttt{BiasSpec}
\rightarrow
\texttt{FeaturePipeline}
\rightarrow
\texttt{Evaluator}
\rightarrow
\texttt{Evidence}
\rightarrow
\texttt{LLM}
}
\]

If that loop is clean, deterministic, cached, and testable, the research layer can evolve without forcing a redesign of the system.