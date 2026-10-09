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

- [ ] **7. Implement operator registry**
  - [ ] Register only allowed operators:
    ```text
    raw_pixels
    topology
    spatial
    symmetry
    stroke_direction
    curvature
    direction_transition
    ```
  - [ ] Reject arbitrary code from LLM
  - [ ] Validate parameter bounds
  - [ ] Add registry tests

- [ ] **8. Implement `FeaturePipeline`**
  - [ ] Compose several `FeatureOperator`s
  - [ ] Concatenate output vectors
  - [ ] Guarantee finite numeric output
  - [ ] Expose `feature_dim`
  - [ ] Batch-transform images
  - [ ] Cache feature matrices by bias hash

- [ ] **9. Implement `BiasCompiler`**
  - [ ] Input:
    ```text
    BiasSpec
    ```
  - [ ] Output:
    ```text
    FeaturePipeline
    ```
  - [ ] Compile operator specs through registry
  - [ ] Reject invalid bias specifications cleanly
  - [ ] Ensure LLM never directly edits evaluator/classifier code

- [ ] **10. Implement `Evaluator`**
  - [ ] Input: `BiasSpec`
  - [ ] Compile representation
  - [ ] Extract/cache features
  - [ ] Train fixed logistic regression
  - [ ] Evaluate at \(n=500\)
  - [ ] Evaluate at \(n=5000\)
  - [ ] Return `Evaluation`
  - [ ] Include:
    ```text
    accuracy_500
    accuracy_5000
    feature_dim
    feature_runtime_ms
    inference_runtime_ms
    confusion_matrix
    ```
  - [ ] Define initial ranking score, e.g.
    \[
    F=
    0.6A_{500}+0.4A_{5000}
    -\lambda_d\log(1+d)
    -\lambda_t\log(1+t)
    \]

- [ ] **11. Create initial human-designed biases**
  - [ ] `B0 = raw_pixels`
  - [ ] `B1 = topology`
  - [ ] `B2 = topology + spatial`
  - [ ] `B3 = topology + curvature`
  - [ ] `B4 = topology + stroke_direction + curvature`
  - [ ] Evaluate all five
  - [ ] Confirm the whole pipeline works without any LLM

- [ ] **12. Define the LLM abstraction**
  - [ ] Create `LLMClient` protocol
    ```python
    class LLMClient(Protocol):
        def generate(self, prompt: str) -> str: ...
    ```
  - [ ] Implement one provider first
  - [ ] Keep provider-specific code isolated
  - [ ] Add mock LLM client for tests

- [ ] **13. Implement structured LLM output**
  - [ ] Require JSON only
  - [ ] Parse JSON → `BiasSpec`
  - [ ] Reject malformed output
  - [ ] Retry once on schema failure
  - [ ] Never execute LLM-generated Python
  - [ ] Record raw LLM response for reproducibility

- [ ] **14. Design proposer prompt**
  - [ ] Include research objective
  - [ ] Include allowed operators
  - [ ] State fixed classifier constraint
  - [ ] Include top-performing hypotheses
  - [ ] Include confusion pairs
  - [ ] Include ablation evidence when available
  - [ ] Require for every proposal:
    ```text
    hypothesis
    representation
    expected effect
    prediction
    falsification condition
    ```
  - [ ] Explicitly tell LLM:
    > Prefer conceptual changes over simply adding more features.

- [ ] **15. Implement `LLMProposer`**
  - [ ] Input: search history
  - [ ] Generate four candidate types:
    ```text
    exploitation
    failure-driven
    simplification
    exploration
    ```
  - [ ] Convert output to validated `BiasSpec`
  - [ ] Remove duplicate biases
  - [ ] Limit context to relevant history

- [ ] **16. Implement search records**
  - [ ] `SearchRecord`
    ```text
    generation
    BiasSpec
    Evaluation
    parent IDs
    prompt/model metadata
    ```
  - [ ] Persist every record to JSONL
  - [ ] Generate deterministic candidate hashes

- [ ] **17. Implement MVP `SearchEngine`**
  - [ ] Seed with five human biases
  - [ ] Evaluate seeds
  - [ ] Keep top 5
  - [ ] Ask LLM for 5 new candidates
  - [ ] Evaluate candidates
  - [ ] Merge + rank
  - [ ] Repeat for 5 generations
  - [ ] Target:
    \[
    30\text{–}40\text{ evaluations}
    \]

- [ ] **18. Add failure-driven feedback**
  - [ ] Extract major confusion pairs
  - [ ] Example:
    ```text
    3 ↔ 5
    4 ↔ 9
    ```
  - [ ] Feed these back to LLM
  - [ ] Ask for hypotheses specifically addressing those failures
  - [ ] Track whether proposed fixes actually improve those pairs

- [ ] **19. Add ablation for elite candidates**
  - [ ] For each top bias
    \[
    B=\{b_1,\dots,b_k\}
    \]
  - [ ] Evaluate
    \[
    B\setminus\{b_i\}
    \]
  - [ ] Compute
    \[
    \Delta_i=A(B)-A(B\setminus\{b_i\})
    \]
  - [ ] Return ablation evidence to LLM
  - [ ] Remove unsupported operators

- [ ] **20. Run the first serious experiment**
  - [ ] 10 generations
  - [ ] Approximately 5–8 new candidates/generation
  - [ ] Target:
    \[
    60\text{–}100\text{ candidates}
    \]
  - [ ] Search using validation data only
  - [ ] Track total LLM tokens
  - [ ] Track CPU evaluation time
  - [ ] Track total search wall-clock time

- [ ] **21. Select finalists**
  - [ ] Keep top 5 by low-data accuracy
  - [ ] Include one smallest representation
  - [ ] Include one fastest representation
  - [ ] Include original stroke-flow hypothesis even if it loses
  - [ ] Freeze search before touching test set

- [ ] **22. Final evaluation**
  - [ ] Evaluate finalists at:
    \[
    n\in\{250,500,1000,5000,60000\}
    \]
  - [ ] Run multiple seeds
    \[
    \{11,23,47\}
    \]
  - [ ] Report mean ± standard deviation
  - [ ] Evaluate test set exactly after candidate selection
  - [ ] Compare against raw pixels and HOG

- [ ] **23. Produce final plots**
  - [ ] Learning curve:
    \[
    x=\log N_{\text{train}},\quad y=\text{accuracy}
    \]
  - [ ] Bias evolution across generations
  - [ ] Feature dimension vs accuracy
  - [ ] Runtime vs accuracy
  - [ ] Confusion matrices for finalists
  - [ ] Ablation contribution plot

- [ ] **24. Answer the research question**
  - [ ] Did LLM-generated biases beat raw pixels in low-data regime?
  - [ ] Did they beat human-designed stroke-flow bias?
  - [ ] Did the LLM revise incorrect hypotheses based on evidence?
  - [ ] Which discovered assumptions survived ablation?
  - [ ] Did explicit stroke direction help?
  - [ ] Was curvature more useful than inferred pen order?
  - [ ] Did search discover simpler representations rather than merely larger ones?

- [ ] **25. Stop condition**
  - [ ] Stop after ~100 candidates unless new generations still improve materially
  - [ ] Stop if last 3 generations improve \(A_{500}\) by less than ~0.2%
  - [ ] Do not expand the project into unrestricted AutoML
  - [ ] Keep the central claim focused on:
    \[
    \boxed{\text{LLM-guided inductive-bias discovery}}
    \]
