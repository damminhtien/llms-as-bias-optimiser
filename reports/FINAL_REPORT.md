# Final report: LLM-guided inductive-bias discovery on MNIST

## Finding

The local Qwen search found a representation that beat the raw-pixel control
under the fixed logistic-regression learner. The best finalist combines raw
pixels with explicit stroke-direction features. At 500 training examples it
reached **86.41% ± 0.39%** test accuracy, compared with **84.62% ± 0.53%** for
raw pixels, a gain of 1.79 percentage points. The HOG control was stronger at
**93.55% ± 0.45%**.

This supports a narrow conclusion: in this MNIST setup, Qwen proposed a small
useful addition to raw pixels, but it did not beat the established HOG
representation. The result does not establish that LLM-guided biases improve
other datasets or classifiers.

## Protocol

- Search used OpenML `mnist_784` version 1, with a fixed 2,000-example
  validation split from the 60,000 official training examples. The official
  10,000-example test partition was unavailable to search.
- The local Ollama model was `qwen3.5:35b-mlx`. Ten generations produced 62
  unique candidate records, including the five human seeds, plus 20 ablation
  records. The run made 30 accepted LLM attempts and recorded 88,414 tokens.
- Final selection was frozen in `results/finalists.json` before final data was
  loaded. The artifact records the search archive SHA-256. Final evaluation
  verified that hash and evaluated eight finalists plus raw-pixel and HOG
  controls at training sizes 250, 500, 1,000, 5,000, and 60,000, with seeds
  11, 23, and 47.
- The learner remained `StandardScaler` plus multinomial logistic regression
  (`C=1`, `lbfgs`, `max_iter=1000`). HOG used 9 orientations, 4×4-pixel cells,
  and 2×2-cell blocks (1,296 features).
- Reported standard deviations are sample standard deviations across the three
  runs. At 60,000 examples all seeds use the same complete training partition,
  so the observed standard deviation is zero; it does not measure variability
  across different train sets.

## Final test accuracy

Values are mean ± standard deviation across seeds. The report contains all
eight finalists and both controls; this table focuses on the strongest Qwen
candidate, the second-ranked Qwen candidate, the human stroke-flow seed, and
the two controls.

| Representation | Features | 250 | 500 | 1,000 | 5,000 | 60,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen: raw pixels + stroke direction | 792 | 83.01% ± 1.16% | 86.41% ± 0.39% | 88.90% ± 0.48% | 91.37% ± 0.35% | 93.70% ± 0.00% |
| Qwen: raw pixels + topology | 788 | 82.90% ± 1.04% | 86.41% ± 0.49% | 88.75% ± 0.60% | 90.99% ± 0.64% | 93.40% ± 0.00% |
| Raw pixels | 784 | 81.14% ± 1.30% | 84.62% ± 0.53% | 86.97% ± 0.63% | 89.04% ± 0.57% | 92.18% ± 0.00% |
| HOG control | 1,296 | 91.21% ± 0.45% | 93.55% ± 0.45% | 95.31% ± 0.19% | 96.70% ± 0.16% | 97.15% ± 0.00% |
| Human topology + direction + curvature | 20 | 59.54% ± 1.02% | 62.11% ± 1.47% | 64.03% ± 0.56% | 65.79% ± 0.09% | 66.40% ± 0.00% |

## Research questions

1. **Did LLM-generated biases beat raw pixels in the low-data regime?** Yes.
   The best Qwen representation improved test accuracy by 1.79 points at 500
   examples and also led raw pixels at 250, 1,000, 5,000, and 60,000 examples.
   HOG remained ahead at every size.

2. **Did they beat the human-designed stroke-flow bias?** Yes. At 500 examples,
   the best Qwen representation scored 86.41%, compared with 62.11% for the
   original topology + direction + curvature seed. The comparison is for the
   representations as specified; the Qwen candidate also includes all raw
   pixels.

3. **Did Qwen revise hypotheses based on evidence?** The search fed targeted
   confusion counts back into proposals. Across 95 recorded targeted
   comparisons, 53 confusion counts decreased, 40 increased, and two were
   unchanged. The feedback loop changed proposals and produced mixed outcomes;
   it did not reliably fix every targeted error.

4. **Which assumptions survived ablation?** All 20 archived operator-removal
   ablations reduced validation accuracy relative to their parent. For the top
   candidate, removing stroke direction reduced validation accuracy@500 from
   84.15% to 81.70% (−2.45 points); removing raw pixels reduced it to 41.00%.
   Removing topology from the second Qwen finalist reduced it by 1.65 points.
   These are validation ablations, not additional test-set tuning.

5. **Did explicit stroke direction help?** Yes, in combination with raw pixels.
   The selected candidate’s test score was 1.79 points above the raw-pixel
   control at 500 examples, and its validation ablation also showed a
   2.45-point loss when stroke direction was removed. Stroke direction by
   itself scored only 41.00% in that ablation, so the evidence supports the
   combination rather than a standalone direction representation.

6. **Was curvature more useful than inferred pen order?** The experiment did
   not infer pen order: the architecture intentionally uses orientation-neutral
   path features. It therefore cannot support a curvature-versus-pen-order
   claim. The original curvature-containing seed scored 62.11% at 500 examples;
   the best Qwen candidate used explicit stroke direction and no curvature.
   Removing curvature lowered its one tested parent by one validation point, a
   modest result from a single ablation.

7. **Did search discover simpler representations instead of larger ones?** Not
   for the top result. The best representation used 792 features, only eight
   more than raw pixels, and improved accuracy modestly. The 2-feature symmetry
   finalist scored 19.32% at 500 examples, and the 4-feature topology-only
   finalist scored 40.78%. The search found a compact addition to a strong pixel
   baseline, not a smaller replacement.

## Stop decision

The search reached 62 unique candidates, within the planned 60–100 range. Its
best validation accuracy@500 was 84.15% in generation 5. The best scores in
generations 8, 9, and 10 were 70.55%, 83.35%, and 68.75%; none improved the
generation-5 result. The search has stopped at the planned boundary and was not
expanded into unrestricted AutoML.

## Reproducibility artifacts

- Search metadata: `results/search_metadata.json`
- Search and ablations: `results/search.jsonl`
- Frozen finalists: `results/finalists.json`
- Final test results: `results/final_evaluation.json`
- PNGs: `results/figures/`
- Search/evaluation/plot commands: `README.md`

The JSON, JSONL, PNG, and cache outputs are local generated artifacts ignored
by Git. The source scripts and this report are tracked; rerunning the documented
commands regenerates the artifacts.
