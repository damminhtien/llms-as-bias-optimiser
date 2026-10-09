# V2 Representation Program Synthesis — Experimental Report

Generated 2026-10-09 from local archives and reports.

## Search results

### Discovery track

- Candidate archive: `results/program_search_discovery.jsonl` (SHA-256 `26ceb8d94b0ba78352a2f3cc3c939b0892cd362c5798f25c8c50d9d3dbbb8772`).
- Records: 200 (20 seeds + 10 × 18 proposals).
- LLM response records: 97; accepted response batches: 55.
- Requested programs per LLM batch: 1: 23, 2: 6, 3: 5, 4: 4, 5: 2, 6: 5, 7: 11, 8: 5, 9: 2, 10: 5, 11: 3, 12: 2, 13: 1, 14: 1, 15: 4, 16: 1, 18: 17.
- Generation counts: 0: 20, 1: 18, 2: 18, 3: 18, 4: 18, 5: 18, 6: 18, 7: 18, 8: 18, 9: 18, 10: 18.
- Implemented niches with no candidates: compositional, frequency_scale, geometry, stroke_dynamics.
- MAP-Elites occupied niches: graph_structure (37), hybrid (141), spatial_relations (21), topology (1).

| Candidate | Niche | Features | Validation accuracy @500 | @5,000 | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| `cycle_angle_hist` (`bd5c928dda412ccbeebbc86a1ccd32210ec06b932870896d250c2618f2788546`) | hybrid | 16 | 0.7180 | 0.7420 | 0.7241 |
| `cycle_angle_hist` (`94fd97945999217041d0a20d517fa967151307f0bb89e13f9dc43c4bfaf2942a`) | hybrid | 20 | 0.6670 | 0.6940 | 0.6741 |
| `spatial_cycle_hist` (`641caab7c8b8870389bb818158a9e8b95142e9c1273e2a5ad35fe8ae54574a0d`) | hybrid | 25 | 0.6665 | 0.7010 | 0.6764 |
| `cycle_angle_hist` (`6f533e03afd8542ff0e4d64023bee3ba1504f038d651eebccd08413cd8e3829b`) | hybrid | 24 | 0.6420 | 0.6755 | 0.6514 |
| `angle_hist_12` (`14fe302bfd1ad3f55d7a323965d2ad110787830a4c77c0c69721fe265704fbcc`) | hybrid | 12 | 0.6345 | 0.6505 | 0.6376 |


### Augmentation track

- Candidate archive: `results/program_search_augmentation.jsonl` (SHA-256 `42f0cc90e0676ef65cdd417862a9eabf74890caa64974e6cf5f3edf49660dbc7`).
- Records: 200 (20 seeds + 10 × 18 proposals).
- LLM response records: 98; accepted response batches: 61.
- Requested programs per LLM batch: 1: 28, 2: 11, 3: 4, 4: 40, 6: 2, 9: 1, 10: 1, 12: 2, 16: 1, 18: 8.
- Generation counts: 0: 20, 1: 18, 2: 18, 3: 18, 4: 18, 5: 18, 6: 18, 7: 18, 8: 18, 9: 18, 10: 18.
- Implemented niches with no candidates: compositional, frequency_scale, geometry, raw_pixels, stroke_dynamics.
- MAP-Elites occupied niches: graph_structure (28), hybrid (166), spatial_relations (5), topology (1).

| Candidate | Niche | Features | Validation accuracy @500 | @5,000 | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| `spatial_angle_mix` (`d9dde133e86560ca41a49a77cb072a91824ed39a74e13aaa14eb2f94a0ddc817`) | hybrid | 11 | 0.6445 | 0.6570 | 0.6464 |
| `geo_angle_hist` (`bae0f44f1c9ff2fc2f6e56e750deb0628db62c5af1630152c7e76e20d5335cbf`) | hybrid | 12 | 0.6340 | 0.6460 | 0.6354 |
| `angle_cycle_ratio` (`9a4948ce785ba2e11955dac290563b60cffc3c2ec99fa0c586a386f7de6403b4`) | hybrid | 16 | 0.6315 | 0.6550 | 0.6374 |
| `curve_hist` (`bfd9b6b097201f5e8b5801fd75173b0063eca43f46051c355796cb9c28ee55c6`) | hybrid | 12 | 0.6305 | 0.6485 | 0.6343 |
| `geo_curve_hist` (`74ce4151ad5276859d399f933809c3dc79d7db116eb482e18798f517b0f54ac8`) | hybrid | 16 | 0.6290 | 0.6510 | 0.6343 |


## Raw-pixel augmentation controls

Selection manifest SHA-256: `98d8973e489e7992547288bb46d6ef6073c3ca69213e9adccdda2b63faaef5dd`.
Official test accessed: `False`.
These are post-search compositions, not new LLM proposals; they use the same MNIST validation split that selected the finalists.

| Representation | Type | Dim | Accuracy @500 | Δ vs raw | Accuracy @5,000 | Δ vs raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw_pixels | raw_pixels_baseline | 784 | 0.8170 | +0.0000 | 0.8765 | +0.0000 |
| raw_plus_angle_hist_12 | raw_pixels_plus_frozen_discovery_program | 796 | 0.8525 | +0.0355 | 0.9170 | +0.0405 |
| raw_plus_spatial_2x3 | raw_pixels_plus_spatial_control | 790 | 0.8170 | +0.0000 | 0.8785 | +0.0020 |

These post-search controls compose raw pixels with depth-compatible frozen finalists plus a fixed spatial control; they are not new LLM proposals and use the same validation split that selected the finalists.

## Frozen transfer evaluation

Manifest SHA-256: `98d8973e489e7992547288bb46d6ef6073c3ca69213e9adccdda2b63faaef5dd`.
Candidate archive SHA-256: `26ceb8d94b0ba78352a2f3cc3c939b0892cd362c5798f25c8c50d9d3dbbb8772`.
Training sizes [500, 5000], train seeds [11, 23, 47], stratified test sample 2000 per domain.

| Dataset | Representation | Type | Dim | Train size | Test accuracy (mean ± SD) |
| --- | --- | --- | ---: | ---: | ---: |
| emnist_digits | angle_hist_12 | discovered_program | 12 | 500 | 0.6530 ± 0.0100 |
| emnist_digits | angle_hist_12 | discovered_program | 12 | 5000 | 0.6705 ± 0.0035 |
| emnist_digits | cycle_angle_hist | discovered_program | 16 | 500 | 0.7402 ± 0.0090 |
| emnist_digits | cycle_angle_hist | discovered_program | 16 | 5000 | 0.7728 ± 0.0035 |
| emnist_digits | cycle_angle_hist | discovered_program | 20 | 500 | 0.6925 ± 0.0069 |
| emnist_digits | cycle_angle_hist | discovered_program | 20 | 5000 | 0.7190 ± 0.0026 |
| emnist_digits | cycle_angle_hist | discovered_program | 24 | 500 | 0.6792 ± 0.0150 |
| emnist_digits | cycle_angle_hist | discovered_program | 24 | 5000 | 0.7068 ± 0.0023 |
| emnist_digits | spatial_cycle_hist | discovered_program | 25 | 500 | 0.7728 ± 0.0123 |
| emnist_digits | spatial_cycle_hist | discovered_program | 25 | 5000 | 0.8120 ± 0.0010 |
| emnist_digits | human_topology | human_structural_control | 4 | 500 | 0.4573 ± 0.0050 |
| emnist_digits | human_topology | human_structural_control | 4 | 5000 | 0.4565 ± 0.0096 |
| emnist_digits | human_topology_curvature | human_structural_control | 12 | 500 | 0.4898 ± 0.0127 |
| emnist_digits | human_topology_curvature | human_structural_control | 12 | 5000 | 0.5192 ± 0.0060 |
| emnist_digits | human_topology_spatial | human_structural_control | 7 | 500 | 0.7670 ± 0.0020 |
| emnist_digits | human_topology_spatial | human_structural_control | 7 | 5000 | 0.7803 ± 0.0028 |
| emnist_letters | angle_hist_12 | discovered_program | 12 | 500 | 0.3383 ± 0.0115 |
| emnist_letters | angle_hist_12 | discovered_program | 12 | 5000 | 0.3720 ± 0.0053 |
| emnist_letters | cycle_angle_hist | discovered_program | 16 | 500 | 0.3885 ± 0.0072 |
| emnist_letters | cycle_angle_hist | discovered_program | 16 | 5000 | 0.4272 ± 0.0073 |
| emnist_letters | cycle_angle_hist | discovered_program | 20 | 500 | 0.3680 ± 0.0044 |
| emnist_letters | cycle_angle_hist | discovered_program | 20 | 5000 | 0.4037 ± 0.0051 |
| emnist_letters | cycle_angle_hist | discovered_program | 24 | 500 | 0.3778 ± 0.0109 |
| emnist_letters | cycle_angle_hist | discovered_program | 24 | 5000 | 0.4115 ± 0.0101 |
| emnist_letters | spatial_cycle_hist | discovered_program | 25 | 500 | 0.4403 ± 0.0053 |
| emnist_letters | spatial_cycle_hist | discovered_program | 25 | 5000 | 0.4993 ± 0.0042 |
| emnist_letters | human_topology | human_structural_control | 4 | 500 | 0.1770 ± 0.0113 |
| emnist_letters | human_topology | human_structural_control | 4 | 5000 | 0.1928 ± 0.0086 |
| emnist_letters | human_topology_curvature | human_structural_control | 12 | 500 | 0.2200 ± 0.0093 |
| emnist_letters | human_topology_curvature | human_structural_control | 12 | 5000 | 0.2478 ± 0.0071 |
| emnist_letters | human_topology_spatial | human_structural_control | 7 | 500 | 0.3160 ± 0.0085 |
| emnist_letters | human_topology_spatial | human_structural_control | 7 | 5000 | 0.3428 ± 0.0070 |
| fashion_mnist | angle_hist_12 | discovered_program | 12 | 500 | 0.3453 ± 0.0086 |
| fashion_mnist | angle_hist_12 | discovered_program | 12 | 5000 | 0.3855 ± 0.0056 |
| fashion_mnist | cycle_angle_hist | discovered_program | 16 | 500 | 0.3783 ± 0.0125 |
| fashion_mnist | cycle_angle_hist | discovered_program | 16 | 5000 | 0.4198 ± 0.0053 |
| fashion_mnist | cycle_angle_hist | discovered_program | 20 | 500 | 0.3620 ± 0.0141 |
| fashion_mnist | cycle_angle_hist | discovered_program | 20 | 5000 | 0.4100 ± 0.0048 |
| fashion_mnist | cycle_angle_hist | discovered_program | 24 | 500 | 0.3593 ± 0.0123 |
| fashion_mnist | cycle_angle_hist | discovered_program | 24 | 5000 | 0.4105 ± 0.0063 |
| fashion_mnist | spatial_cycle_hist | discovered_program | 25 | 500 | 0.6672 ± 0.0111 |
| fashion_mnist | spatial_cycle_hist | discovered_program | 25 | 5000 | 0.7080 ± 0.0046 |
| fashion_mnist | human_topology | human_structural_control | 4 | 500 | 0.1658 ± 0.0155 |
| fashion_mnist | human_topology | human_structural_control | 4 | 5000 | 0.1712 ± 0.0042 |
| fashion_mnist | human_topology_curvature | human_structural_control | 12 | 500 | 0.2150 ± 0.0065 |
| fashion_mnist | human_topology_curvature | human_structural_control | 12 | 5000 | 0.2403 ± 0.0015 |
| fashion_mnist | human_topology_spatial | human_structural_control | 7 | 500 | 0.4462 ± 0.0105 |
| fashion_mnist | human_topology_spatial | human_structural_control | 7 | 5000 | 0.4653 ± 0.0047 |
| kmnist | angle_hist_12 | discovered_program | 12 | 500 | 0.2697 ± 0.0020 |
| kmnist | angle_hist_12 | discovered_program | 12 | 5000 | 0.2948 ± 0.0058 |
| kmnist | cycle_angle_hist | discovered_program | 16 | 500 | 0.2838 ± 0.0068 |
| kmnist | cycle_angle_hist | discovered_program | 16 | 5000 | 0.3035 ± 0.0074 |
| kmnist | cycle_angle_hist | discovered_program | 20 | 500 | 0.2723 ± 0.0013 |
| kmnist | cycle_angle_hist | discovered_program | 20 | 5000 | 0.2988 ± 0.0047 |
| kmnist | cycle_angle_hist | discovered_program | 24 | 500 | 0.2772 ± 0.0028 |
| kmnist | cycle_angle_hist | discovered_program | 24 | 5000 | 0.3072 ± 0.0038 |
| kmnist | spatial_cycle_hist | discovered_program | 25 | 500 | 0.4243 ± 0.0089 |
| kmnist | spatial_cycle_hist | discovered_program | 25 | 5000 | 0.4538 ± 0.0115 |
| kmnist | human_topology | human_structural_control | 4 | 500 | 0.2138 ± 0.0085 |
| kmnist | human_topology | human_structural_control | 4 | 5000 | 0.2248 ± 0.0028 |
| kmnist | human_topology_curvature | human_structural_control | 12 | 500 | 0.2415 ± 0.0046 |
| kmnist | human_topology_curvature | human_structural_control | 12 | 5000 | 0.2538 ± 0.0056 |
| kmnist | human_topology_spatial | human_structural_control | 7 | 500 | 0.2948 ± 0.0132 |
| kmnist | human_topology_spatial | human_structural_control | 7 | 5000 | 0.3030 ± 0.0005 |

Test split provenance (partition, source URL, and checksum):
- `emnist_digits`: publisher's official test split; source `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip` (SHA-256 `fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e`).
- `emnist_letters`: publisher's official test split; source `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip` (SHA-256 `fb9bb67e33772a9cc0b895e4ecf36d2cf35be8b709693c3564cea2a019fcda8e`).
- `kmnist`: publisher's official test split; source `http://codh.rois.ac.jp/kmnist/dataset/kmnist` (SHA-256 `75699c742c073b9130e0dfeae18975ec0c98aaa5635e42a4fe625d2f4a7fe5eb`).
- `fashion_mnist`: publisher's official test split; source `https://fashion-mnist.s3.eu-central-1.amazonaws.com` (SHA-256 `3c93b65338c05f2ad2e10fa644d754a8d307b5db1d3c478c460ce170039200d3`).

## Sequence counterfactuals

Manifest SHA-256: `98d8973e489e7992547288bb46d6ef6073c3ca69213e9adccdda2b63faaef5dd`; test sample n=2000.

| Candidate | Sequence node | Mean accuracy change | Mean prediction agreement |
| --- | --- | ---: | ---: |
| cycle_angle_hist | angles | +0.0000 | 1.0000 |
| cycle_angle_hist | delta | +0.0000 | 1.0000 |
| cycle_angle_hist | angles | +0.0000 | 1.0000 |
| cycle_angle_hist | delta | +0.0000 | 1.0000 |
| spatial_cycle_hist | delta | +0.0000 | 1.0000 |
| cycle_angle_hist | angles | +0.0000 | 1.0000 |
| angle_hist_12 | angles | +0.0000 | 1.0000 |

Every tested sequence-shuffle intervention left predictions unchanged (mean accuracy change +0.0000; agreement 1.0000). These finalists provide no evidence that sequence order itself matters; their distributional features do not support an order-sensitive mechanism.

## Invariance search and orbit pooling

Official test accessed: `False`.
Baseline validation accuracy: 0.7580.

| Proposal | Transform program | Accuracy delta | Agreement | Passed rule |
| --- | --- | ---: | ---: | --- |
| Minor Horizontal Shift | `{"steps":[{"op":"translate","params":{"dx":2,"dy":0}}]}` | +0.0000 | 1.0000 | True |
| Slight Counter-Clockwise Rotation | `{"steps":[{"op":"rotate","params":{"degrees":10.0}}]}` | -0.0440 | 0.7460 | False |
| Stroke Thinning via Erosion | `{"steps":[{"op":"erode","params":{"radius":1}}]}` | -0.5320 | 0.2460 | False |
| Combined Shear and Elastic Distortion | `{"steps":[{"op":"shear","params":{"amount":0.15}},{"op":"elastic","params":{"amplitude":1.5,"sigma":4.0}}]}` | -0.0680 | 0.7060 | False |

Orbit mean on validation holdout: 0.7440 → 0.7420 (Δ -0.0020).

## Cross-candidate cache profile

Compared 16 programs × 96 images. Feature matrices exactly equal: `True`. Uncached 0.502s; cached 0.160s; speedup 3.136×; metrics `{'bytes': 1442140, 'entries': 480, 'evictions': 0, 'hit_rate': 0.7916666666666666, 'hits': 1824, 'max_bytes': 268435456, 'misses': 480}`.

## Interpretation boundary

The discovery track excludes raw pixels and the augmentation track permits them. Transfer candidates are selected on MNIST validation and frozen before transfer test access. The transfer report uses stratified test samples of the stated size; it does not claim full-test accuracy. The invariance search uses MNIST train/validation only. Counterfactual accuracy changes support or weaken sequence-mechanism claims for the tested sample and representation. Both searches evaluated the same number of candidates, but augmentation request batches were capped at four after output truncation; cross-track maxima are descriptive, not a controlled comparison. None of these results alone proves a general or universal handwriting inductive bias.

Dataset sources: [NIST EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset), [CODH KMNIST](https://github.com/rois-codh/kmnist), [OpenML KMNIST mirror](https://www.openml.org/d/41982), and [Zalando Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).
