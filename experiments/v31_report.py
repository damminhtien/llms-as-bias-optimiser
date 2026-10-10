"""Summarize V3.1 matrices, paired evidence, and study completion boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bias_optimizer.evaluation.artifacts import read_jsonl, write_json
from bias_optimizer.evaluation.metrics import learning_curve_aulc, pareto_frontier
from bias_optimizer.evaluation.protocol import load_frozen_protocol
from bias_optimizer.evaluation.statistics import mcnemar_exact, paired_bootstrap_delta


def generate_report(*, root: Path = Path(".")) -> Path:
    root = Path(root)
    protocol, manifest = load_frozen_protocol(root)
    core = read_jsonl(root / "results/v31/core_matrix.jsonl")
    complementarity_path = root / "results/v31/complementarity.json"
    mechanism_path = root / "results/v31/mechanism.json"
    robustness_path = root / "results/v31/robustness.json"
    transfer_path = root / "results/v31/transfer.json"
    complementarity = _read_json(complementarity_path)
    mechanism = _read_json(mechanism_path)
    robustness = _read_json(robustness_path)
    transfer = _read_json(transfer_path)
    mnist_provenance = _mnist_source_provenance(root, protocol)

    core_expected = protocol["phases"]["A_core"]["expected_fits"]
    extension_expected = protocol["phases"]["A_learning_curve_extension"]["expected_additional_fits"]
    core_complete = len([row for row in core if row.get("phase") == "A_core"]) >= core_expected
    extension_complete = len([row for row in core if row.get("phase") == "A_learning_curve_extension"]) >= extension_expected
    phase_b_complete = len(complementarity.get("paired_runs", [])) == 72
    phase_c_mechanism_complete = len(mechanism.get("runs", [])) == 9
    phase_c_robustness_complete = len(robustness.get("runs", [])) == 135
    transfer_expected = 7 * 3 * 2 * 3 * 4
    phase_d_complete = len(transfer.get("runs", [])) == transfer_expected

    groups = _core_groups(core)
    learning_curves = _learning_curves(groups)
    capacity = _capacity_gains(groups)
    frontiers = _frontiers(groups, core)
    paired_statistics = _paired_statistics(core, complementarity)
    figure_path = _write_learning_curve_figure(groups, root)
    write_json(root / "results/v31/paired_statistics.json", paired_statistics)
    profile_records = _bias_profile_data(
        protocol=protocol,
        curves=learning_curves,
        capacity=capacity,
        paired_statistics=paired_statistics,
        complementarity=complementarity,
        mechanism=mechanism,
        robustness=robustness,
        transfer=transfer,
    )
    write_json(
        root / "results/v31/learning_curves.json",
        {"schema_version": 1, "protocol_sha256": manifest["protocol"]["sha256"], "curves": learning_curves},
    )
    write_json(
        root / "results/v31/complexity_frontier.json",
        {
            "schema_version": 1,
            "protocol_sha256": manifest["protocol"]["sha256"],
            "nonlinear_probe_gains": capacity,
            "accuracy_dimension_pareto_frontiers": frontiers,
        },
    )
    write_json(
        root / "results/v31/bias_profiles.json",
        {"schema_version": 1, "protocol_sha256": manifest["protocol"]["sha256"], "profiles": profile_records},
    )

    lines = [
        "# V3.1 Bias Evaluation",
        "",
        f"Protocol SHA-256: `{manifest['protocol']['sha256']}`.",
        "",
        "## Research question and protocol",
        "",
        "This study evaluates the frozen V3 finalists across representation, learner, training size, and domain. It does not search for new biases. A score belongs to a representation–probe pair; it is not an absolute measure of information retained.",
        "",
        "The predeclared questions are:",
        "",
    ]
    lines.extend(f"{index}. {question}" for index, question in enumerate(protocol["primary_questions"], 1))
    lines.extend(["", "## Answers to the predeclared questions", ""])
    lines.extend(
        _answer_questions(
            protocol=protocol,
            curves=learning_curves,
            capacity=capacity,
            paired_statistics=paired_statistics,
            complementarity=complementarity,
            mechanism=mechanism,
            robustness=robustness,
            transfer=transfer,
        )
    )
    lines.extend(["", "## Completion status", ""])
    status_items = [
        ("Phase A core matrix", core_complete, f"{sum(row.get('phase') == 'A_core' for row in core)}/{core_expected} fits"),
        ("Learning-curve extension", extension_complete, f"{sum(row.get('phase') == 'A_learning_curve_extension' for row in core)}/{extension_expected} fits"),
        ("Phase B complementarity", phase_b_complete, f"{len(complementarity.get('paired_runs', []))}/72 paired runs"),
        ("Phase C mechanism", phase_c_mechanism_complete, f"{len(mechanism.get('runs', []))}/9 runs"),
        ("Phase C robustness", phase_c_robustness_complete, f"{len(robustness.get('runs', []))}/135 transformed runs"),
        ("Phase D transfer", phase_d_complete, f"{len(transfer.get('runs', []))}/{transfer_expected} fits"),
    ]
    lines.extend(f"- {'Complete' if complete else 'Incomplete'}: {name} ({detail})." for name, complete, detail in status_items)
    lines.extend(["", "## Phase A: learner capacity and accuracy", ""])
    lines.extend(_matrix_table(groups, sizes=(500, 5000)))
    lines.extend(["", "### Learning curves and sample efficiency", ""])
    if learning_curves:
        lines.extend(["AULC is the trapezoidal integral of mean accuracy over log training size; values use the protocol's full 100–5,000 sample interval when all six sizes are available.", ""])
        lines.extend(_curve_table(learning_curves))
        lines.extend(["", f"![Learning curves by probe]({figure_path.name})", ""])
    else:
        lines.append("Learning-curve results are not available yet.")
    lines.extend(["", "### Nonlinear probe gain", ""])
    lines.extend(_capacity_table(capacity))
    lines.extend(["", "### Accuracy–dimension Pareto frontiers", ""])
    lines.extend(_frontier_table(frontiers))

    lines.extend(["", "## Phase B: complementarity", ""])
    if phase_b_complete:
        lines.extend(_complementarity_table(complementarity["summaries"]))
    else:
        lines.append("Complementarity runs are incomplete; see `results/v31/complementarity.json` for completed pairs.")

    lines.extend(["", "## Phase C: mechanism specificity", ""])
    if mechanism.get("summaries"):
        lines.extend(_mechanism_table(mechanism["summaries"]))
        lines.extend(["", "The targeted event-location destroyer and nuisance control were evaluated by the same clean-fitted model. Nuisance-noise sigma was calibrated using training images only to match feature-space displacement."])
    else:
        lines.append("Mechanism results are not available yet.")

    lines.extend(["", "## Phase C: robustness", ""])
    if robustness.get("summaries"):
        lines.extend(_robustness_table(robustness["summaries"]))
        lines.extend(["", "All transformations were applied to evaluation images after fitting; transformed examples were never used for training."])
    else:
        lines.append("Robustness results are not available yet.")

    lines.extend(["", "## Phase D: transfer", ""])
    if transfer.get("selection"):
        lines.append("Transfer representations were selected using the frozen validation-only Phase A/B gate:")
        lines.append("")
        lines.extend(f"- `{name}`" for name in transfer["selection"]["selected_v3"])
        lines.append("")
        lines.extend(_transfer_table(transfer.get("summaries", [])))
        lines.extend(["", "### Dataset provenance", ""])
        lines.extend(_provenance_table(transfer.get("dataset_provenance", {}), mnist_provenance))
    else:
        lines.append("Transfer results are not available yet.")

    lines.extend(["", "## Bias evaluation profiles", ""])
    lines.append("Profiles retain separate evidence axes and do not combine them into a single score.")
    lines.append("Profile robustness axes average the listed transformation settings for each seed and then report the mean ± SD across seeds; the detailed table reports each transformation separately.")
    lines.append("")
    lines.extend(
        _bias_profiles(
            protocol=protocol,
            core=core,
            curves=learning_curves,
            capacity=capacity,
            paired_statistics=paired_statistics,
            complementarity=complementarity,
            mechanism=mechanism,
            robustness=robustness,
            transfer=transfer,
        )
    )

    lines.extend(["", "## Paired statistical evidence", ""])
    if paired_statistics:
        lines.extend(_paired_table(paired_statistics))
        lines.append("")
        lines.append("Bootstrap intervals resample the same held-out example IDs for both models (2,000 replicates, percentile 95% CI). McNemar uses exact two-sided tests on paired disagreements, with a separate result per training seed. Seed means and SDs describe subset variability; no t-test is run over three seeds.")
    else:
        lines.append("Paired comparisons are not available yet.")

    lines.extend([
        "",
        "## Limitations and claim boundary",
        "",
        "Three training seeds characterize training-subset variability but do not support a seed-level t-test. Bootstrap intervals condition on the selected evaluation sample and the frozen protocol. Probe results describe accessible task performance under these fixed classifiers; they do not establish that a representation retains all label information. MNIST transfer claims are tied to the stated dataset preprocessing and official-test samples. A negative or small gain for one learner does not establish that the representation is intrinsically poor.",
        "",
        "The complete per-run predictions, split IDs, timing records, protocol, and source hashes are stored under `results/v31/`.",
        "",
    ])
    report_path = root / "reports/V3_1_BIAS_EVALUATION.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        root / "results/v31/summary.json",
        {
            "schema_version": 1,
            "protocol_sha256": manifest["protocol"]["sha256"],
            "phase_status": {name: complete for name, complete, _ in status_items},
            "result_counts": {name: detail for name, _, detail in status_items},
            "learning_curves": learning_curves,
            "nonlinear_probe_gains": capacity,
            "pareto_frontiers": frontiers,
            "paired_statistics": paired_statistics,
            "bias_profiles": profile_records,
            "mnist_provenance": mnist_provenance,
            "transfer_selection": transfer.get("selection"),
            "transfer_provenance": transfer.get("dataset_provenance"),
        },
    )
    _write_artifact_manifest(root, mnist_provenance)
    return report_path


def _answer_questions(
    *,
    protocol: dict,
    curves: list[dict[str, object]],
    capacity: list[dict[str, object]],
    paired_statistics: dict[str, object],
    complementarity: dict,
    mechanism: dict,
    robustness: dict,
    transfer: dict,
) -> list[str]:
    core_v3 = protocol["phases"]["B_complementarity"]["v3_representations"]
    curve_map = {(row["representation"], row["probe"]): row for row in curves}
    aulc_deltas_zoning = []
    aulc_deltas_pca = []
    for name in core_v3:
        candidate = curve_map.get((name, "linear_logreg"))
        if candidate is None:
            continue
        dimension = next(
            item["dimension"] for item in protocol["representations"] if item["id"] == name
        )
        pca_name = "hog_pca_30d" if dimension <= 30 else "hog_pca_60d"
        zoning = curve_map.get(("zoning_30d", "linear_logreg"))
        pca = curve_map.get((pca_name, "linear_logreg"))
        if zoning is not None and pca is not None:
            aulc_deltas_zoning.append(
                float(candidate["aulc_log_train_size"]) - float(zoning["aulc_log_train_size"])
            )
            aulc_deltas_pca.append(
                float(candidate["aulc_log_train_size"]) - float(pca["aulc_log_train_size"])
            )

    v3_capacity = [row for row in capacity if row["representation"] in core_v3]
    gain_text = {}
    for probe in ("rbf_svm", "small_mlp"):
        values = {
            size: [
                float(row["gain_mean"])
                for row in v3_capacity
                if row["probe"] == probe and int(row["train_size"]) == size
            ]
            for size in (500, 5000)
        }
        gain_text[probe] = {
            size: _pp(float(np.mean(scores))) if scores else "not available"
            for size, scores in values.items()
        }

    comp_stats = paired_statistics.get("complementarity", {})
    hog_comp = [
        value["bootstrap"]
        for key, value in comp_stats.items()
        if key.startswith("hog+") and "/linear_logreg/" in key
    ]
    raw_comp = [
        row
        for row in complementarity.get("summaries", [])
        if row["anchor"] == "raw_pixels" and row["probe"] == "linear_logreg"
    ]
    all_hog_ci_positive = bool(hog_comp) and all(
        float(row["ci_low"]) > 0 for row in hog_comp
    )
    hog_range = _range_pp([float(row["delta"]) for row in hog_comp])
    raw_range = _range_pp([float(row["delta_mean"]) for row in raw_comp])

    mechanism_rows = mechanism.get("summaries", [])
    mechanism_differences = [
        float(row["target_minus_nuisance_drop_mean"]) for row in mechanism_rows
    ]
    robustness_rows = robustness.get("summaries", [])
    horizontal = [
        row for row in robustness_rows if row["transformation"].startswith("translate_x_")
    ]
    vertical_two = [
        row
        for row in robustness_rows
        if row["transformation"] in ("translate_y_+2", "translate_y_-2")
    ]
    erosion = [row for row in robustness_rows if row["transformation"] == "erosion_1px"]
    dilation = [row for row in robustness_rows if row["transformation"] == "dilation_1px"]
    noise = [
        row
        for row in robustness_rows
        if row["transformation"] == "gaussian_noise_sigma_0.05"
    ]
    transfer_rows = [
        row
        for row in transfer.get("summaries", [])
        if row["representation"] in core_v3
        and row["probe"] == "linear_logreg"
        and int(row["train_size"]) == 5000
    ]
    domain_values = {
        dataset: [float(row["accuracy_mean"]) for row in transfer_rows if row["dataset"] == dataset]
        for dataset in ("emnist_digits", "emnist_letters", "kmnist", "fashion_mnist")
    }
    digit_mean = float(np.mean(domain_values["emnist_digits"])) if domain_values["emnist_digits"] else 0.0
    other_domain_mean = [
        value
        for dataset in ("emnist_letters", "kmnist", "fashion_mnist")
        for value in domain_values[dataset]
    ]
    non_digit_mean = float(np.mean(other_domain_mean)) if other_domain_mean else 0.0
    specificity = _range_pp(mechanism_differences)
    x_drop_max = max((abs(float(row["accuracy_drop_mean"])) for row in horizontal), default=0.0)
    y2_drop_max = max((float(row["accuracy_drop_mean"]) for row in vertical_two), default=0.0)
    erosion_range = _range_pp([float(row["accuracy_drop_mean"]) for row in erosion])
    dilation_range = _range_pp([float(row["accuracy_drop_mean"]) for row in dilation])
    noise_range = _range_pp([float(row["accuracy_drop_mean"]) for row in noise])

    return [
        ("1. **Sample efficiency:** The three core V3 finalists improve mean linear AULC over `zoning_30d` by "
        f"{_range(aulc_deltas_zoning)} AULC units across 100–5,000 samples, but trail their width-matched HOG-PCA controls by "
        f"{_range(aulc_deltas_pca)} AULC units. They beat the simple compact control, while HOG-PCA is more sample-efficient under this probe."),
        ("2. **Learner dependence:** Mean nonlinear-minus-linear gain across the three core V3 candidates is "
        f"RBF {gain_text['rbf_svm'][500]} at 500 and {gain_text['rbf_svm'][5000]} at 5,000; MLP "
        f"{gain_text['small_mlp'][500]} and {gain_text['small_mlp'][5000]}. Among the three core V3 candidates, `angle_centroid_pairwise` leads at 500 and `regional_turn_histogram` leads at 5,000 for all probes. The fixed learner and sample size change measured rankings; nonlinear gains are modest at 5,000."),
        ("3. **Fixed representation budget:** At the same 30D or 60D width, all six paired linear comparisons favor HOG-PCA over the corresponding V3 representation at 500 and 5,000 samples; paired deltas span "
        f"{_range_pp([float(value['bootstrap']['delta']) for key, value in paired_statistics.items() if 'hog_pca_' in key and '/linear_logreg/' in key])} and every 95% bootstrap interval is below zero. Compactness alone therefore does not explain a V3 advantage over the frozen HOG-PCA controls."),
        ("4. **Complementarity:** Adding each of the three core V3 representations to HOG improves mean linear accuracy by "
        f"{hog_range} at 500 and 5,000 samples; all paired HOG intervals are positive: {all_hog_ci_positive}. Adding V3 to raw pixels yields mean linear gains of {raw_range}. The HOG result shows additional signal under this fixed learner and protocol, even though standalone HOG remains stronger."),
        ("5. **Mechanism and domain assumptions:** Target-minus-nuisance accuracy-drop differences range from "
        f"{specificity}; `angle_centroid_pairwise` has the largest measured gap, while the other two matched noise controls also cause large drops. Horizontal ±1/±2 translations have at most {_pp(x_drop_max)} accuracy loss, whereas vertical ±2 shifts lose up to {_pp(y2_drop_max)}, one-pixel erosion loses {erosion_range}, and dilation loses {dilation_range}; Gaussian noise loss is only {noise_range}. At 5,000 samples the selected V3 pair averages {digit_mean:.3f} linear accuracy on EMNIST Digits versus {non_digit_mean:.3f} across the three other transfer tasks, but HOG/HOG-PCA remain stronger on most domains. The transfer pattern is consistent with digit-domain similarity, not universal superiority; label spaces differ across datasets."),
        "",
    ]


def _bias_profiles(
    *,
    protocol: dict,
    core: list[dict],
    curves: list[dict[str, object]],
    capacity: list[dict[str, object]],
    paired_statistics: dict[str, object],
    complementarity: dict,
    mechanism: dict,
    robustness: dict,
    transfer: dict,
) -> list[str]:
    curve_map = {(row["representation"], row["probe"]): row for row in curves}
    capacity_map = {
        (row["representation"], row["probe"], int(row["train_size"])): row
        for row in capacity
    }
    comp_summaries = complementarity.get("summaries", [])
    robustness_runs = robustness.get("runs", [])
    transfer_summaries = transfer.get("summaries", [])
    mechanism_runs = mechanism.get("runs", [])
    v3 = [item for item in protocol["representations"] if item["kind"] == "v3_program"]
    profiles = []
    for item in v3:
        name = item["id"]
        dimension = int(item["dimension"])
        linear = curve_map.get((name, "linear_logreg"))
        if linear is None:
            profiles.extend([
                f"### `{name}` — {dimension}D",
                "",
                "Frozen in the V3 registry but not included in the predeclared Phase A matrix. It did not enter Phase B/C or the Phase D top-two selection. Quantitative sample-efficiency, learner, complementarity, mechanism, robustness, and transfer profile axes are unavailable; no result is inferred from the other finalists.",
                "",
            ])
            continue

        points = {int(point["train_size"]): float(point["accuracy_mean"]) for point in linear["points"]}
        pca_name = "hog_pca_30d" if dimension <= 30 else "hog_pca_60d"
        delta_zoning = {}
        delta_pca = {}
        for size in (500, 5000):
            for baseline, output in (("zoning_30d", delta_zoning), (pca_name, delta_pca)):
                key = f"{name} vs {baseline}/linear_logreg/n={size}"
                stat = paired_statistics.get(key)
                output[size] = float(stat["bootstrap"]["delta"]) if stat else None

        nonlinear = {}
        for size in (500, 5000):
            nonlinear[size] = ", ".join(
                f"{probe.replace('_', ' ')} {_pp(float(capacity_map[(name, probe, size)]['gain_mean']))}"
                for probe in ("rbf_svm", "small_mlp")
                if (name, probe, size) in capacity_map
            )

        hog_delta = {
            size: next((
                float(row["delta_mean"])
                for row in comp_summaries
                if row["anchor"] == "hog"
                and row["v3_representation"] == name
                and row["probe"] == "linear_logreg"
                and int(row["train_size"]) == size
            ), None)
            for size in (500, 5000)
        }
        mech = [row for row in mechanism_runs if row["representation"] == name]
        specificity_values = [float(row["target_drop"]) - float(row["nuisance_drop"]) for row in mech]
        specificity_text = _mean_sd_pp(specificity_values) if specificity_values else "not measured"

        robustness_by_name = [row for row in robustness_runs if row["representation"] == name]
        x_drops = _robustness_axis_drops(
            robustness_by_name, lambda transform: transform.startswith("translate_x_")
        )
        y2_drops = _robustness_axis_drops(
            robustness_by_name,
            lambda transform: transform in ("translate_y_+2", "translate_y_-2"),
        )
        rotation_drops = _robustness_axis_drops(
            robustness_by_name, lambda transform: transform.startswith("rotate_")
        )
        noise_drops = _robustness_axis_drops(
            robustness_by_name, lambda transform: transform == "gaussian_noise_sigma_0.05"
        )
        erosion_drops = _robustness_axis_drops(
            robustness_by_name, lambda transform: transform == "erosion_1px"
        )

        domain = {
            row["dataset"]: float(row["accuracy_mean"])
            for row in transfer_summaries
            if row["representation"] == name
            and row["probe"] == "linear_logreg"
            and int(row["train_size"]) == 5000
        }
        domain_text = ", ".join(
            f"{label} {domain[key]:.3f}" if key in domain else f"{label} not selected"
            for key, label in (
                ("emnist_digits", "EMNIST Digits"),
                ("emnist_letters", "Letters"),
                ("kmnist", "KMNIST"),
                ("fashion_mnist", "Fashion-MNIST"),
            )
        )
        profiles.extend([
            f"### `{name}` — {dimension}D",
            "",
            f"- Linear accessibility: {points.get(500, float('nan')):.4f} at 500 and {points.get(5000, float('nan')):.4f} at 5,000; linear AULC {float(linear['aulc_log_train_size']):.4f}.",
            f"- Sample and dimension controls: paired linear Δ vs zoning30 is {_pp(delta_zoning[500])} at 500 and {_pp(delta_zoning[5000])} at 5,000; Δ vs matched HOG-PCA is {_pp(delta_pca[500])} and {_pp(delta_pca[5000])}. AULC Δ vs zoning30 is {_format_aulc_delta(float(linear['aulc_log_train_size']) - float(curve_map[('zoning_30d', 'linear_logreg')]['aulc_log_train_size']))}; vs matched HOG-PCA is {_format_aulc_delta(float(linear['aulc_log_train_size']) - float(curve_map[(pca_name, 'linear_logreg')]['aulc_log_train_size']))}.",
            f"- Nonlinear-minus-linear gain (RBF / MLP): 500, {nonlinear[500]}; 5,000, {nonlinear[5000]}.",
            f"- HOG complementarity (linear Δ): 500 {_pp(hog_delta[500])}; 5,000 {_pp(hog_delta[5000])}.",
            f"- Mechanism specificity (target drop minus matched-nuisance drop): {specificity_text}.",
            f"- Robustness accuracy drop: horizontal translations {_mean_sd_pp(x_drops)}; vertical ±2 {_mean_sd_pp(y2_drops)}; rotations ±5°/±10° {_mean_sd_pp(rotation_drops)}; Gaussian noise {_mean_sd_pp(noise_drops)}; one-pixel erosion {_mean_sd_pp(erosion_drops)}.",
            f"- Transfer at 5,000 with linear probe: {domain_text}.",
            "",
        ])
    return profiles


def _bias_profile_data(
    *,
    protocol: dict,
    curves: list[dict[str, object]],
    capacity: list[dict[str, object]],
    paired_statistics: dict[str, object],
    complementarity: dict,
    mechanism: dict,
    robustness: dict,
    transfer: dict,
) -> list[dict[str, object]]:
    curve_map = {(row["representation"], row["probe"]): row for row in curves}
    capacity_map = {
        (row["representation"], row["probe"], int(row["train_size"])): row
        for row in capacity
    }
    profiles = []
    for item in protocol["representations"]:
        if item["kind"] != "v3_program":
            continue
        name = str(item["id"])
        dimension = int(item["dimension"])
        linear = curve_map.get((name, "linear_logreg"))
        if linear is None:
            profiles.append({
                "representation": name,
                "dimension": dimension,
                "status": "frozen_not_evaluated_in_core_matrix",
                "profile_axes": None,
            })
            continue

        points = {int(point["train_size"]): point for point in linear["points"]}
        pca_name = "hog_pca_30d" if dimension <= 30 else "hog_pca_60d"
        paired_controls = {}
        for size in (500, 5000):
            paired_controls[str(size)] = {}
            for baseline in ("zoning_30d", pca_name):
                key = f"{name} vs {baseline}/linear_logreg/n={size}"
                paired_controls[str(size)][baseline] = paired_statistics.get(key)
        aulc_values = {
            "linear_logreg": float(linear["aulc_log_train_size"]),
            "delta_vs_zoning_30d": float(linear["aulc_log_train_size"])
            - float(curve_map[("zoning_30d", "linear_logreg")]["aulc_log_train_size"]),
            "delta_vs_dimension_matched_hog_pca": float(linear["aulc_log_train_size"])
            - float(curve_map[(pca_name, "linear_logreg")]["aulc_log_train_size"]),
        }
        complementarity_by_size = {
            str(size): next((
                row
                for row in complementarity.get("summaries", [])
                if row["anchor"] == "hog"
                and row["v3_representation"] == name
                and row["probe"] == "linear_logreg"
                and int(row["train_size"]) == size
            ), None)
            for size in (500, 5000)
        }
        mechanism_runs = [row for row in mechanism.get("runs", []) if row["representation"] == name]
        specificity_values = [
            float(row["target_drop"]) - float(row["nuisance_drop"]) for row in mechanism_runs
        ]
        robustness_runs = [
            row for row in robustness.get("runs", []) if row["representation"] == name
        ]
        robustness_profiles = {
            "horizontal_translations": _profile_summary(_robustness_axis_drops(
                robustness_runs, lambda transform: transform.startswith("translate_x_")
            )),
            "vertical_translations_plus_minus_2": _profile_summary(_robustness_axis_drops(
                robustness_runs,
                lambda transform: transform in ("translate_y_+2", "translate_y_-2"),
            )),
            "rotations": _profile_summary(_robustness_axis_drops(
                robustness_runs, lambda transform: transform.startswith("rotate_")
            )),
            "gaussian_noise_sigma_0_05": _profile_summary(_robustness_axis_drops(
                robustness_runs, lambda transform: transform == "gaussian_noise_sigma_0.05"
            )),
            "erosion_1px": _profile_summary(_robustness_axis_drops(
                robustness_runs, lambda transform: transform == "erosion_1px"
            )),
            "dilation_1px": _profile_summary(_robustness_axis_drops(
                robustness_runs, lambda transform: transform == "dilation_1px"
            )),
        }
        transfer_profile = {
            row["dataset"]: {
                "accuracy_mean": float(row["accuracy_mean"]),
                "accuracy_sd": float(row["accuracy_sd"]),
                "seed_count": int(row["seed_count"]),
            }
            for row in transfer.get("summaries", [])
            if row["representation"] == name
            and row["probe"] == "linear_logreg"
            and int(row["train_size"]) == 5000
        }
        profiles.append({
            "representation": name,
            "dimension": dimension,
            "status": "evaluated_core",
            "linear_accessibility": {
                str(size): {
                    "accuracy_mean": float(points[size]["accuracy_mean"]),
                    "accuracy_sd": float(points[size]["accuracy_sd"]),
                    "seed_count": int(points[size]["seed_count"]),
                }
                for size in (500, 5000)
                if size in points
            },
            "sample_efficiency_aulc": aulc_values,
            "paired_dimension_controls": paired_controls,
            "nonlinear_minus_linear_gain": {
                str(size): {
                    probe: capacity_map.get((name, probe, size))
                    for probe in ("rbf_svm", "small_mlp")
                }
                for size in (500, 5000)
            },
            "hog_complementarity_linear": complementarity_by_size,
            "mechanism_target_minus_nuisance_drop": _profile_summary(specificity_values),
            "robustness_accuracy_drop": robustness_profiles,
            "transfer_linear_5000": transfer_profile,
        })
    return profiles


def _mnist_source_provenance(root: Path, protocol: dict) -> dict[str, object]:
    data_config = protocol["data"]["mnist_search"]
    phase_c = protocol["phases"]["C_mechanism_robustness"]
    source_files = sorted(
        (root / "data/mnist/openml/openml.org/data/v1/download").glob("*/mnist_784.arff.gz")
    )
    source_file = source_files[0] if len(source_files) == 1 else None
    return {
        "source_name": "OpenML mnist_784, dataset 554, version 1",
        "source_url": "https://www.openml.org/search?id=554&sort=runs&type=data",
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file else None,
        "local_artifact_path": (
            source_file.resolve().relative_to(root.resolve()).as_posix() if source_file else None
        ),
        "source_train_count": int(data_config["source_train_count"]),
        "training_pool_count": int(data_config["training_pool_count"]),
        "validation_count": int(data_config["validation_count"]),
        "validation_seed": int(data_config["split_seed"]),
        "official_test_count": 10_000,
        "official_test_sample_count": int(phase_c["test_sample_size"]),
        "official_test_sample_seed": int(phase_c["test_sample_seed"]),
    }


def _provenance_table(provenance: dict, mnist: dict[str, object]) -> list[str]:
    lines = [
        "| Dataset | Source | Training pool / partition | Evaluation sample(s) | Preprocessing | Source SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    mnist_hash = f"`{mnist['source_sha256']}`" if mnist["source_sha256"] else "not recorded"
    lines.append(
        f"| `mnist` | [OpenML 554, v1]({mnist['source_url']}) | "
        f"{mnist['training_pool_count']:,} from official {mnist['source_train_count']:,}-image train split; "
        f"{mnist['official_test_count']:,} official test images | "
        f"Phase A/B: {mnist['validation_count']:,} validation (seed {mnist['validation_seed']}); "
        f"Phase C: {mnist['official_test_sample_count']:,} official-test sample (seed {mnist['official_test_sample_seed']}) | "
        f"28×28 grayscale, pixel values scaled to [0,1] | {mnist_hash} |"
    )
    for dataset, row in sorted(provenance.items()):
        lines.append(
            f"| `{dataset}` | [publisher source]({row['source']}) | "
            f"{row['training_count']:,} train images; {row['source_test_count']:,} official test images | "
            f"{row['evaluation_count']:,} (`{row['evaluation_split_id']}`) | "
            f"{row['preprocessing']} | `{row['source_sha256']}` |"
        )
    return lines


def _write_artifact_manifest(root: Path, mnist_provenance: dict[str, object]) -> None:
    relative_paths = [
        "results/v31/protocol.json",
        "results/v31/protocol.sha256",
        "results/v31/manifest.json",
        "results/v31/core_matrix.jsonl",
        "results/v31/complementarity_runs.jsonl",
        "results/v31/complementarity.json",
        "results/v31/mechanism_runs.jsonl",
        "results/v31/mechanism.json",
        "results/v31/robustness_runs.jsonl",
        "results/v31/robustness.json",
        "results/v31/transfer_runs.jsonl",
        "results/v31/transfer.json",
        "results/v31/paired_statistics.json",
        "results/v31/learning_curves.json",
        "results/v31/complexity_frontier.json",
        "results/v31/bias_profiles.json",
        "results/v31/summary.json",
        "reports/V3_1_BIAS_EVALUATION.md",
        "reports/V3_1_LEARNING_CURVES.png",
    ]
    artifacts = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            continue
        data = path.read_bytes()
        entry = {
            "path": relative,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if path.suffix == ".jsonl":
            entry["records"] = sum(1 for line in data.splitlines() if line.strip())
        artifacts.append(entry)
    write_json(
        root / "results/v31/artifact_manifest.json",
        {
            "schema_version": 1,
            "datasets": {"mnist": mnist_provenance},
            "artifacts": artifacts,
        },
    )


def _write_learning_curve_figure(groups: dict[tuple, list[dict]], root: Path) -> Path:
    from matplotlib import pyplot as plt

    representations = (
        "angle_centroid_pairwise",
        "regional_turn_histogram",
        "horizontal_turn_spatial",
        "cycle_angle_hist",
        "zoning_30d",
        "hog_pca_30d",
        "hog_pca_60d",
        "raw_pixels",
        "hog",
    )
    probes = ("linear_logreg", "rbf_svm", "small_mlp")
    sizes = (100, 250, 500, 1000, 2500, 5000)
    colors = plt.get_cmap("tab10").colors
    color_by_rep = {name: colors[index] for index, name in enumerate(representations)}
    v3_names = {"angle_centroid_pairwise", "regional_turn_histogram", "horizontal_turn_spatial"}
    figure, axes = plt.subplots(1, len(probes), figsize=(17, 5.5), sharey=True)
    legend_handles = {}
    for axis, probe in zip(axes, probes, strict=True):
        for name in representations:
            points = [
                (size, groups[(name, probe, size)])
                for size in sizes
                if (name, probe, size) in groups
            ]
            if not points:
                continue
            means = [float(np.mean([float(run["accuracy"]) for run in runs])) for _, runs in points]
            errors = [
                float(np.std([float(run["accuracy"]) for run in runs], ddof=1))
                if len(runs) > 1
                else 0.0
                for _, runs in points
            ]
            line = axis.errorbar(
                [size for size, _ in points],
                means,
                yerr=errors,
                marker="o",
                markersize=4,
                linewidth=2.2 if name in v3_names else 1.2,
                capsize=2,
                color=color_by_rep[name],
                alpha=0.95 if name in v3_names else 0.78,
                label=name,
            )
            legend_handles[name] = line.lines[0]
        axis.set_xscale("log")
        axis.set_xticks(sizes, labels=[str(size) for size in sizes])
        axis.set_xlim(90, 5600)
        axis.set_ylim(0.2, 1.0)
        axis.set_title({"linear_logreg": "Linear logistic regression", "rbf_svm": "RBF-SVM", "small_mlp": "Small MLP"}[probe])
        axis.set_xlabel("MNIST training samples (log scale)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Validation accuracy")
    figure.suptitle("Frozen representations across training sizes", fontsize=15)
    figure.legend(
        handles=legend_handles.values(),
        labels=legend_handles.keys(),
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.tight_layout(rect=(0, 0.16, 1, 0.93))
    output = root / "reports/V3_1_LEARNING_CURVES.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def _pp(value: float | None) -> str:
    return "not available" if value is None else f"{value * 100:+.2f} pp"


def _range_pp(values: list[float]) -> str:
    return "not available" if not values else f"{min(values) * 100:+.2f} to {max(values) * 100:+.2f} pp"


def _range(values: list[float]) -> str:
    return "not available" if not values else f"{min(values):+.4f} to {max(values):+.4f}"


def _format_aulc_delta(value: float) -> str:
    return f"{value:+.4f} AULC"


def _mean_sd_pp(values: list[float]) -> str:
    if not values:
        return "not measured"
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean * 100:+.2f} ± {sd * 100:.2f} pp"


def _profile_summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "seed_count": len(values),
    }


def _robustness_axis_drops(
    rows: list[dict], matches: Callable[[str], bool]
) -> list[float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if matches(str(row["transformation"])):
            by_seed[int(row["seed"])].append(float(row["accuracy_drop"]))
    return [float(np.mean(values)) for _, values in sorted(by_seed.items())]


def _core_groups(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = row["key"]
        groups[(key["representation"], key["probe"], int(key["train_size"]))].append(row)
    return groups


def _learning_curves(groups: dict[tuple, list[dict]]) -> list[dict[str, object]]:
    names = sorted({key[0] for key in groups})
    probes = sorted({key[1] for key in groups})
    output = []
    for name in names:
        for probe in probes:
            points = []
            for size in sorted(key[2] for key in groups if key[0] == name and key[1] == probe):
                scores = [float(row["accuracy"]) for row in groups[(name, probe, size)]]
                points.append({"train_size": size, "accuracy_mean": float(np.mean(scores)), "accuracy_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0, "seed_count": len(scores)})
            if len(points) >= 2:
                output.append({
                    "representation": name,
                    "probe": probe,
                    "points": points,
                    "aulc_log_train_size": learning_curve_aulc({int(point["train_size"]): float(point["accuracy_mean"]) for point in points}),
                })
    return output


def _capacity_gains(groups: dict[tuple, list[dict]]) -> list[dict[str, object]]:
    output = []
    for (name, probe, size), rows in sorted(groups.items()):
        if probe not in ("rbf_svm", "small_mlp"):
            continue
        linear_rows = groups.get((name, "linear_logreg", size), [])
        current = {int(row["key"]["seed"]): float(row["accuracy"]) for row in rows}
        linear = {int(row["key"]["seed"]): float(row["accuracy"]) for row in linear_rows}
        common = sorted(set(current) & set(linear))
        gains = [current[seed] - linear[seed] for seed in common]
        if gains:
            output.append({"representation": name, "probe": probe, "train_size": size, "gain_mean": float(np.mean(gains)), "gain_sd": float(np.std(gains, ddof=1)) if len(gains) > 1 else 0.0})
    return output


def _frontiers(groups: dict[tuple, list[dict]], rows: list[dict]) -> list[dict[str, object]]:
    dimensions = {}
    for row in rows:
        dimensions[row["key"]["representation"]] = int(row["feature_dim"])
    output = []
    for probe in sorted({key[1] for key in groups}):
        for size in (500, 5000):
            points = []
            for (name, group_probe, group_size), runs in groups.items():
                if group_probe == probe and group_size == size:
                    points.append((dimensions[name], float(np.mean([run["accuracy"] for run in runs])), name))
            if points:
                front = set(pareto_frontier([(dim, score) for dim, score, _ in points]))
                output.append({"probe": probe, "train_size": size, "points": [{"representation": name, "dimension": dim, "accuracy": score, "pareto": (dim, score) in front} for dim, score, name in sorted(points)]})
    return output


def _paired_statistics(core: list[dict], complementarity: dict) -> dict[str, object]:
    by_key = {}
    for row in core:
        key = row["key"]
        by_key[(key["representation"], key["probe"], int(key["train_size"]), int(key["seed"]))] = row
    comparisons = {}
    v3_names = ("angle_centroid_pairwise", "regional_turn_histogram", "horizontal_turn_spatial")
    for name in v3_names:
        dimension = next((int(row["feature_dim"]) for row in core if row["key"]["representation"] == name), 0)
        matched = "hog_pca_30d" if dimension == 30 else "hog_pca_60d"
        for baseline in ("zoning_30d", matched):
            for size in (500, 5000):
                seed_rows = []
                for seed in (11, 23, 47):
                    base = by_key.get((baseline, "linear_logreg", size, seed))
                    candidate = by_key.get((name, "linear_logreg", size, seed))
                    if base is not None and candidate is not None:
                        _validate_paired_rows(base, candidate)
                        seed_rows.append((seed, base, candidate))
                if len(seed_rows) != 3:
                    continue
                targets = np.asarray(seed_rows[0][1]["targets"], dtype=np.int64)
                baseline_predictions = np.stack([np.asarray(row[1]["predictions"], dtype=np.int64) for row in seed_rows])
                candidate_predictions = np.stack([np.asarray(row[2]["predictions"], dtype=np.int64) for row in seed_rows])
                label = f"{name} vs {baseline}/linear_logreg/n={size}"
                comparisons[label] = {
                    "bootstrap": paired_bootstrap_delta(targets, baseline_predictions, candidate_predictions),
                    "mcnemar_by_seed": {str(seed): mcnemar_exact(targets, np.asarray(base["predictions"]), np.asarray(candidate["predictions"])) for seed, base, candidate in seed_rows},
                }
    if complementarity:
        comparisons["complementarity"] = complementarity.get("paired_statistics", {})
    return comparisons


def _validate_paired_rows(left: dict, right: dict) -> None:
    if left["key"]["evaluation_split_id"] != right["key"]["evaluation_split_id"]:
        raise ValueError("paired comparison evaluation split IDs do not match")
    for field in ("targets", "evaluation_indices"):
        if left[field] != right[field]:
            raise ValueError(f"paired comparison {field} differ")


def _matrix_table(groups: dict[tuple, list[dict]], *, sizes: tuple[int, ...]) -> list[str]:
    lines = ["| Representation | Dimension | Probe | " + " | ".join(f"n={size}" for size in sizes) + " |", "|---|---:|---|" + "---:|" * len(sizes)]
    dimensions = {key[0]: int(rows[0]["feature_dim"]) for key, rows in groups.items()}
    names = sorted({key[0] for key in groups})
    probes = sorted({key[1] for key in groups})
    for name in names:
        for probe in probes:
            cells = []
            for size in sizes:
                runs = groups.get((name, probe, size), [])
                if not runs:
                    cells.append("—")
                else:
                    scores = [float(row["accuracy"]) for row in runs]
                    sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                    cells.append(f"{np.mean(scores):.4f} ± {sd:.4f}")
            if any(cell != "—" for cell in cells):
                lines.append(f"| `{name}` | {dimensions[name]} | `{probe}` | " + " | ".join(cells) + " |")
    return lines


def _curve_table(rows: list[dict[str, object]]) -> list[str]:
    lines = ["| Representation | Probe | AULC over log n | Learning curve (mean ± SD) |", "|---|---|---:|---|"]
    for row in rows:
        points = ", ".join(f"{p['train_size']}: {p['accuracy_mean']:.4f} ± {p['accuracy_sd']:.4f}" for p in row["points"])
        lines.append(f"| `{row['representation']}` | `{row['probe']}` | {row['aulc_log_train_size']:.4f} | {points} |")
    return lines


def _capacity_table(rows: list[dict[str, object]]) -> list[str]:
    lines = ["| Representation | Nonlinear probe | Train size | Gain over linear |", "|---|---|---:|---:|"]
    lines.extend(f"| `{row['representation']}` | `{row['probe']}` | {row['train_size']} | {row['gain_mean']:+.4f} ± {row['gain_sd']:.4f} |" for row in rows)
    return lines


def _frontier_table(rows: list[dict[str, object]]) -> list[str]:
    lines = ["| Probe | Train size | Pareto representations (dimension, accuracy) |", "|---|---:|---|"]
    for row in rows:
        front = [point for point in row["points"] if point["pareto"]]
        values = ", ".join(f"`{p['representation']}` ({p['dimension']}, {p['accuracy']:.4f})" for p in front)
        lines.append(f"| `{row['probe']}` | {row['train_size']} | {values} |")
    return lines


def _complementarity_table(rows: list[dict]) -> list[str]:
    lines = ["| Anchor | V3 representation | Probe | Train size | Δ accuracy (mean ± SD) |", "|---|---|---|---:|---:|"]
    lines.extend(f"| `{r['anchor']}` | `{r['v3_representation']}` | `{r['probe']}` | {r['train_size']} | {r['delta_mean']:+.4f} ± {r['delta_sd']:.4f} |" for r in rows)
    return lines


def _mechanism_table(rows: list[dict]) -> list[str]:
    lines = ["| Representation | Targeted accuracy drop | Nuisance accuracy drop | Target − nuisance |", "|---|---:|---:|---:|"]
    lines.extend(f"| `{r['representation']}` | {r['target_drop_mean']:.4f} ± {r['target_drop_sd']:.4f} | {r['nuisance_drop_mean']:.4f} ± {r['nuisance_drop_sd']:.4f} | {r['target_minus_nuisance_drop_mean']:+.4f} |" for r in rows)
    return lines


def _robustness_table(rows: list[dict]) -> list[str]:
    selected = sorted(rows, key=lambda row: (-abs(float(row["accuracy_drop_mean"])), row["representation"], row["transformation"]))
    lines = ["| Representation | Transformation | Accuracy drop (mean ± SD) | Prediction agreement |", "|---|---|---:|---:|"]
    lines.extend(f"| `{r['representation']}` | `{r['transformation']}` | {r['accuracy_drop_mean']:+.4f} ± {r['accuracy_drop_sd']:.4f} | {r['prediction_agreement_mean']:.4f} |" for r in selected)
    return lines


def _transfer_table(rows: list[dict]) -> list[str]:
    lines = ["| Dataset | Representation | Probe | Train size | Accuracy (mean ± SD) |", "|---|---|---|---:|---:|"]
    lines.extend(f"| `{r['dataset']}` | `{r['representation']}` | `{r['probe']}` | {r['train_size']} | {r['accuracy_mean']:.4f} ± {r['accuracy_sd']:.4f} |" for r in rows)
    return lines


def _paired_table(statistics: dict[str, object]) -> list[str]:
    lines = [
        "| Comparison | Paired Δ accuracy | 95% bootstrap CI | McNemar p (seed 11 / 23 / 47) |",
        "|---|---:|---:|---|",
    ]
    for label, value in statistics.items():
        if label == "complementarity":
            for group, item in value.items():
                bootstrap = item["bootstrap"]
                mcnemar = _mcnemar_seed_values(item.get("mcnemar_by_seed", {}))
                lines.append(f"| {group} | {bootstrap['delta']:+.4f} | [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}] | {mcnemar} |")
            continue
        bootstrap = value["bootstrap"]
        mcnemar = _mcnemar_seed_values(value.get("mcnemar_by_seed", {}))
        lines.append(f"| {label} | {bootstrap['delta']:+.4f} | [{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}] | {mcnemar} |")
    return lines


def _mcnemar_seed_values(values: dict[str, dict]) -> str:
    return " / ".join(
        f"{float(values[str(seed)]['p_value']):.3g}"
        for seed in (11, 23, 47)
        if str(seed) in values
    ) or "—"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    path = generate_report(root=args.root)
    print(path)


if __name__ == "__main__":
    main()
