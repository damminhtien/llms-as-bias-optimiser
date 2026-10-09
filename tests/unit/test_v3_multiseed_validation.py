"""Frozen finalist validation can be repeated under fixed learner seeds."""

import hashlib
import json

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.ast import Expr
from experiments.evaluate_v3_multiseed_validation import evaluate_multiseed_validation


def _bias() -> ProgramBiasSpec:
    return ProgramBiasSpec(
        name="cycle_rank",
        hypothesis="Loop structure may encode digit identity.",
        mechanism="The feature measures skeleton graph cycle rank.",
        program=Expr(
            "cycle_rank",
            (Expr("graph", (Expr("skeletonize", (Expr("image"),)),)),),
        ),
        prediction="Loop-bearing digits should be easier to distinguish.",
        falsification="Reject the mechanism if the loop statistic adds no value.",
    )


class _FakeEvaluator:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def evaluate(self, _bias: ProgramBiasSpec) -> Evaluation:
        score = self.seed / 100.0
        return Evaluation(
            accuracy_500=score,
            accuracy_5000=score + 0.1,
            feature_dim=1,
            feature_runtime_ms=2.0,
            training_runtime_ms=1.0,
            inference_runtime_ms=1.0,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        )


def test_multiseed_runner_records_fixed_validation_scores_without_test_access(
    tmp_path,
) -> None:
    archive = tmp_path / "archive.jsonl"
    archive.write_text("frozen candidates\n", encoding="utf-8")
    bias = _bias()
    manifest = {
        "finalist_selection_frozen": True,
        "search_archive": str(archive),
        "search_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "finalists": [
            {"candidate_id": program_bias_hash(bias), "bias": bias.to_dict()}
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = evaluate_multiseed_validation(
        manifest_path,
        tmp_path / "multi.json",
        seeds=(11, 23, 47),
        evaluator_factory=_FakeEvaluator,
    )

    assert result["mnist_test_accessed"] is False
    assert [item["seed"] for item in result["results"][0]["evaluations"]] == [
        11,
        23,
        47,
    ]
    assert result["results"][0]["summary"]["accuracy_500_mean"] == 0.27
