"""The second evaluation stage selects candidates by their 500-sample scores."""

import hashlib
import json

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.novelty.descriptors import describe_program
from experiments.evaluate_v3_finalists import evaluate_finalists
from experiments.freeze_v3_finalists import freeze_finalists


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _record(
    name: str,
    accuracy: float,
    program: Expr,
    *,
    generation: int = 1,
) -> ProgramSearchRecord:
    descriptor = describe_program(program)
    bias = ProgramBiasSpec(
        name=name,
        hypothesis="A compact structure may support digit recognition.",
        mechanism="The expression preserves a measurable image relationship.",
        program=program,
        prediction="The program should improve low-data validation accuracy.",
        falsification="Reject it if the validation score remains at chance.",
    )
    return ProgramSearchRecord(
        generation=generation,
        track="discovery",
        bias=bias,
        evaluation=Evaluation(
            accuracy_500=accuracy,
            accuracy_5000=None,
            feature_dim=6,
            feature_runtime_ms=1,
            training_runtime_ms=1,
            inference_runtime_ms=1,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        ),
        source=descriptor.source,
        order=descriptor.order,
        spatial=descriptor.spatial,
        composition=descriptor.composition,
        complexity=descriptor.complexity,
        novelty=1.0,
    )


class _Evaluator:
    def __init__(self) -> None:
        self.names: list[str] = []

    def evaluate(self, bias: ProgramBiasSpec) -> Evaluation:
        self.names.append(bias.name)
        return Evaluation(
            accuracy_500=0.82,
            accuracy_5000=0.91,
            feature_dim=6,
            feature_runtime_ms=2,
            training_runtime_ms=3,
            inference_runtime_ms=1,
            confusion_matrix=np.zeros((10, 10), dtype=np.int64),
        )


def test_only_top_stage_one_candidates_get_5000_sample_evaluation(tmp_path) -> None:
    image = _node("image")
    graph = _node("graph", _node("skeletonize", image))
    paths = _node("paths", graph)
    records = (
        _record("low", 0.6, _node("cycle_rank", graph)),
        _record("high", 0.9, _node("histogram", _node("angles", paths), bins=6)),
        _record(
            "middle", 0.8, _node("autocorrelation", _node("degree_sequence", graph))
        ),
        _record(
            "seed",
            0.85,
            _node("count", graph),
            generation=0,
        ),
    )
    archive = tmp_path / "archive.jsonl"
    archive.write_text("\n".join(record.to_json() for record in records) + "\n")
    output = tmp_path / "finalists.json"
    evaluator = _Evaluator()

    summary = evaluate_finalists(archive, output, count=3, evaluator=evaluator)

    assert evaluator.names == ["high", "seed", "middle"]
    assert summary["evaluated_train_sizes"] == [500, 5_000]
    assert [item["bias"]["name"] for item in summary["evaluations"]] == [
        "high",
        "seed",
        "middle",
    ]
    assert (
        summary["search_archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert json.loads(output.read_text())["finalist_count"] == 3

    manifest_path = tmp_path / "frozen.json"
    manifest = freeze_finalists(archive, output, manifest_path, count=1)

    assert manifest["finalist_selection_frozen"] is True
    assert manifest["test_set_accessed"] is False
    assert manifest["finalist_count"] == 1
    finalist = manifest["finalists"][0]
    assert finalist["expected_transfer_behavior"]["fashion_mnist"].startswith(
        "Negative-control domain"
    )
    assert "expected_counterfactual" in finalist
    assert (
        json.loads(manifest_path.read_text())["search_archive_sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
