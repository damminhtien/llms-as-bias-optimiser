"""The V3 report keeps the best candidate in each five-axis cell."""

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.novelty.descriptors import describe_program
from experiments.summarize_v3_search import summarize_search


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _record(name: str, accuracy: float, program: Expr) -> ProgramSearchRecord:
    descriptor = describe_program(program)
    bias = ProgramBiasSpec(
        name=name,
        hypothesis="A compact signal may distinguish visual categories.",
        mechanism="The program preserves a stable structural property.",
        program=program,
        prediction="The representation should improve validation accuracy.",
        falsification="Reject it if its validation score does not improve.",
    )
    return ProgramSearchRecord(
        generation=1,
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
        novelty=0.5,
    )


def test_search_summary_selects_best_per_descriptor_cell(tmp_path) -> None:
    image = _node("image")
    low = _node(
        "cycle_rank", _node("graph", _node("skeletonize", image, threshold=0.4))
    )
    high = _node(
        "cycle_rank", _node("graph", _node("skeletonize", image, threshold=0.7))
    )
    geometry = _node(
        "histogram",
        _node("angles", _node("paths", _node("graph", _node("skeletonize", image)))),
        bins=6,
    )
    records = (
        _record("low", 0.5, low),
        _record("high", 0.8, high),
        _record("geometry", 0.7, geometry),
    )
    archive = tmp_path / "search.jsonl"
    archive.write_text("\n".join(record.to_json() for record in records) + "\n")

    summary = summarize_search(archive)

    assert summary["candidate_count"] == 3
    assert summary["descriptor_cell_count"] == 2
    best_names = {item["name"] for item in summary["best_per_cell"]}
    assert best_names == {"high", "geometry"}
    assert all("program" in item for item in summary["best_per_cell"])
    assert summary["finalist_evaluations"] is None
