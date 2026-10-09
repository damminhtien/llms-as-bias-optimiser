"""The pilot auditor uses the predeclared diversity threshold."""

import json

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.novelty.descriptors import describe_program
from experiments.summarize_v3_pilot import summarize_pilot


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _record(index: int, generation: int, program: Expr) -> ProgramSearchRecord:
    bias = ProgramBiasSpec(
        name=f"pilot_{index}",
        hypothesis="A structural cue may improve recognition.",
        mechanism="The AST preserves a measurable property of the image.",
        program=program,
        prediction="Validation predictions should improve for related classes.",
        falsification="Reject the cue if the matched intervention changes nothing.",
    )
    descriptor = describe_program(program)
    return ProgramSearchRecord(
        generation=generation,
        track="discovery",
        bias=bias,
        evaluation=Evaluation(
            accuracy_500=0.75,
            accuracy_5000=0.8,
            feature_dim=8,
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


def test_pilot_auditor_applies_the_30_percent_gate_and_counts_failures(
    tmp_path,
) -> None:
    image = _node("image")
    skeleton = _node("skeletonize", image)
    graph = _node("graph", skeleton)
    paths = _node("paths", graph)
    angle_histogram = _node("histogram", _node("angles", paths), bins=6)
    records = (
        _record(0, 0, angle_histogram),
        _record(1, 0, _node("cycle_rank", graph)),
        _record(2, 1, _node("spatial_split", image, rows=2, cols=3)),
        _record(3, 1, angle_histogram),
    )
    archive = tmp_path / "pilot.jsonl"
    archive.write_text("\n".join(record.to_json() for record in records) + "\n")
    response = {
        "prompt": (
            "Propose exactly 2 distinct representation programs.\n"
            "Requested mechanism family for this batch: order_sensitive."
        ),
        "response": json.dumps(
            {
                "proposals": [
                    records[0].bias.to_dict(),
                    {"name": "invalid"},
                ]
            }
        ),
    }
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(response) + "\n")

    summary = summarize_pilot(
        archive,
        responses,
        expected_seeds=2,
        expected_generations=1,
        candidates_per_generation=2,
    )

    assert summary["decision"] == "GO"
    assert summary["outside_global_histogram_fraction"] == 0.5
    assert summary["invalid_ast_count"] == 1
    assert summary["duplicate_proposal_count"] == 1
    assert summary["proposal_attempt_count"] == 2
    assert len(summary["archive_sha256"]) == 64
    assert len(summary["response_archive_sha256"]) == 64
    assert summary["requested_mechanism_counts"] == {"order_sensitive": 2}
