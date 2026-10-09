"""Behavioral descriptor coverage and MAP-Elites cell separation."""

import numpy as np

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.domain.program_search import ProgramSearchRecord
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.novelty.descriptors import describe_program
from bias_optimizer.novelty.map_elites import MapElitesArchive


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _paths() -> Expr:
    return _node("paths", _node("graph", _node("skeletonize", _node("image"))))


def _handcrafted_programs() -> tuple[Expr, ...]:
    paths = _paths()
    angles = _node("angles", paths)
    turns = _node("delta_angle", angles)
    degree = _node(
        "degree_sequence", _node("graph", _node("skeletonize", _node("image")))
    )
    spatial_angles = _node("spatial_condition", angles)
    spatial_lengths = _node("spatial_condition", _node("edge_lengths", _paths()))
    spatial_turns = _node("spatial_condition", turns)
    pixels = _node("spatial_split", _node("image"), rows=2, cols=2)

    return (
        _node("cycle_rank", _node("graph", _node("skeletonize", _node("image")))),
        _node("histogram", degree, bins=6, low=0, high=12),
        _node("normalize", _node("autocorrelation", degree)),
        _node("histogram", angles, bins=6, low=-3.141593, high=3.141593),
        _node("histogram", turns, bins=6, low=-3.141593, high=3.141593),
        _node("autocorrelation", turns, lags=[1, 2, 4]),
        _node("normalize", spatial_turns),
        _node("cross_histogram", spatial_angles, spatial_lengths),
        pixels,
        _node("concat", pixels, _node("histogram", angles, bins=6)),
        _node(
            "concat",
            _node(
                "spatial_condition",
                _node("run_length_encode", _node("sign", turns)),
            ),
            pixels,
        ),
        _node(
            "spatial_condition",
            _node("path_summary", _paths(), measure="is_loop"),
        ),
    )


def _record(index: int, program: Expr) -> ProgramSearchRecord:
    bias = ProgramBiasSpec(
        name=f"descriptor_{index}",
        hypothesis="This compact statistic may encode a class-relevant structure.",
        mechanism="The representation preserves a targeted structural relation.",
        program=program,
        prediction="It should separate at least one visually similar digit pair.",
        falsification="Reject the mechanism if its matched intervention changes no predictions.",
    )
    descriptor = describe_program(program)
    return ProgramSearchRecord(
        generation=0,
        track="discovery",
        bias=bias,
        evaluation=Evaluation(
            accuracy_500=0.7 + index / 100,
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


def test_angle_hist_is_orderless_path_geometry() -> None:
    descriptor = describe_program(_node("histogram", _node("angles", _paths()), bins=6))

    assert descriptor.source == "path_geometry"
    assert descriptor.order == "orderless"
    assert descriptor.spatial == "global"
    assert descriptor.composition == "single"


def test_turn_autocorrelation_is_higher_order() -> None:
    turns = _node("delta_angle", _node("angles", _paths()))

    assert describe_program(_node("autocorrelation", turns)).order == "higher_order"


def test_spatial_angle_is_localized() -> None:
    program = _node(
        "spatial_condition", _node("delta_angle", _node("angles", _paths()))
    )

    descriptor = describe_program(program)
    assert descriptor.source == "path_geometry"
    assert descriptor.spatial == "localized"


def test_concat_is_composite() -> None:
    program = _node(
        "concat",
        _node("spatial_split", _node("image")),
        _node("histogram", _node("angles", _paths())),
    )

    descriptor = describe_program(program)
    assert descriptor.source == "mixed"
    assert descriptor.composition == "composite"


def test_handcrafted_suite_populates_twelve_distinct_archive_cells(tmp_path) -> None:
    programs = _handcrafted_programs()
    descriptors = tuple(describe_program(program) for program in programs)
    assert len({descriptor.cell for descriptor in descriptors}) == 12

    archive = MapElitesArchive(tmp_path / "programs.jsonl")
    for index, program in enumerate(programs):
        archive.add(_record(index, program))

    assert len(archive.occupied_cells) == 12
    assert len(archive.elites) == 12
    assert len(MapElitesArchive(tmp_path / "programs.jsonl").occupied_cells) == 12


def test_legacy_v2_search_record_keeps_its_report_niche() -> None:
    record = _record(0, _handcrafted_programs()[0])
    payload = record.to_dict()
    payload["niche"] = "topology"
    payload["complexity_bin"] = payload["descriptor"]["complexity"]
    del payload["descriptor"]

    restored = ProgramSearchRecord.from_dict(payload)

    assert restored.niche == "topology"
    assert restored.descriptor == record.descriptor
