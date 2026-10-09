"""V3 transfer comparisons include the predeclared classical and V2 controls."""

import hashlib
import json

from bias_optimizer.domain.program import ProgramBiasSpec, program_bias_hash
from bias_optimizer.dsl.ast import Expr
from experiments.evaluate_v3_transfer import _representations


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _program_bias(name: str) -> ProgramBiasSpec:
    image = _node("image")
    paths = _node("paths", _node("graph", _node("skeletonize", image)))
    return ProgramBiasSpec(
        name=name,
        hypothesis="A compact stroke representation may generalize across glyphs.",
        mechanism="Path geometry preserves a structural property of handwriting.",
        program=_node("histogram", _node("angles", paths), bins=6),
        prediction="The structural representation should improve glyph recognition.",
        falsification="Reject it if matched transfer accuracy does not improve.",
    )


def test_transfer_representations_include_required_baselines(tmp_path) -> None:
    v3_bias = _program_bias("v3_best")
    v2_bias = _program_bias("spatial_cycle_hist")
    v2_archive = tmp_path / "v2_archive.jsonl"
    v2_archive.write_text("frozen v2 archive\n")
    v2_manifest = tmp_path / "v2.json"
    v2_manifest.write_text(
        json.dumps(
            {
                "finalist_selection_frozen": True,
                "test_set_accessed": False,
                "search_archive": str(v2_archive),
                "search_archive_sha256": hashlib.sha256(
                    v2_archive.read_bytes()
                ).hexdigest(),
                "finalists": [
                    {
                        "name": "spatial_cycle_hist",
                        "candidate_id": program_bias_hash(v2_bias),
                        "bias": v2_bias.to_dict(),
                    }
                ],
            }
        )
    )

    representations = _representations(
        [
            {
                "name": v3_bias.name,
                "candidate_id": program_bias_hash(v3_bias),
                "bias": v3_bias.to_dict(),
            }
        ],
        v2_manifest_path=v2_manifest,
    )

    assert {item[1] for item in representations} == {
        "v3_best",
        "human_topology",
        "human_topology_spatial",
        "raw_pixels",
        "hog_9bin_4x4",
        "zoning_4x4",
        "zoning_5x6",
        "spatial_cycle_hist_v2_best",
    }
    assert all(
        len(item[0]) == 64
        and all(character in "0123456789abcdef" for character in item[0])
        for item in representations
    )
    assert {item[2] for item in representations} == {
        "v3_discovered",
        "human_structural_control",
        "classical_baseline",
        "compact_classical_baseline",
        "v2_best",
    }
