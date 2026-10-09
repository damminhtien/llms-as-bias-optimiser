"""Deterministic, typed seed programs for initializing either search track."""

from __future__ import annotations

import random

from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op=op, args=tuple(args), params=params)


def generate_seed_programs(
    count: int = 20,
    *,
    seed: int = 42,
    track: SearchTrack = SearchTrack.DISCOVERY,
    max_feature_dim: int | None = None,
) -> tuple[ProgramBiasSpec, ...]:
    """Build a reproducible pool of valid programs from small AST templates."""
    if type(count) is not int or count <= 0:
        raise ValueError("count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    track = SearchTrack(track)
    dimension_limit = (
        max_feature_dim
        if max_feature_dim is not None
        else (128 if track is SearchTrack.DISCOVERY else 1024)
    )
    compiler = ProgramCompiler(
        ProgramConstraints(
            track=track,
            max_feature_dim=dimension_limit,
            forbid_raw_pixels=track is SearchTrack.AUGMENTATION,
            require_vector_root=track is SearchTrack.AUGMENTATION,
        )
    )
    image = _node("image")
    pool: list[tuple[Expr, str]] = []
    ordered_seed_pool: list[tuple[Expr, str]] = []

    seed_skeleton = _node("skeletonize", image)
    seed_graph = _node("graph", seed_skeleton)
    seed_paths = _node("paths", seed_graph)
    seed_turns = _node("delta_angle", _node("angles", seed_paths))
    ordered_seed_pool.append(
        (
            _node("autocorrelation", seed_turns, lags=[1, 2, 4]),
            "Turn autocorrelation tests whether neighboring curvature changes persist along a path.",
        )
    )
    degree_sequence = _node("degree_sequence", seed_graph)
    degree_runs = _node("run_length_encode", degree_sequence)
    ordered_seed_pool.extend(
        (
            (
                _node("histogram", degree_runs, bins=6, low=-8, high=8),
                "Graph-degree run lengths test whether local branching states persist in traversal order.",
            ),
            (
                _node("moments", degree_runs, orders=[1, 2, 3]),
                "Moments of graph-degree run lengths test higher-order persistence in the ordered graph trace.",
            ),
        )
    )

    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        binary = _node("threshold", image, value=threshold)
        skeleton = _node("skeletonize", image, threshold=threshold)
        graph = _node("graph", skeleton)
        paths = _node("paths", graph)
        degree = _node("degree_sequence", graph)
        angles = _node("angles", paths)
        turns = _node("delta_angle", angles)
        components = _node("connected_components", binary)
        cycles = _node("cycle_rank", graph)
        pool.extend(
            (
                (
                    components,
                    "Foreground component count tests a compact topological cue.",
                ),
                (
                    cycles,
                    "Graph cycle rank tests whether loop structure predicts identity.",
                ),
            )
        )

        for rows, cols in ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3)):
            pool.append(
                (
                    _node("spatial_split", image, rows=rows, cols=cols),
                    "Coarse spatial ink distribution tests location-sensitive structure.",
                )
            )

        for bins in (4, 6, 8, 12, 16):
            pool.append(
                (
                    _node("histogram", degree, bins=bins, low=0, high=16),
                    "The graph degree distribution tests skeleton branching structure.",
                )
            )
            pool.append(
                (
                    _node("histogram", turns, bins=bins, low=-3.141593, high=3.141593),
                    "The turning-angle distribution tests local stroke geometry.",
                )
            )
        pool.extend(
            (
                (
                    _node("moments", degree, orders=[1, 2, 3]),
                    "Moments summarize graph degree structure.",
                ),
                (
                    _node("quantiles", degree, quantiles=[0.25, 0.5, 0.75]),
                    "Degree quantiles test robust branching summaries.",
                ),
                (
                    _node("autocorrelation", degree, lags=[1, 2, 4]),
                    "Degree-sequence autocorrelation tests ordering in the graph traversal.",
                ),
                (
                    _node("moments", turns, orders=[1, 2, 3]),
                    "Turning moments test aggregate stroke dynamics.",
                ),
                (
                    _node("autocorrelation", turns, lags=[1, 2, 4]),
                    "Turn autocorrelation tests persistence across neighboring path steps.",
                ),
                (
                    _node(
                        "histogram",
                        _node("edge_lengths", paths),
                        bins=8,
                        low=0,
                        high=2.0,
                    ),
                    "Edge-length frequencies test local stroke geometry.",
                ),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected: list[tuple[Expr, str]] = []
    seen: set[str] = set()
    for expr, mechanism in (*ordered_seed_pool, *pool):
        signature = expr.to_json()
        if signature in seen:
            continue
        try:
            compiler.compile(expr)
        except TypeError, ValueError:
            continue
        seen.add(signature)
        selected.append((expr, mechanism))
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(
            f"seed grammar produced only {len(selected)} distinct valid programs"
        )

    return tuple(
        ProgramBiasSpec(
            name=f"typed_seed_{index:03d}",
            hypothesis=(
                "This typed representation may encode a compact structural cue "
                "that helps classify handwritten digits from limited labels."
            ),
            mechanism=mechanism,
            program=expr,
            prediction="It should improve validation accuracy or reduce model size versus nearby alternatives.",
            falsification="Reject it if validation quality does not improve after complexity is accounted for.",
        )
        for index, (expr, mechanism) in enumerate(selected)
    )


def generate_v3_pilot_seed_programs() -> tuple[ProgramBiasSpec, ...]:
    """Return ten explicit pilot programs spanning V3 mechanisms and controls."""
    image = _node("image")
    skeleton = _node("skeletonize", image)
    graph = _node("graph", skeleton)
    paths = _node("paths", graph)
    angles = _node("angles", paths)
    turns = _node("delta_angle", angles)
    degree = _node("degree_sequence", graph)
    degree_runs = _node("run_length_encode", degree)

    programs = (
        (
            _node("autocorrelation", turns, lags=[1, 2, 4]),
            "Turn autocorrelation tests whether adjacent curvature changes persist along paths.",
        ),
        (
            _node("histogram", degree_runs, bins=6, low=-8, high=8),
            "Graph-degree run lengths test whether branching states persist in traversal order.",
        ),
        (
            _node("moments", degree_runs, orders=[1, 2, 3]),
            "Moments of graph-degree runs summarize ordered branch persistence.",
        ),
        (
            _node(
                "spatial_condition",
                turns,
                axis="vertical",
                regions=3,
                bins=8,
                low=-3.141593,
                high=3.141593,
            ),
            "Turn changes conditioned on vertical region test curvature-by-height structure.",
        ),
        (
            _node(
                "spatial_condition",
                angles,
                axis="vertical",
                regions=3,
                bins=8,
                low=-3.141593,
                high=3.141593,
            ),
            "Stroke orientation conditioned on vertical region tests location-specific geometry.",
        ),
        (
            _node(
                "cross_histogram",
                _node("path_summary", paths, measure="is_loop"),
                _node("path_summary", paths, measure="centroid_y"),
                bins_x=2,
                bins_y=4,
                low_x=0,
                high_x=1,
                low_y=0,
                high_y=1,
            ),
            "Loop presence paired with centroid height tests loop-position interaction.",
        ),
        (
            _node(
                "cross_histogram",
                _node("path_summary", paths, measure="branch_endpoints"),
                _node("path_summary", paths, measure="centroid_y"),
                bins_x=3,
                bins_y=4,
                low_x=0,
                high_x=2,
                low_y=0,
                high_y=1,
            ),
            "Branch endpoint counts paired with centroid height test branch-location interaction.",
        ),
        (
            _node(
                "histogram",
                _node("path_summary", paths, measure="branch_endpoints"),
                bins=4,
                low=0,
                high=2,
            ),
            "The branch endpoint distribution provides a graph-structure control.",
        ),
        (
            _node("cycle_rank", graph),
            "Cycle rank provides a compact topological control.",
        ),
        (
            _node(
                "histogram",
                _node("edge_lengths", paths),
                bins=8,
                low=0,
                high=2,
            ),
            "Edge-length frequencies provide a global path-geometry control.",
        ),
    )

    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.DISCOVERY,
            max_feature_dim=128,
        )
    )
    seen: set[str] = set()
    seeds: list[ProgramBiasSpec] = []
    for index, (program, mechanism) in enumerate(programs):
        compiler.compile(program)
        signature = program.to_json()
        if signature in seen:
            raise ValueError("V3 pilot seed programs must be structurally distinct")
        seen.add(signature)
        seeds.append(
            ProgramBiasSpec(
                name=f"v3_pilot_seed_{index + 1:02d}",
                hypothesis="A compact relational or structural signal may support few-shot digit recognition.",
                mechanism=mechanism,
                program=program,
                prediction="The representation should improve validation prediction or transfer beyond a matched compact control.",
                falsification="Reject it if a mechanism-specific intervention leaves its predictive behavior unchanged.",
            )
        )
    return tuple(seeds)
