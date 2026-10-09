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
        ProgramConstraints(track=track, max_feature_dim=dimension_limit)
    )
    image = _node("image")
    pool: list[tuple[Expr, str]] = []

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

    if track is SearchTrack.AUGMENTATION:
        pixels = _node("flatten_pixels", image)
        for rows, cols in ((1, 3), (2, 2), (2, 3), (3, 3)):
            spatial = _node("spatial_split", image, rows=rows, cols=cols)
            pool.append(
                (
                    _node("concat", pixels, spatial),
                    "Raw pixels augmented with a coarse spatial prior provide an augmentation control.",
                )
            )
        pool.append((pixels, "Raw pixels are the direct augmentation-track control."))

    rng = random.Random(seed)
    rng.shuffle(pool)
    selected: list[tuple[Expr, str]] = []
    seen: set[str] = set()
    for expr, mechanism in pool:
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
