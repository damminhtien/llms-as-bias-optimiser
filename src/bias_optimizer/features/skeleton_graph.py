"""Graph and path utilities for binary skeleton images."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
from numpy.typing import NDArray

type Pixel = tuple[int, int]
type SkeletonGraph = nx.Graph


def skeleton_to_graph(skeleton: NDArray[np.bool_]) -> SkeletonGraph:
    """Build an 8-connected pixel graph without redundant diagonal shortcuts."""
    mask = np.asarray(skeleton, dtype=np.bool_)
    if mask.ndim != 2:
        raise ValueError("skeleton must be a two-dimensional array")

    height, width = mask.shape
    pixels = [tuple(int(v) for v in point) for point in np.argwhere(mask)]
    graph = nx.Graph()
    graph.add_nodes_from(pixels)

    forward_neighbors = ((0, 1), (1, -1), (1, 0), (1, 1))
    for row, col in pixels:
        for row_step, col_step in forward_neighbors:
            next_row = row + row_step
            next_col = col + col_step
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not mask[next_row, next_col]:
                continue
            if row_step and col_step and (mask[row, next_col] or mask[next_row, col]):
                continue
            graph.add_edge((row, col), (next_row, next_col))
    return graph


def _junction_clusters(graph: SkeletonGraph) -> tuple[tuple[Pixel, ...], ...]:
    junction_pixels = {node for node, degree in graph.degree if degree >= 3}
    components = nx.connected_components(graph.subgraph(junction_pixels))
    return tuple(sorted(tuple(sorted(component)) for component in components))


def detect_endpoints_and_junctions(
    graph: SkeletonGraph,
) -> tuple[tuple[Pixel, ...], tuple[Pixel, ...]]:
    """Return degree-one pixels and stable representatives of junction clusters."""
    endpoints = tuple(sorted(node for node, degree in graph.degree if degree == 1))
    junctions = tuple(cluster[0] for cluster in _junction_clusters(graph))
    return endpoints, junctions


def _canonical_edge(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
    return (first, second) if first < second else (second, first)


def _contract_junctions(graph: SkeletonGraph) -> SkeletonGraph:
    junction_clusters = _junction_clusters(graph)
    representatives = {
        pixel: cluster[0] for cluster in junction_clusters for pixel in cluster
    }
    contracted = nx.Graph()
    for node in graph:
        representative = representatives.get(node, node)
        contracted.add_node(representative)
    for first, second in graph.edges:
        first_rep = representatives.get(first, first)
        second_rep = representatives.get(second, second)
        if first_rep != second_rep:
            contracted.add_edge(first_rep, second_rep)
    return contracted


def _walk_path(
    graph: SkeletonGraph,
    start: Pixel,
    neighbor: Pixel,
    visited: set[tuple[Pixel, Pixel]],
) -> tuple[Pixel, ...]:
    path = [start, neighbor]
    visited.add(_canonical_edge(start, neighbor))
    previous, current = start, neighbor

    while graph.degree[current] == 2:
        choices = sorted(node for node in graph.neighbors(current) if node != previous)
        if not choices:
            break
        following = choices[0]
        edge = _canonical_edge(current, following)
        if edge in visited:
            break
        visited.add(edge)
        path.append(following)
        previous, current = current, following
    return tuple(path)


def _cycle_paths(
    graph: SkeletonGraph,
    components: Iterable[set[Pixel]],
    visited: set[tuple[Pixel, Pixel]],
) -> list[tuple[Pixel, ...]]:
    paths: list[tuple[Pixel, ...]] = []
    for component in components:
        if not component or any(graph.degree[node] != 2 for node in component):
            continue
        start = min(component)
        neighbor = min(graph.neighbors(start))
        path = [start, neighbor]
        visited.add(_canonical_edge(start, neighbor))
        previous, current = start, neighbor
        while current != start:
            following = next(
                node for node in graph.neighbors(current) if node != previous
            )
            edge = _canonical_edge(current, following)
            if edge in visited:
                break
            visited.add(edge)
            path.append(following)
            previous, current = current, following
        if path[-1] == start:
            paths.append(tuple(path))
    return paths


def extract_graph_paths(graph: SkeletonGraph) -> tuple[tuple[Pixel, ...], ...]:
    """Decompose a skeleton graph into maximal paths between critical nodes."""
    contracted = _contract_junctions(graph)
    visited: set[tuple[Pixel, Pixel]] = set()
    paths: list[tuple[Pixel, ...]] = []
    critical_nodes = sorted(node for node, degree in contracted.degree if degree != 2)

    for node in critical_nodes:
        if contracted.degree[node] == 0:
            paths.append((node,))
            continue
        for neighbor in sorted(contracted.neighbors(node)):
            if _canonical_edge(node, neighbor) not in visited:
                paths.append(_walk_path(contracted, node, neighbor, visited))

    cycle_components = (
        set(component) for component in nx.connected_components(contracted)
    )
    paths.extend(_cycle_paths(contracted, cycle_components, visited))
    return tuple(sorted(paths))
