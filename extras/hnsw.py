"""HNSW: the approximate index minirag deliberately does not need.

``src/minirag/vectors.py`` argues that brute force wins at this scale --
``matrix @ query`` is exact, three lines, and faster than any graph once
you count build time. That claim is only honest if the alternative is on
the table, so here it is: a minimal Hierarchical Navigable Small World
index (Malkov & Yashunin, 2016), living **outside** the 800-line budget
because it is a different promise. Same spirit, though: no dependency
beyond numpy, every parameter explained, and a test that measures what it
costs you instead of asserting it is fine.

**The idea.** Build a graph whose edges are "near neighbours", then walk it
greedily towards the query. One flat graph gets stuck in local minima, so
HNSW stacks several: each node is assigned a level from an exponentially
decaying distribution, so the top layer is a sparse sketch of the corpus
and layer 0 holds everything. A search drops in at the top, greedily walks
to the closest node it can find, uses that as the entry point one layer
down, and only at layer 0 widens into a proper beam of width ``ef``.

**What it buys and what it costs.** Search touches ``O(log n)`` nodes
instead of all ``n`` -- the reason ANN exists. The price is recall: the
walk can miss a true neighbour that no edge leads to, and unlike a wrong
BM25 weight this failure is silent. ``ef`` is the dial; raising it trades
latency back for recall. ``tests/test_hnsw.py`` measures both ends of that
trade against brute force on the fixture corpus, which is the only way the
claim in ``vectors.py`` stays checkable.

Vectors are assumed L2-normalised, as :class:`~minirag.vectors.DenseIndex`
assumes, so distance is ``1 - cosine`` and the two indexes are directly
comparable -- same input matrix, same return shape, same ``sim > 0`` filter.
"""

from __future__ import annotations

import heapq
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

DEFAULT_M = 8
"""Edges kept per node per layer. Layer 0 keeps ``2 * M`` -- it is the only
layer that must stay connected enough to answer, so it gets the budget."""

DEFAULT_EF_CONSTRUCTION = 32
"""Beam width while inserting. Wider means better edges and a slower build."""

DEFAULT_EF_SEARCH = 32
"""Beam width while searching. The recall dial; must be >= ``top_k``."""

_Graph = Mapping[int, tuple[int, ...]]


def _distance(matrix: np.ndarray, node: int, vector: np.ndarray) -> float:
    """Cosine distance from row ``node`` to ``vector``; 0.0 means identical."""
    return 1.0 - float(matrix[node] @ vector)


def _greedy(matrix: np.ndarray, graph: _Graph, entry: int, vector: np.ndarray) -> int:
    """Walk downhill from ``entry`` until no neighbour is closer to ``vector``."""
    current, best = entry, _distance(matrix, entry, vector)
    improved = True
    while improved:
        improved = False
        for neighbour in graph.get(current, ()):
            candidate = _distance(matrix, neighbour, vector)
            if candidate < best:
                current, best, improved = neighbour, candidate, True
    return current


def _search_layer(
    matrix: np.ndarray, graph: _Graph, entries: Sequence[int], vector: np.ndarray, ef: int
) -> list[tuple[float, int]]:
    """Beam search one layer, returning up to ``ef`` ``(distance, node)`` pairs.

    Two heaps: ``frontier`` is a min-heap of nodes still worth expanding,
    ``found`` a max-heap (negated) of the best ``ef`` seen so far. The walk
    stops as soon as the nearest unexpanded candidate is further away than
    the worst result held -- nothing behind it can improve the beam.
    """
    visited = set(entries)
    frontier = [(_distance(matrix, node, vector), node) for node in entries]
    heapq.heapify(frontier)
    found = [(-distance, node) for distance, node in frontier]
    heapq.heapify(found)
    while frontier:
        distance, node = heapq.heappop(frontier)
        if len(found) >= ef and distance > -found[0][0]:
            break
        for neighbour in graph.get(node, ()):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            candidate = _distance(matrix, neighbour, vector)
            if len(found) < ef or candidate < -found[0][0]:
                heapq.heappush(frontier, (candidate, neighbour))
                heapq.heappush(found, (-candidate, neighbour))
                if len(found) > ef:
                    heapq.heappop(found)
    return sorted((-negated, node) for negated, node in found)


def _nearest(
    matrix: np.ndarray, node: int, candidates: Sequence[int], keep: int
) -> tuple[int, ...]:
    """Keep the ``keep`` neighbours closest to ``node``, ties broken by index.

    The naive pruning rule. The paper's heuristic also keeps a few *distant*
    neighbours to bridge clusters, which raises recall on clustered data;
    this is the honest simple version, and the recall tests report what it
    actually gets rather than what the paper gets.
    """
    vector = matrix[node]
    ranked = sorted(set(candidates) - {node}, key=lambda i: (_distance(matrix, i, vector), i))
    return tuple(ranked[:keep])


@dataclass(frozen=True, slots=True, eq=False)
class HnswIndex:
    """A built HNSW graph over a matrix of unit-norm row vectors.

    ``layers[0]`` contains every node; higher layers get exponentially
    fewer. Each layer maps a row index to its outgoing edges.
    """

    matrix: np.ndarray
    layers: tuple[_Graph, ...]
    entry_point: int

    @classmethod
    def build(
        cls,
        matrix: np.ndarray,
        *,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        seed: int = 0,
    ) -> HnswIndex:
        """Insert every row of ``matrix`` into a fresh graph.

        Levels come from a seeded :class:`random.Random`, never the global
        one: a graph that differs between runs would make recall
        unmeasurable and bug reports unreproducible.

        Raises:
            ValueError: If ``m < 2`` or ``ef_construction < 1``.
        """
        if m < 2:
            raise ValueError("m must be >= 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be >= 1")
        if matrix.size == 0:
            return cls(matrix=matrix, layers=(MappingProxyType({}),), entry_point=-1)

        rng = random.Random(seed)
        scale = 1.0 / math.log(m)
        graphs: list[dict[int, tuple[int, ...]]] = [{}]
        entry_point = 0
        for node in range(matrix.shape[0]):
            level = int(-math.log(max(rng.random(), 1e-12)) * scale)
            top = len(graphs) - 1
            graphs.extend({} for _ in range(level - top))
            for layer in range(level + 1):
                graphs[layer][node] = ()
            if node == 0:
                continue
            cls._insert(matrix, graphs, node, level, top, entry_point, m, ef_construction)
            if level > top:
                entry_point = node
        return cls(
            matrix=matrix,
            layers=tuple(MappingProxyType(dict(graph)) for graph in graphs),
            entry_point=entry_point,
        )

    @staticmethod
    def _insert(
        matrix: np.ndarray,
        graphs: list[dict[int, tuple[int, ...]]],
        node: int,
        level: int,
        top: int,
        entry_point: int,
        m: int,
        ef_construction: int,
    ) -> None:
        """Link ``node`` into every layer at or below its level."""
        vector = matrix[node]
        current = entry_point
        for layer in range(top, level, -1):
            current = _greedy(matrix, graphs[layer], current, vector)
        for layer in range(min(level, top), -1, -1):
            found = _search_layer(matrix, graphs[layer], [current], vector, ef_construction)
            degree = 2 * m if layer == 0 else m
            neighbours = _nearest(matrix, node, [i for _, i in found], m)
            graphs[layer][node] = neighbours
            for neighbour in neighbours:
                linked = (*graphs[layer][neighbour], node)
                graphs[layer][neighbour] = (
                    linked if len(linked) <= degree else _nearest(matrix, neighbour, linked, degree)
                )
            current = neighbours[0] if neighbours else current

    def search(
        self, query_vector: np.ndarray, *, top_k: int = 10, ef: int = DEFAULT_EF_SEARCH
    ) -> tuple[tuple[int, float], ...]:
        """Return up to ``top_k`` ``(row_index, cosine)`` pairs, best first.

        Same signature and same ``cosine > 0`` filter as
        :meth:`~minirag.vectors.DenseIndex.search`, so the two are drop-in
        comparable -- only the answers differ, and only sometimes.

        Raises:
            ValueError: If ``top_k < 1``, ``ef < 1``, or the query width
                does not match the index.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if ef < 1:
            raise ValueError("ef must be >= 1")
        if self.matrix.size == 0:
            return ()
        if query_vector.shape[-1] != self.matrix.shape[1]:
            raise ValueError(
                f"query has {query_vector.shape[-1]} dims, index has {self.matrix.shape[1]}"
            )
        vector = query_vector.reshape(-1)
        current = self.entry_point
        for layer in range(len(self.layers) - 1, 0, -1):
            current = _greedy(self.matrix, self.layers[layer], current, vector)
        found = _search_layer(self.matrix, self.layers[0], [current], vector, max(ef, top_k))
        hits = ((node, 1.0 - distance) for distance, node in found[:top_k])
        return tuple((node, cosine) for node, cosine in hits if cosine > 0.0)
