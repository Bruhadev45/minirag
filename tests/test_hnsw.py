"""What the approximate index actually costs, measured against brute force.

``src/minirag/vectors.py`` claims brute force is the right call at
minirag's scale. ``extras/hnsw.py`` is the alternative, so this file checks
the claim instead of repeating it: on the fixture corpus the graph loses
*nothing* (recall 1.0 -- 40 chunks is far below where ANN pays), and on a
harder synthetic set it loses a lot unless you pay for ``ef``. Recall is
the number an ANN index must be judged on, and the failure is silent
otherwise: a missing neighbour looks exactly like a correct answer.

Every number here comes from a seeded graph over deterministic vectors, so
a regression in the walk shows up as a failing assertion rather than as
slightly worse search nobody notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from extras.hnsw import HnswIndex
from minirag.chunker import chunk_document
from minirag.tokenize import tokenize
from minirag.vectors import DenseIndex, HashedTfidfEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "corpus.json"


def _recall_at_k(
    brute: DenseIndex, graph: HnswIndex, queries: np.ndarray, *, k: int, ef: int
) -> float:
    """Fraction of brute-force neighbours the graph also returned."""
    expected = found = 0
    for query in queries:
        exact = {i for i, _ in brute.search(query, top_k=k)}
        approximate = {i for i, _ in graph.search(query, top_k=k, ef=ef)}
        expected += len(exact)
        found += len(exact & approximate)
    return found / expected


@pytest.fixture(scope="module")
def corpus_vectors() -> tuple[np.ndarray, np.ndarray]:
    """The fixture corpus as ``(chunk_matrix, query_matrix)``, exactly as the
    engine would build it: chunked at 60 tokens, hashed TF-IDF, unit norm."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chunks = [
        chunk
        for document in data["documents"]
        for chunk in chunk_document(
            document["text"], document["doc_id"], max_tokens=60, overlap_tokens=20
        )
    ]
    tokens = [tokenize(chunk.text) for chunk in chunks]
    embedder = HashedTfidfEmbedder.fit(tokens, dim=1024)
    queries = [q["query"] for q in data["queries"]] + [d["text"][:80] for d in data["documents"]]
    return embedder.encode(tokens), embedder.encode([tokenize(q) for q in queries])


@pytest.fixture(scope="module")
def random_vectors() -> tuple[np.ndarray, np.ndarray]:
    """800 unit vectors and 60 queries in 64 dimensions, seeded.

    Gaussian noise is the hard case for a navigable graph: no clusters to
    exploit, every point roughly equidistant from every other. Real
    embeddings are kinder, which is why the recall here is a floor.
    """
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(800, 64))
    queries = rng.normal(size=(60, 64))
    return (
        matrix / np.linalg.norm(matrix, axis=1, keepdims=True),
        queries / np.linalg.norm(queries, axis=1, keepdims=True),
    )


def test_the_fixture_corpus_is_small_enough_that_the_graph_loses_nothing(
    corpus_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, queries = corpus_vectors
    assert matrix.shape[0] == 40
    recall = _recall_at_k(
        DenseIndex(matrix=matrix), HnswIndex.build(matrix), queries, k=10, ef=32
    )
    assert recall == 1.0, f"recall@10 dropped to {recall:.4f} on 40 chunks"


def test_top_result_matches_brute_force_on_every_corpus_query(
    corpus_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, queries = corpus_vectors
    brute, graph = DenseIndex(matrix=matrix), HnswIndex.build(matrix)
    for query in queries:
        assert graph.search(query, top_k=5)[0][0] == brute.search(query, top_k=5)[0][0]


def test_scores_agree_with_brute_force_where_both_return_a_document(
    corpus_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    # Same cosine, computed the same way -- only the candidate set differs.
    matrix, queries = corpus_vectors
    exact = dict(DenseIndex(matrix=matrix).search(queries[0], top_k=10))
    for index, score in HnswIndex.build(matrix).search(queries[0], top_k=10):
        assert score == pytest.approx(exact[index])


@pytest.mark.parametrize(("ef", "floor"), [(16, 0.60), (32, 0.80), (64, 0.90)])
def test_recall_on_unclustered_vectors_is_bought_with_ef(
    random_vectors: tuple[np.ndarray, np.ndarray], ef: int, floor: float
) -> None:
    matrix, queries = random_vectors
    recall = _recall_at_k(
        DenseIndex(matrix=matrix), HnswIndex.build(matrix), queries, k=10, ef=ef
    )
    assert recall >= floor, f"recall@10 with ef={ef} was {recall:.4f}"


def test_more_edges_beat_a_wider_beam(random_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    # The build-time dial is the stronger one: m=16 recovers what ef alone
    # cannot, at the cost of a slower build and twice the memory per node.
    matrix, queries = random_vectors
    brute = DenseIndex(matrix=matrix)
    recall = _recall_at_k(brute, HnswIndex.build(matrix, m=16, ef_construction=32), queries,
                          k=10, ef=64)
    assert recall >= 0.99, f"recall@10 with m=16 was {recall:.4f}"


def test_building_twice_with_the_same_seed_gives_the_same_graph(
    random_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, _ = random_vectors
    first, second = HnswIndex.build(matrix, seed=3), HnswIndex.build(matrix, seed=3)
    assert first.entry_point == second.entry_point
    assert [dict(layer) for layer in first.layers] == [dict(layer) for layer in second.layers]


def test_a_different_seed_gives_a_different_graph_that_still_works(
    random_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, queries = random_vectors
    graph = HnswIndex.build(matrix, seed=11)
    assert [dict(layer) for layer in graph.layers] != [
        dict(layer) for layer in HnswIndex.build(matrix, seed=3).layers
    ]
    assert _recall_at_k(DenseIndex(matrix=matrix), graph, queries, k=10, ef=64) >= 0.90


def test_layers_shrink_and_layer_zero_holds_every_row(
    random_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, _ = random_vectors
    graph = HnswIndex.build(matrix)
    sizes = [len(layer) for layer in graph.layers]
    assert sizes[0] == matrix.shape[0]
    assert sizes == sorted(sizes, reverse=True)
    assert graph.entry_point in graph.layers[-1]


def test_degree_caps_hold_on_every_layer(random_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    matrix, _ = random_vectors
    graph = HnswIndex.build(matrix, m=8)
    for level, layer in enumerate(graph.layers):
        cap = 16 if level == 0 else 8
        assert all(len(edges) <= cap for edges in layer.values())
        assert all(node not in edges for node, edges in layer.items())


def test_edges_are_symmetric_enough_to_walk_back(
    random_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    # Pruning can drop a back-edge, but layer 0 must stay connected: an
    # orphan is unreachable, and unreachable is invisible, not slow.
    matrix, _ = random_vectors
    layer_zero = HnswIndex.build(matrix).layers[0]
    incoming = {node for edges in layer_zero.values() for node in edges}
    assert set(layer_zero) - incoming == set()


def test_an_empty_index_returns_nothing() -> None:
    graph = HnswIndex.build(np.zeros((0, 8)))
    assert graph.search(np.ones(8), top_k=5) == ()


def test_a_single_vector_index_returns_it() -> None:
    matrix = np.array([[1.0, 0.0]])
    assert HnswIndex.build(matrix).search(np.array([1.0, 0.0]), top_k=5) == ((0, 1.0),)


def test_orthogonal_queries_are_filtered_like_the_brute_force_index() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    query = np.array([1.0, 0.0])
    assert HnswIndex.build(matrix).search(query) == DenseIndex(matrix=matrix).search(query)


def test_top_k_larger_than_the_corpus_is_not_an_error(
    corpus_vectors: tuple[np.ndarray, np.ndarray],
) -> None:
    matrix, queries = corpus_vectors
    assert len(HnswIndex.build(matrix).search(queries[0], top_k=500)) <= matrix.shape[0]


def test_a_query_of_the_wrong_width_raises(corpus_vectors: tuple[np.ndarray, np.ndarray]) -> None:
    matrix, _ = corpus_vectors
    with pytest.raises(ValueError, match="dims"):
        HnswIndex.build(matrix).search(np.ones(7))


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"ef": 0}])
def test_invalid_search_parameters_raise(kwargs: dict[str, int]) -> None:
    graph = HnswIndex.build(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        graph.search(np.array([1.0, 0.0]), **kwargs)


@pytest.mark.parametrize("kwargs", [{"m": 1}, {"ef_construction": 0}])
def test_invalid_build_parameters_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        HnswIndex.build(np.array([[1.0, 0.0]]), **kwargs)
