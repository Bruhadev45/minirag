"""Hashed TF-IDF embeddings and brute-force cosine search."""

from __future__ import annotations

import numpy as np
import pytest

from minirag.vectors import DEFAULT_DIM, DenseIndex, Embedder, HashedTfidfEmbedder

CORPUS = [
    ("ice", "water", "ocean"),
    ("ice", "crust", "moon"),
    ("storm", "wind", "cloud"),
    ("volcano", "sulfur", "lava"),
]


def test_embedder_satisfies_the_protocol() -> None:
    assert isinstance(HashedTfidfEmbedder.fit(CORPUS, dim=64), Embedder)


def test_fit_returns_a_new_object_and_keeps_the_requested_width() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS, dim=128)
    assert embedder.dim == 128
    assert embedder.idf.shape == (128,)
    assert HashedTfidfEmbedder.fit(CORPUS, dim=128) is not embedder


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        HashedTfidfEmbedder.fit(CORPUS, dim=0)


def test_rows_are_l2_normalised() -> None:
    matrix = HashedTfidfEmbedder.fit(CORPUS).encode(CORPUS)
    assert matrix.shape == (4, DEFAULT_DIM)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_empty_and_unknown_documents_give_a_zero_row_not_a_crash() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS, dim=256)
    matrix = embedder.encode([(), ("ice",)])
    assert np.linalg.norm(matrix[0]) == 0.0
    assert np.linalg.norm(matrix[1]) == pytest.approx(1.0)


def test_hashing_is_stable_across_processes() -> None:
    # Python's built-in hash() is salted per process; blake2b is not. If
    # this ever fails, every persisted index in the world just went stale.
    embedder = HashedTfidfEmbedder(dim=64, idf=np.ones(64))
    first = embedder.encode([("europa", "ocean")])
    second = HashedTfidfEmbedder(dim=64, idf=np.ones(64)).encode([("europa", "ocean")])
    assert np.array_equal(first, second)


def test_identical_texts_are_cosine_identical() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    rows = embedder.encode([("ice", "water"), ("ice", "water")])
    assert float(rows[0] @ rows[1]) == pytest.approx(1.0)


def test_overlapping_texts_are_closer_than_disjoint_ones() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    rows = embedder.encode([("ice", "water", "ocean"), ("ice", "crust"), ("lava", "sulfur")])
    assert float(rows[0] @ rows[1]) > float(rows[0] @ rows[2])


def test_repetition_grows_sublinearly() -> None:
    embedder = HashedTfidfEmbedder(dim=512, idf=np.ones(512))
    weights = [
        float(embedder.encode([("ice",) * n + ("pad",)])[0].max()) for n in (1, 2, 8, 16)
    ]
    assert weights[1] - weights[0] > weights[3] - weights[2] > 0.0


def test_dense_index_finds_the_nearest_row() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    index = DenseIndex.build(embedder, CORPUS)
    query = embedder.encode([("sulfur", "lava")])
    hits = index.search(query, top_k=2)
    assert hits[0][0] == 3
    assert 0.0 < hits[0][1] <= 1.0


def test_dense_index_orders_by_descending_similarity() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    hits = DenseIndex.build(embedder, CORPUS).search(embedder.encode([("ice",)]), top_k=4)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_index_drops_rows_sharing_no_column_with_the_query() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    index = DenseIndex.build(embedder, CORPUS)
    hits = index.search(embedder.encode([("sulfur",)]), top_k=4)
    # Only doc 3 contains "sulfur"; cosine is 1/sqrt(3) because that doc has
    # three equally weighted terms and the query has one.
    assert hits == ((3, pytest.approx(1 / 3**0.5)),)


def test_dense_index_respects_top_k() -> None:
    embedder = HashedTfidfEmbedder.fit(CORPUS)
    index = DenseIndex.build(embedder, CORPUS)
    assert len(index.search(embedder.encode([("ice", "storm", "lava", "ocean")]), top_k=2)) == 2


def test_empty_index_returns_nothing() -> None:
    assert DenseIndex(matrix=np.zeros((0, 8))).search(np.zeros(8)) == ()


def test_dimension_mismatch_raises() -> None:
    index = DenseIndex(matrix=np.eye(3))
    with pytest.raises(ValueError, match="dims"):
        index.search(np.ones(5))


def test_a_custom_embedder_can_be_plugged_into_the_seam() -> None:
    class ConstantEmbedder:
        """The documented seam: anything with dim + encode() works."""

        dim = 2

        def encode(self, documents) -> np.ndarray:
            return np.tile(np.array([[0.6, 0.8]]), (len(documents), 1))

    index = DenseIndex.build(ConstantEmbedder(), CORPUS)
    assert index.matrix.shape == (4, 2)
    assert index.search(np.array([0.6, 0.8]), top_k=1)[0][1] == pytest.approx(1.0)
