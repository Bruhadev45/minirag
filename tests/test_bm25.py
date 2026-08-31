"""BM25: index structure, IDF, k1 saturation, b length normalisation."""

from __future__ import annotations

import pytest

from minirag.bm25 import build_index, idf, score_query, search

CORPUS = [
    ("ice", "water", "ocean"),
    ("ice", "ice", "ice", "rock"),
    ("storm", "wind", "cloud"),
    ("ice", "storm"),
]


def test_postings_map_terms_to_documents_and_counts() -> None:
    index = build_index(CORPUS)
    assert index.postings["ice"] == ((0, 1), (1, 3), (3, 1))
    assert index.postings["water"] == ((0, 1),)
    assert "unknown" not in index.postings


def test_document_statistics() -> None:
    index = build_index(CORPUS)
    assert index.n_docs == 4
    assert index.doc_lengths == (3, 4, 3, 2)
    assert index.avg_doc_length == pytest.approx(3.0)


def test_empty_corpus_is_legal() -> None:
    index = build_index([])
    assert index.n_docs == 0
    assert score_query(index, ["ice"]) == ()
    assert search(index, ["ice"]) == ()


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.5, -0.1), (1.5, 1.1)])
def test_invalid_parameters_raise(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        build_index(CORPUS, k1=k1, b=b)


def test_idf_is_higher_for_rarer_terms() -> None:
    index = build_index(CORPUS)
    assert idf(index, "water") > idf(index, "storm") > idf(index, "ice")


def test_idf_is_never_negative_even_for_very_common_terms() -> None:
    index = build_index([("a", "x"), ("a", "y"), ("a", "z")])
    assert idf(index, "a") >= 0.0


def test_idf_of_unknown_term_is_zero_not_an_error() -> None:
    assert idf(build_index(CORPUS), "quasar") == 0.0


def test_documents_without_query_terms_score_exactly_zero() -> None:
    scores = score_query(build_index(CORPUS), ["water"])
    assert scores[0] > 0.0
    assert scores[1] == scores[2] == scores[3] == 0.0


def test_term_frequency_saturates() -> None:
    # Doubling from 1 to 2 occurrences must help more than 8 to 16 does.
    corpus = [tuple(["ice"] * n) + ("pad",) * (16 - n) for n in (1, 2, 8, 16)]
    scores = score_query(build_index(corpus), ["ice"])
    assert scores[1] - scores[0] > scores[3] - scores[2] > 0.0


def test_k1_zero_makes_term_frequency_binary() -> None:
    corpus = [("ice",), ("ice", "ice", "ice", "ice")]
    flat = score_query(build_index(corpus, k1=0.0, b=0.0), ["ice"])
    assert flat[0] == pytest.approx(flat[1])


def test_b_penalises_long_documents() -> None:
    corpus = [("ice",) + ("pad",) * 2, ("ice",) + ("pad",) * 40]
    normalised = score_query(build_index(corpus, b=1.0), ["ice"])
    unnormalised = score_query(build_index(corpus, b=0.0), ["ice"])
    assert normalised[0] > normalised[1]
    assert unnormalised[0] == pytest.approx(unnormalised[1])


def test_repeated_query_terms_do_not_double_count() -> None:
    index = build_index(CORPUS)
    assert score_query(index, ["ice", "ice"]) == score_query(index, ["ice"])


def test_search_returns_pairs_sorted_by_descending_score() -> None:
    hits = search(build_index(CORPUS), ["ice", "storm"], top_k=10)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0.0 for score in scores)


def test_search_omits_documents_sharing_no_query_term() -> None:
    # Doc 0 is ("ice", "water", "ocean"); a "rock" query must not reach it.
    assert [i for i, _ in search(build_index(CORPUS), ["rock"])] == [1]


def test_search_respects_top_k() -> None:
    assert len(search(build_index(CORPUS), ["ice"], top_k=2)) == 2


def test_search_is_deterministic_on_ties() -> None:
    corpus = [("ice",), ("ice",), ("ice",)]
    index = build_index(corpus)
    assert [i for i, _ in search(index, ["ice"])] == [0, 1, 2]
