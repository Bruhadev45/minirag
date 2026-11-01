"""The explain extra: same hits as hybrid search, and the sums that make them.

``extras/explain.py`` promises that its hybrid hits *are* the engine's
hybrid hits, and that every hit's score is the sum of the contributions it
prints. If either drifts, the README's worked example shows arithmetic the
engine never did, so both are checked against ``engine.search`` directly.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter

import numpy as np
import pytest

from extras.explain import (
    QUERIES,
    RETRIEVERS,
    Explanation,
    explain,
    format_explanation,
    main,
)
from minirag import Document, MiniRAG, tokenize
from minirag.vectors import _column

EXTRA_QUERIES = (
    "icy moon of saturn",
    "volcanoes throwing sulfur",  # one BM25 hit, so dense alone ranks the rest
    "zzzz qqqq",  # no word in the corpus
    "the of and",  # nothing left after stopwords
)


@pytest.fixture(scope="module")
def engine(documents: tuple[Document, ...]) -> MiniRAG:
    """The fixture corpus, indexed as the README and ``python -m extras.explain`` do."""
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(documents)
    return rag


@pytest.fixture(scope="module")
def all_queries(query_cases: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return (*(query for query, _ in query_cases), *QUERIES, *EXTRA_QUERIES)


def check_against_search(
    rag: MiniRAG, query: str, top_k: int, where: dict[str, str] | None
) -> Explanation:
    """Assert the two promises for one call, and return the explanation."""
    explanation = explain(rag, query, top_k=top_k, where=where)
    expected = rag.search(query, top_k=top_k, mode="hybrid", where=where)
    assert tuple(item.hit for item in explanation.hits) == expected
    assert [item.rank for item in explanation.hits] == list(range(1, len(expected) + 1))
    for item in explanation.hits:
        assert sum(c.rrf for c in item.contributions) == item.hit.score  # exactly, not approx
    return explanation


@pytest.mark.parametrize("top_k", [1, 3, 5, 20])
@pytest.mark.parametrize("where", [None, {"title": "Europa"}, {"title": "no such title"}])
def test_hits_are_hybrid_search_and_sum_to_its_scores(
    engine: MiniRAG, all_queries: tuple[str, ...], top_k: int, where: dict[str, str] | None
) -> None:
    for query in all_queries:
        check_against_search(engine, query, top_k, where)


def test_each_contribution_is_weight_over_k_plus_that_lists_rank(
    documents: tuple[Document, ...], all_queries: tuple[str, ...]
) -> None:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20, rrf_k=10, weights={"bm25": 2.0, "dense": 0.5})
    rag.index(documents)
    for query in all_queries:
        explanation = check_against_search(rag, query, 5, None)
        assert explanation.k == 10
        for item in explanation.hits:
            for c, name in zip(item.contributions, RETRIEVERS, strict=True):
                assert (c.retriever, c.weight) == (name, {"bm25": 2.0, "dense": 0.5}[name])
                ids = [ranked.chunk_id for ranked in explanation.lists[name]]
                if c.rank is None:
                    assert item.hit.chunk_id not in ids and c.rrf == 0.0
                else:
                    assert ids[c.rank - 1] == item.hit.chunk_id
                    assert c.rrf == c.weight / (10 + c.rank)


def test_a_weight_left_out_defaults_to_one_as_in_fusion(documents: tuple[Document, ...]) -> None:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20, weights={"bm25": 3.0})
    rag.index(documents)
    explanation = check_against_search(rag, QUERIES[0], 5, None)
    assert [c.weight for c in explanation.hits[0].contributions] == [3.0, 1.0]


def test_each_list_collapses_to_that_modes_search(
    engine: MiniRAG, all_queries: tuple[str, ...]
) -> None:
    for query in all_queries:
        explanation = explain(engine, query, top_k=5)
        for name in RETRIEVERS:
            best: dict[str, tuple[str, str, float]] = {}
            for ranked in explanation.lists[name]:
                best.setdefault(ranked.doc_id, (ranked.doc_id, ranked.chunk_id, ranked.score))
            expected = engine.search(query, top_k=5, mode=name)  # type: ignore[arg-type]
            assert list(best.values())[:5] == [(h.doc_id, h.chunk_id, h.score) for h in expected]
            assert [r.rank for r in explanation.lists[name]] == list(
                range(1, len(explanation.lists[name]) + 1)
            )


def test_where_filters_both_lists_before_ranking(engine: MiniRAG) -> None:
    explanation = explain(engine, QUERIES[1], top_k=5, where={"title": "Europa"})
    for name in RETRIEVERS:
        assert explanation.lists[name]
        assert {ranked.doc_id for ranked in explanation.lists[name]} == {"europa"}
        assert explanation.lists[name][0].rank == 1


def test_a_chunk_one_retriever_did_not_rank_adds_zero() -> None:
    # Two hash columns: every word collides, so dense ranks chunks BM25 cannot.
    rag = MiniRAG(dim=2)
    rag.index(
        [
            Document("comets", "Comets carry frozen gases."),
            Document("rivers", "Rivers carve canyons slowly."),
        ]
    )
    explanation = check_against_search(rag, "frozen comets", 3, None)
    bm25, dense = explanation.hits[1].contributions
    assert explanation.hits[1].hit.doc_id == "rivers"
    assert (bm25.rank, bm25.rrf, dense.rank) == (None, 0.0, 2)
    assert "not ranked = 0" in format_explanation(explanation)


def test_built_in_queries_are_ones_the_retrievers_disagree_on(engine: MiniRAG) -> None:
    for query in QUERIES:
        explanation = explain(engine, query, top_k=3)
        bm25, dense = (explanation.lists[name][0].chunk_id for name in RETRIEVERS)
        assert bm25 != dense
        assert "disagree on the top chunk" in format_explanation(explanation)


def test_the_readmes_reason_dense_puts_the_ceres_chunk_first(engine: MiniRAG) -> None:
    """README: unnormalised, Pluto's chunk scores 3.90 to the Ceres chunk's 2.97,
    but it has 28 tokens to 17, and dividing by the vector's length flips them.

    Deliberately coupled to ``minirag.vectors``: it redoes the embedder's
    weighting without the final division, and the ``approx`` check against
    the engine's own cosine fails first if the embedder changes. Then
    re-measure and update the README paragraph with these numbers.
    """
    state = engine._state
    assert state is not None
    embedder, query = state.embedder, state.embedder.encode([tokenize(QUERIES[0])])[0]
    cosine = {r.chunk_id: r.score for r in explain(engine, QUERIES[0]).lists["dense"]}
    measured = {}
    for chunk in state.chunks:
        if chunk.chunk_id in ("pluto#0", "asteroid_belt#1"):
            tokens = tokenize(chunk.text)
            raw = np.zeros(embedder.dim)  # HashedTfidfEmbedder.encode, minus the last step
            for term, freq in Counter(tokens).items():
                raw[_column(term, embedder.dim)] += 1.0 + math.log(freq)
            raw *= embedder.idf
            assert raw @ query / np.linalg.norm(raw) == pytest.approx(cosine[chunk.chunk_id])
            measured[chunk.chunk_id] = (len(tokens), round(float(raw @ query), 2))
    assert measured == {"pluto#0": (28, 3.90), "asteroid_belt#1": (17, 2.97)}
    assert cosine["pluto#0"] < cosine["asteroid_belt#1"]


def test_an_empty_query_explains_nothing(engine: MiniRAG) -> None:
    explanation = explain(engine, "the of and")
    assert explanation.hits == () and all(not ranked for ranked in explanation.lists.values())
    assert format_explanation(explanation).splitlines() == [
        "query: 'the of and'",
        "",
        "  bm25 and dense ranked nothing.",
        "",
        "  hybrid: no hits.",
    ]


def test_formats_custom_weights_and_k(documents: tuple[Document, ...]) -> None:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20, rrf_k=10, weights={"bm25": 2.0, "dense": 0.5})
    rag.index(documents)
    text = format_explanation(explain(rag, QUERIES[0], top_k=1), shown=1)
    assert "2/(10+1) = 0.181818" in text
    assert "0.5/(10+2) = 0.041667" in text


def test_rejects_an_unindexed_engine_and_a_bad_top_k(engine: MiniRAG) -> None:
    with pytest.raises(RuntimeError, match="index"):
        explain(MiniRAG(), "ocean")
    with pytest.raises(ValueError, match="top_k"):
        explain(engine, "ocean", top_k=0)


def test_an_explanation_is_read_only(engine: MiniRAG) -> None:
    explanation = explain(engine, QUERIES[0], top_k=1)
    with pytest.raises(TypeError):
        explanation.lists["bm25"] = ()  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        explanation.hits[0].contributions[0].rrf = 1.0  # type: ignore[misc]


def test_main_explains_the_built_in_queries_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    out = capsys.readouterr().out
    assert out.startswith("fixture corpus: 20 documents -> 40 chunks\n")
    assert [line for line in out.splitlines() if line.startswith("query:")] == [
        f"query: {query!r}" for query in QUERIES
    ]


def test_main_explains_given_queries_to_the_given_depth(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["icy moon of saturn", "--top-k", "2"])
    out = capsys.readouterr().out
    assert "query: 'icy moon of saturn'" in out and "dwarf" not in out
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  rank  hybrid"))
    assert [line.split()[0] for line in lines[start + 1 :]] == ["1", "2"]


def test_main_rejects_a_top_k_below_one(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--top-k", "0"])
    assert exc.value.code == 2
    assert "--top-k must be >= 1" in capsys.readouterr().err
