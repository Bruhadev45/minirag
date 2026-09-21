"""Metadata filtering: ``search(..., where={"field": value})``.

The property that matters is *pre*-ranking: the filter runs before results
are truncated, so a matching document can never be crowded out by
non-matching ones that happened to rank higher.
"""

from __future__ import annotations

import pytest

from minirag import Document, MiniRAG

MODES = ("bm25", "dense", "hybrid")


@pytest.fixture(scope="module")
def engine(documents: tuple[Document, ...]) -> MiniRAG:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(documents)
    return rag


@pytest.fixture(scope="module")
def crowded() -> MiniRAG:
    """30 planets that match "ocean" strongly, then one moon that barely does."""
    planets = [
        Document(f"p{i:02d}", f"Planet {i} has a deep ocean, a vast ocean.", {"kind": "planet"})
        for i in range(30)
    ]
    moon = Document(
        "moon", "A small cratered moon with dust, rock and one frozen ocean far below.",
        {"kind": "moon", "parent": "p00"},
    )
    rag = MiniRAG()
    rag.index([*planets, moon])
    return rag


@pytest.mark.parametrize("mode", MODES)
def test_every_hit_matches_the_filter(engine: MiniRAG, mode: str) -> None:
    where = {"title": "Mars"}
    hits = engine.search("atmosphere", top_k=5, mode=mode, where=where)  # type: ignore[arg-type]
    assert [h.doc_id for h in hits] == ["mars"]


@pytest.mark.parametrize("mode", MODES)
def test_filter_runs_before_truncation(crowded: MiniRAG, mode: str) -> None:
    # Unfiltered, the moon is outranked by 30 planets -- deeper than the
    # engine's over-fetch of 20 -- so a post-filter would find nothing.
    unfiltered = crowded.search("ocean", top_k=20, mode=mode)  # type: ignore[arg-type]
    assert "moon" not in [h.doc_id for h in unfiltered]
    where = {"kind": "moon"}
    hits = crowded.search("ocean", top_k=1, mode=mode, where=where)  # type: ignore[arg-type]
    assert [h.doc_id for h in hits] == ["moon"]


def test_filtered_results_still_fill_top_k(crowded: MiniRAG) -> None:
    hits = crowded.search("ocean", top_k=5, where={"kind": "planet"})
    assert len(hits) == 5
    assert all(h.metadata["kind"] == "planet" for h in hits)


def test_multiple_pairs_must_all_match(crowded: MiniRAG) -> None:
    hits = crowded.search("ocean", where={"kind": "moon", "parent": "p00"})
    assert [h.doc_id for h in hits] == ["moon"]
    assert crowded.search("ocean", where={"kind": "moon", "parent": "p01"}) == ()


def test_a_missing_field_excludes_the_document(crowded: MiniRAG) -> None:
    # Only the moon has a "parent" field; planets lacking it never match.
    hits = crowded.search("ocean", top_k=10, where={"parent": "p00"})
    assert [h.doc_id for h in hits] == ["moon"]


def test_no_matching_document_returns_nothing(engine: MiniRAG) -> None:
    assert engine.search("planet", where={"title": "Vulcan"}) == ()
    assert engine.search("planet", where={"no_such_field": "x"}) == ()


@pytest.mark.parametrize("where", [None, {}])
def test_no_filter_is_identical_to_unfiltered(engine: MiniRAG, where: dict | None) -> None:
    unfiltered = engine.search("icy moon ocean", top_k=5)
    assert engine.search("icy moon ocean", top_k=5, where=where) == unfiltered


def test_bm25_scores_are_unchanged_by_filtering(engine: MiniRAG) -> None:
    # The filter narrows the candidates, not the statistics: IDF is still
    # computed over the whole corpus, so a surviving hit keeps its score.
    full = {h.doc_id: h.score for h in engine.search("ocean ice moon", top_k=20, mode="bm25")}
    hits = engine.search("ocean ice moon", top_k=5, mode="bm25", where={"title": "Europa"})
    assert [h.doc_id for h in hits] == ["europa"]
    assert hits[0].score == pytest.approx(full["europa"])


def test_filtering_does_not_mutate_where(engine: MiniRAG) -> None:
    where = {"title": "Mars"}
    engine.search("planet", where=where)
    assert where == {"title": "Mars"}
