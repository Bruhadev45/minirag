"""End-to-end: index the 20-document corpus and search it three ways."""

from __future__ import annotations

import pytest

from minirag import Document, Hit, MiniRAG


@pytest.fixture(scope="module")
def engine(documents: tuple[Document, ...]) -> MiniRAG:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(documents)
    return rag


def test_indexing_produces_more_chunks_than_documents(
    engine: MiniRAG, documents: tuple[Document, ...]
) -> None:
    assert len(documents) == 20
    assert engine.chunk_count > len(documents)


def test_hybrid_search_finds_the_right_document_for_every_query(
    engine: MiniRAG, query_cases: tuple[tuple[str, str], ...]
) -> None:
    assert len(query_cases) == 5
    for query, expected in query_cases:
        hits = engine.search(query, top_k=3, mode="hybrid")
        assert hits, f"no hits for {query!r}"
        assert hits[0].doc_id == expected, (
            f"{query!r} -> {[h.doc_id for h in hits]}, expected {expected!r} first"
        )


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_every_mode_recalls_the_answer_in_its_top_three(
    engine: MiniRAG, query_cases: tuple[tuple[str, str], ...], mode: str
) -> None:
    for query, expected in query_cases:
        hits = engine.search(query, top_k=3, mode=mode)  # type: ignore[arg-type]
        assert expected in [h.doc_id for h in hits], f"{mode} missed {expected!r}"


def test_hits_carry_chunk_text_metadata_and_mode(engine: MiniRAG) -> None:
    hit = engine.search("great red spot storm", top_k=1)[0]
    assert isinstance(hit, Hit)
    assert hit.doc_id == "jupiter"
    assert hit.chunk_id.startswith("jupiter#")
    assert "Great Red Spot" in hit.text
    assert hit.metadata["title"] == "Jupiter"
    assert hit.mode == "hybrid"
    assert hit.score > 0.0


def test_results_are_deduplicated_to_one_chunk_per_document(engine: MiniRAG) -> None:
    hits = engine.search("planet atmosphere surface ice", top_k=8)
    doc_ids = [h.doc_id for h in hits]
    assert len(doc_ids) == len(set(doc_ids))


def test_top_k_is_respected(engine: MiniRAG) -> None:
    assert len(engine.search("planet", top_k=2)) <= 2


def test_scores_are_monotonically_decreasing(engine: MiniRAG) -> None:
    scores = [h.score for h in engine.search("ice moon ocean", top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_lexical_mode_nails_a_rare_exact_term(engine: MiniRAG) -> None:
    # "Valles Marineris" appears once in the whole corpus. Exact matching is
    # what BM25 is for.
    assert engine.search("Valles Marineris", top_k=1, mode="bm25")[0].doc_id == "mars"


def test_empty_and_stopword_only_queries_return_nothing(engine: MiniRAG) -> None:
    assert engine.search("") == ()
    assert engine.search("the of and") == ()


def test_out_of_vocabulary_query_finds_nothing_lexically(engine: MiniRAG) -> None:
    assert engine.search("quantum chromodynamics lagrangian", mode="bm25") == ()


def test_hash_collisions_can_produce_spurious_dense_hits(engine: MiniRAG) -> None:
    # An honest limitation of the hashing trick, pinned rather than hidden.
    # Every term lands in *some* column, so an out-of-vocabulary query can
    # collide with real terms and score above zero. "No lexical match" does
    # not imply "no dense match" -- but the scores stay far below a real one.
    dense = engine.search("quantum chromodynamics lagrangian", mode="dense", top_k=10)
    assert all(hit.score < 0.25 for hit in dense), [(h.doc_id, h.score) for h in dense]
    genuine = engine.search("subsurface saltwater ocean icy crust", mode="dense", top_k=1)
    assert genuine[0].score > max((h.score for h in dense), default=0.0) * 2


def test_search_before_index_raises() -> None:
    with pytest.raises(RuntimeError, match="index"):
        MiniRAG().search("anything")


def test_unknown_mode_raises(engine: MiniRAG) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        engine.search("planet", mode="magic")  # type: ignore[arg-type]


def test_invalid_top_k_raises(engine: MiniRAG) -> None:
    with pytest.raises(ValueError, match="top_k"):
        engine.search("planet", top_k=0)


def test_duplicate_doc_ids_raise() -> None:
    rag = MiniRAG()
    with pytest.raises(ValueError, match="duplicate doc_id"):
        rag.index([Document("a", "One."), Document("a", "Two.")])


def test_indexing_does_not_mutate_the_caller_s_documents(
    documents: tuple[Document, ...],
) -> None:
    before = [(d.doc_id, d.text, dict(d.metadata)) for d in documents]
    MiniRAG().index(list(documents))
    assert [(d.doc_id, d.text, dict(d.metadata)) for d in documents] == before


def test_hit_is_immutable(engine: MiniRAG) -> None:
    hit = engine.search("planet", top_k=1)[0]
    with pytest.raises(AttributeError):
        hit.score = 99.0  # type: ignore[misc]


def test_reindexing_replaces_the_previous_corpus(documents: tuple[Document, ...]) -> None:
    rag = MiniRAG()
    rag.index(documents)
    rag.index([Document("only", "Titan has methane lakes.")])
    assert rag.search("methane lakes", top_k=3)[0].doc_id == "only"
    assert {h.doc_id for h in rag.search("jupiter storm", top_k=5)} <= {"only"}


def test_search_is_repeatable(engine: MiniRAG) -> None:
    first = engine.search("icy moon subsurface ocean", top_k=5)
    assert first == engine.search("icy moon subsurface ocean", top_k=5)


def test_weights_are_configurable(documents: tuple[Document, ...]) -> None:
    lexical_heavy = MiniRAG(weights={"bm25": 10.0, "dense": 0.0})
    lexical_heavy.index(documents)
    assert lexical_heavy.search("Olympus Mons", top_k=1)[0].doc_id == "mars"
