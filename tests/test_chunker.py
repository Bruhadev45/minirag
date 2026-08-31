"""Chunking: sentence alignment, the size budget, and overlap guarantees."""

from __future__ import annotations

from itertools import pairwise

import pytest

from minirag.chunker import chunk_corpus, chunk_document, split_sentences

PARAGRAPH = " ".join(f"Sentence number {i} has exactly seven words." for i in range(1, 21))


def test_split_sentences_basic() -> None:
    assert split_sentences("One. Two! Three?") == ("One.", "Two!", "Three?")


def test_split_sentences_ignores_blank_input() -> None:
    assert split_sentences("   ") == ()


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_document("", "d") == ()


def test_chunk_ids_are_unique_and_ordered() -> None:
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=20, overlap_tokens=7)
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"doc#{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "doc" for c in chunks)


def test_chunks_are_whole_sentences() -> None:
    sentences = set(split_sentences(PARAGRAPH))
    for chunk in chunk_document(PARAGRAPH, "doc", max_tokens=20, overlap_tokens=7):
        assert set(split_sentences(chunk.text)) <= sentences


def test_respects_the_token_budget() -> None:
    for chunk in chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=7):
        assert len(chunk.text.split()) <= 21


def test_single_oversized_sentence_becomes_its_own_chunk() -> None:
    long_sentence = "word " * 50 + "end."
    chunks = chunk_document(long_sentence, "doc", max_tokens=10, overlap_tokens=3)
    assert len(chunks) == 1
    assert len(chunks[0].text.split()) == 51


def test_overlap_repeats_trailing_context() -> None:
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=7)
    assert len(chunks) > 1
    for earlier, later in pairwise(chunks):
        assert later.sentence_span[0] < earlier.sentence_span[1]


def test_every_adjacent_sentence_pair_shares_a_chunk() -> None:
    # The whole point of overlap: no boundary separates two neighbouring
    # sentences in *every* chunk.
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=7)
    covered = {
        (i, i + 1)
        for c in chunks
        for i in range(c.sentence_span[0], c.sentence_span[1] - 1)
    }
    n_sentences = len(split_sentences(PARAGRAPH))
    assert covered == {(i, i + 1) for i in range(n_sentences - 1)}


def test_zero_overlap_partitions_without_repeats() -> None:
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=0)
    spans = [c.sentence_span for c in chunks]
    assert all(a[1] == b[0] for a, b in pairwise(spans))


def test_chunks_cover_the_whole_document() -> None:
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=7)
    assert chunks[0].sentence_span[0] == 0
    assert chunks[-1].sentence_span[1] == len(split_sentences(PARAGRAPH))


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens"),
    [(0, 0), (10, 10), (10, 20), (10, -1)],
)
def test_invalid_window_configuration_raises(max_tokens: int, overlap_tokens: int) -> None:
    with pytest.raises(ValueError):
        chunk_document("A sentence.", "d", max_tokens=max_tokens, overlap_tokens=overlap_tokens)


def test_chunk_corpus_does_not_mutate_its_input() -> None:
    documents = {"a": "First doc. Second sentence.", "b": "Other doc."}
    snapshot = dict(documents)
    chunks = chunk_corpus(documents, max_tokens=20, overlap_tokens=5)
    assert documents == snapshot
    assert {c.doc_id for c in chunks} == {"a", "b"}


def test_chunk_is_immutable() -> None:
    chunk = chunk_document("Only one sentence here.", "d")[0]
    with pytest.raises(AttributeError):
        chunk.text = "mutated"  # type: ignore[misc]
