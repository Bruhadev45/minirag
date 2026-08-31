"""Tokeniser behaviour, including the edge cases that break retrieval."""

from __future__ import annotations

import pytest

from minirag.tokenize import STOPWORDS, tokenize


def test_lowercases_and_strips_punctuation() -> None:
    assert tokenize("Mars, Jupiter; SATURN!", remove_stopwords=False) == (
        "mars",
        "jupiter",
        "saturn",
    )


def test_keeps_duplicates_because_bm25_needs_term_frequency() -> None:
    assert tokenize("ice ice ice", remove_stopwords=False) == ("ice", "ice", "ice")


def test_possessives_are_truncated_at_the_apostrophe() -> None:
    assert tokenize("Earth's oceans", remove_stopwords=False) == ("earth", "oceans")


def test_digits_survive() -> None:
    assert "1846" in tokenize("Neptune was found in 1846")


def test_stopwords_removed_by_default() -> None:
    tokens = tokenize("the storm is in the atmosphere of the planet")
    assert tokens == ("storm", "atmosphere", "planet")
    assert not set(tokens) & STOPWORDS


def test_stopwords_kept_when_disabled() -> None:
    assert "the" in tokenize("the storm", remove_stopwords=False)


def test_all_stopword_query_is_empty_not_none() -> None:
    assert tokenize("the of and") == ()


def test_empty_string() -> None:
    assert tokenize("") == ()


def test_non_string_raises() -> None:
    with pytest.raises(TypeError):
        tokenize(b"bytes")  # type: ignore[arg-type]


def test_hyphenated_words_split_into_two_terms() -> None:
    # Documented behaviour, not an accident: worth pinning so a future
    # regex change is a deliberate decision.
    assert tokenize("co-orbital", remove_stopwords=False) == ("co", "orbital")
