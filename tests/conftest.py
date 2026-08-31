"""Shared fixtures: the tiny solar-system corpus used across the suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minirag import Document

FIXTURE = Path(__file__).parent / "fixtures" / "corpus.json"


@pytest.fixture(scope="session")
def raw_corpus() -> dict:
    """The parsed ``corpus.json`` fixture."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def documents(raw_corpus: dict) -> tuple[Document, ...]:
    """The corpus as :class:`minirag.Document` objects."""
    return tuple(
        Document(doc_id=d["doc_id"], text=d["text"], metadata={"title": d["title"]})
        for d in raw_corpus["documents"]
    )


@pytest.fixture(scope="session")
def query_cases(raw_corpus: dict) -> tuple[tuple[str, str], ...]:
    """``(query, expected_doc_id)`` pairs for the end-to-end assertions."""
    return tuple((q["query"], q["expected_doc_id"]) for q in raw_corpus["queries"])
