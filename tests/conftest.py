"""Shared fixtures: the tiny solar-system corpus used across the suite, and a fake
sentence-transformers module so the learned-embedding adapter is tested without torch."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
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


# --- A stand-in for the optional sentence-transformers package ------------------

FAKE_DIM = 16
# A crude "learned" space for plumbing tests only: these words share a column,
# so a paraphrase can match where BM25 sees no common term. Not a model.
SYNONYMS = {"boat": "vessel", "ship": "vessel", "surveyed": "inspection",
            "survey": "inspection", "inspected": "inspection", "inspections": "inspection"}


def fake_vector(text: str) -> list[float]:
    """Deterministic, deliberately unnormalised, so the adapter must normalise it."""
    row = [0.0] * FAKE_DIM
    for word in text.lower().replace(".", " ").replace(",", " ").split():
        word = SYNONYMS.get(word, word)
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
        row[int.from_bytes(digest, "big") % FAKE_DIM] += 3.0
    return row


class FakeSentenceTransformer:
    """Records how it was built and called; embeds with :func:`fake_vector`."""

    built: ClassVar[list[FakeSentenceTransformer]] = []

    def __init__(self, model_name_or_path: str, **kwargs: object) -> None:
        self.name, self.kwargs, self.calls = model_name_or_path, kwargs, []
        self.dim: object = FAKE_DIM
        FakeSentenceTransformer.built.append(self)

    def get_embedding_dimension(self) -> object:
        return self.dim

    def _encode(self, kind: str, texts: list[str], kwargs: dict) -> np.ndarray:
        self.calls.append((kind, list(texts), kwargs))
        return np.array([fake_vector(t) for t in texts], dtype=np.float32)

    def encode_document(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return self._encode("document", texts, kwargs)

    def encode_query(self, texts: list[str], **kwargs: object) -> np.ndarray:
        return self._encode("query", texts, kwargs)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    """Install a fake ``sentence_transformers`` module for one test; no torch, no download."""
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(FakeSentenceTransformer, "built", [])
    return FakeSentenceTransformer
