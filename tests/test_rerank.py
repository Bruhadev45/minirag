"""Reranking: the BM25 math, the re-ordering contract, and a fake cross-encoder.

``extras/rerank.py`` makes three promises worth pinning: it only re-orders
what ``search`` returned (so the ``where`` filter still holds), it never
lets a malformed score scramble the order silently, and it imports
``sentence_transformers`` only when asked to. The cross-encoder runs against
a fake module here -- these tests check the adapter's plumbing, not the
quality of any real model.
"""

from __future__ import annotations

import math
import subprocess
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from extras.rerank import (
    DEFAULT_CANDIDATES,
    BM25Reranker,
    CrossEncoderReranker,
    RerankedHit,
    Reranker,
    apply,
)
from minirag import Document, MiniRAG

REPO = Path(__file__).resolve().parent.parent


class Scripted:
    """A reranker that scores each text with ``rule`` and records every call."""

    def __init__(self, rule: Callable[[str], Any]) -> None:
        self.rule = rule
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[Any]:
        self.calls.append((query, tuple(texts)))
        return [self.rule(text) for text in texts]


@pytest.fixture(scope="module")
def engine() -> MiniRAG:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(
        [
            Document(
                "europa",
                "Europa is a moon of Jupiter with an ocean under its ice.",
                {"kind": "moon"},
            ),
            Document(
                "titan", "Titan is a moon of Saturn with lakes of liquid methane.", {"kind": "moon"}
            ),
            Document("io", "Io is a volcanic moon of Jupiter, heated by tides.", {"kind": "moon"}),
            Document(
                "mars",
                "Mars has polar caps of water ice and a thin atmosphere.",
                {"kind": "planet"},
            ),
        ]
    )
    return rag


@pytest.fixture(scope="module")
def fixture_engine(documents: tuple[Document, ...]) -> MiniRAG:
    """The fixture corpus, indexed exactly as the README quickstart does."""
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(documents)
    return rag


# --- BM25Reranker ------------------------------------------------------------


def test_bm25_reranker_matches_the_formula_by_hand() -> None:
    # "red" is in 2 of 3 texts: IDF = ln(1 + 1.5 / 2.5). avgdl = 2. Text 0 has
    # f=1 at average length, so TF_sat = 1; text 1 has f=2 at 1.5x average:
    # 2 * 2.5 / (2 + 1.5 * (0.25 + 0.75 * 1.5)) = 5 / 4.0625.
    scores = BM25Reranker().score("red", ["Mars red", "red, red planet", "blue"])
    idf = math.log(1.6)
    assert scores == pytest.approx((idf, idf * 5 / 4.0625, 0.0))


def test_bm25_reranker_idf_comes_from_the_shortlist() -> None:
    # Every candidate says "ice", so it barely separates them; "ring" does.
    scores = BM25Reranker().score("ice ring", ["ice ring", "ice dust", "ice rock"])
    assert scores[1] == scores[2]
    assert scores[0] > 5 * scores[1] > 0.0


def test_bm25_reranker_handles_nothing_to_score() -> None:
    assert BM25Reranker().score("ice", []) == ()
    assert BM25Reranker().score("the", ["ice", "rock"]) == (0.0, 0.0)  # all-stopword query


@pytest.mark.parametrize(("k1", "b"), [(-0.1, 0.75), (1.5, 1.5), (1.5, -0.1)])
def test_bm25_reranker_rejects_bad_parameters_at_construction(k1: float, b: float) -> None:
    with pytest.raises(ValueError):
        BM25Reranker(k1=k1, b=b)


def test_both_rerankers_satisfy_the_protocol(fake_st: type) -> None:
    assert isinstance(BM25Reranker(), Reranker)
    assert isinstance(CrossEncoderReranker("fake/model"), Reranker)


# --- apply -------------------------------------------------------------------


def test_apply_reorders_by_reranker_score_and_records_provenance(engine: MiniRAG) -> None:
    first = engine.search("moon ice", top_k=DEFAULT_CANDIDATES)
    assert len(first) == 4  # all four documents, so the reversal is not trivial
    last_first = {hit.text: rank for rank, hit in enumerate(first)}
    reranked = apply(engine, "moon ice", Scripted(last_first.__getitem__), top_k=len(first))
    assert [r.hit for r in reranked] == list(reversed(first))  # the Hits themselves, untouched
    assert [r.first_stage_rank for r in reranked] == list(range(len(first), 0, -1))
    assert [r.score for r in reranked] == sorted((r.score for r in reranked), reverse=True)
    assert reranked[0].doc_id == first[-1].doc_id


def test_apply_breaks_ties_on_the_first_stage_order(engine: MiniRAG) -> None:
    first = engine.search("moon ice", top_k=3)
    reranked = apply(engine, "moon ice", Scripted(lambda _: 1.0), top_k=3)
    assert reranked == tuple(RerankedHit(hit, 1.0, i) for i, hit in enumerate(first, start=1))


def test_apply_reranks_only_the_candidates_and_returns_top_k(engine: MiniRAG) -> None:
    reranker = Scripted(len)
    reranked = apply(engine, "moon ice", reranker, top_k=1, candidates=2)
    [(query, texts)] = reranker.calls
    assert query == "moon ice"
    assert texts == tuple(hit.text for hit in engine.search("moon ice", top_k=2))
    assert len(reranked) == 1
    assert reranked[0].score == max(len(text) for text in texts)


def test_apply_cannot_surface_a_document_the_filter_excluded(engine: MiniRAG) -> None:
    # The scorer would put mars first; the filter means it never sees mars.
    reranker = Scripted(lambda text: 100.0 if text.startswith("Mars") else 1.0)
    reranked = apply(engine, "ice", reranker, mode="bm25", where={"kind": "moon"})
    assert [r.doc_id for r in reranked] == ["europa"]
    assert all(not text.startswith("Mars") for _, texts in reranker.calls for text in texts)


def test_apply_skips_the_reranker_when_search_finds_nothing(engine: MiniRAG) -> None:
    reranker = Scripted(lambda _: 1.0)
    assert apply(engine, "the of and", reranker) == ()
    assert apply(engine, "ice", reranker, where={"kind": "comet"}) == ()
    assert reranker.calls == []


@pytest.mark.parametrize(("top_k", "candidates"), [(0, 20), (5, 4)])
def test_apply_rejects_impossible_sizes(engine: MiniRAG, top_k: int, candidates: int) -> None:
    reranker = Scripted(lambda _: 1.0)
    with pytest.raises(ValueError):
        apply(engine, "ice", reranker, top_k=top_k, candidates=candidates)
    assert reranker.calls == []


def test_apply_needs_an_indexed_engine() -> None:
    with pytest.raises(RuntimeError, match="index"):
        apply(MiniRAG(), "ice", BM25Reranker())


def test_apply_rejects_a_reranker_that_drops_or_adds_scores(engine: MiniRAG) -> None:
    class Short:
        def score(self, query: str, texts: Sequence[str]) -> list[float]:
            return [1.0] * (len(texts) - 1)

    with pytest.raises(ValueError, match="scores for"):
        apply(engine, "moon", Short())


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (math.nan, ValueError),
        (math.inf, ValueError),
        (10**400, ValueError),
        ("1.0", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_apply_rejects_scores_that_would_scramble_the_order(
    engine: MiniRAG, bad: object, error: type[Exception]
) -> None:
    with pytest.raises(error):
        apply(engine, "moon", Scripted(lambda _: bad))


def test_apply_accepts_numpy_scores_and_returns_python_floats(engine: MiniRAG) -> None:
    reranked = apply(engine, "moon", Scripted(lambda text: np.float32(len(text))))
    assert all(type(r.score) is float for r in reranked)


# --- On the fixture corpus ---------------------------------------------------


def test_bm25_rerank_keeps_every_labelled_answer_first(
    fixture_engine: MiniRAG, query_cases: tuple[tuple[str, str], ...]
) -> None:
    # A tripwire like tests/test_eval.py: hybrid already ranks all five first,
    # and the lexical second opinion must not lose one of them.
    for query, expected in query_cases:
        assert apply(fixture_engine, query, BM25Reranker())[0].doc_id == expected, query


def test_bm25_rerank_lifts_europa_on_the_readme_question(fixture_engine: MiniRAG) -> None:
    # Measured, and quoted in the README: hybrid puts the Moon and Pluto above
    # Europa. Re-scored lexically, Europa's chunk ("ocean", "ice") beats Pluto's
    # ("ice" alone), but not the Moon's ("moon", "ocean"): Europa's text says
    # "moons", and with no stemmer that never matches "moon".
    query = "which moon has an ocean under the ice"
    assert [h.doc_id for h in fixture_engine.search(query, top_k=3)] == ["moon", "pluto", "europa"]
    reranked = apply(fixture_engine, query, BM25Reranker(), top_k=3)
    assert [(r.doc_id, r.first_stage_rank, round(r.score, 4)) for r in reranked] == [
        ("moon", 1, 1.7158),
        ("europa", 3, 1.5657),
        ("pluto", 2, 1.3522),
    ]


# --- CrossEncoderReranker, against a fake sentence_transformers ---------------


class FakeCrossEncoder:
    """Stands in for ``sentence_transformers.CrossEncoder``; scores word overlap."""

    loaded: ClassVar[list[str]] = []
    calls: ClassVar[list[tuple[list[tuple[str, str]], int]]] = []
    output: ClassVar[Callable[[list[tuple[str, str]]], object] | None] = None

    def __init__(self, model_name: str) -> None:
        FakeCrossEncoder.loaded.append(model_name)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> object:
        FakeCrossEncoder.calls.append((list(pairs), batch_size))
        if FakeCrossEncoder.output is not None:
            return FakeCrossEncoder.output(pairs)
        overlap = [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]
        return np.asarray(overlap, dtype=np.float32)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[FakeCrossEncoder]:
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(FakeCrossEncoder, "loaded", [])
    monkeypatch.setattr(FakeCrossEncoder, "calls", [])
    monkeypatch.setattr(FakeCrossEncoder, "output", None)
    return FakeCrossEncoder


def test_cross_encoder_loads_the_named_model_and_scores_pairs(fake_st: type) -> None:
    reranker = CrossEncoderReranker("fake/model", batch_size=8)
    scores = reranker.score("moon ocean", ["an ocean moon", "rock", "moon"])
    assert fake_st.loaded == ["fake/model"]
    assert fake_st.calls == [
        ([("moon ocean", "an ocean moon"), ("moon ocean", "rock"), ("moon ocean", "moon")], 8)
    ]
    assert scores == (2.0, 0.0, 1.0)
    assert all(type(s) is float for s in scores)


def test_cross_encoder_drives_apply_end_to_end(fake_st: type, engine: MiniRAG) -> None:
    reranked = apply(engine, "liquid methane lakes", CrossEncoderReranker("fake/model"))
    assert reranked[0].doc_id == "titan"
    assert [r.score for r in reranked] == sorted((r.score for r in reranked), reverse=True)


def test_cross_encoder_does_not_call_the_model_for_nothing(fake_st: type) -> None:
    assert CrossEncoderReranker("fake/model").score("moon", []) == ()
    assert fake_st.calls == []


@pytest.mark.parametrize(
    "output",
    [lambda pairs: np.zeros((len(pairs), 3)), lambda pairs: np.zeros(len(pairs) + 1)],
    ids=["multi-label", "too-many"],
)
def test_cross_encoder_rejects_output_it_cannot_align(
    fake_st: type, output: Callable[[list[tuple[str, str]]], object]
) -> None:
    fake_st.output = output
    with pytest.raises(ValueError, match="one score per text"):
        CrossEncoderReranker("fake/model").score("moon", ["a", "b"])


def test_cross_encoder_without_the_package_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import now fails
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_cross_encoder_rejects_a_bad_batch_size_before_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ValueError, match="batch_size"):
        CrossEncoderReranker("fake/model", batch_size=0)


def test_importing_the_module_does_not_import_the_optional_package() -> None:
    code = (
        "import sys, extras.rerank; "
        "assert 'sentence_transformers' not in sys.modules and 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)
