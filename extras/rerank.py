"""Reranking: a second, slower opinion on the first stage's shortlist.

``MiniRAG.search`` is a *first stage*: it scores every chunk cheaply and
keeps the best few. A reranker then reads only that shortlist and re-orders
it with a scorer too expensive to run over the whole corpus. The classic
choice is a cross-encoder, which reads query and passage *together* instead
of comparing two vectors computed apart -- slower per pair and often more
accurate, which is why it only sees the top ``candidates``.

**Why it lives here.** The core sits at 799 of its 800 lines, so there is no
room for a ``search(..., rerank=...)`` hook. :func:`apply` wraps the engine
from outside instead: same filter, same one-hit-per-document collapse, then
a re-order. Because it only ever re-orders what ``search`` returned, a
reranker cannot surface a document the ``where`` filter excluded -- that
limits which hits come back, not what the reranker reads: it is trusted
code and sees every candidate's text. It also sees only each document's
best first-stage chunk, never the document's other chunks.

Two rerankers ship, both behind the one-method :class:`Reranker` protocol:

- :class:`BM25Reranker` -- no dependency. Re-scores the shortlist's chunk
  text against the full query with BM25, whose IDF is computed over the
  shortlist itself: a term every candidate shares counts for little, a term
  that tells them apart counts for a lot. Useful after ``hybrid``, where
  RRF threw the BM25 scores away and kept only ranks.
- :class:`CrossEncoderReranker` -- wraps a ``sentence_transformers``
  ``CrossEncoder``. The package is optional and imported only when you build
  one; it pulls in torch, and the model downloads on first use. It calls
  only ``CrossEncoder(model_name)`` and ``predict(pairs, batch_size=...)``,
  and has not run against a real install: the tests drive it with a fake
  model, so nothing here measures a real cross-encoder.

Reranker scores order one shortlist and nothing more: a cross-encoder logit
is not a probability, and BM25 over five passages is not comparable to BM25
over the corpus.
"""

from __future__ import annotations

import importlib
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from minirag import Hit, MiniRAG, SearchMode, tokenize
from minirag import bm25 as _bm25

DEFAULT_CANDIDATES = 20
"""Shortlist size. The engine already over-fetches 20 chunks per query, and
on the 20-document fixture this is the whole corpus; raise it for real data."""


@runtime_checkable
class Reranker(Protocol):
    """Anything that scores ``texts`` against ``query``."""

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """One finite score per text, aligned with ``texts``; higher is better."""
        ...


@dataclass(frozen=True, slots=True)
class RerankedHit:
    """A first-stage :class:`~minirag.Hit` and the score that re-ordered it.

    ``hit`` is untouched, so ``hit.score`` and ``hit.mode`` still describe
    the first stage; ``first_stage_rank`` (1-based) records where it was.
    """

    hit: Hit
    score: float
    first_stage_rank: int

    @property
    def doc_id(self) -> str:
        """Shortcut for ``hit.doc_id``."""
        return self.hit.doc_id


@dataclass(frozen=True, slots=True)
class BM25Reranker:
    """BM25 over the shortlist alone; ``k1`` and ``b`` as in :mod:`minirag.bm25`."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        _bm25.build_index((), k1=self.k1, b=self.b)  # the engine's own validation

    def score(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        """BM25 of each text against ``query``; ``0.0`` shares no query term."""
        index = _bm25.build_index([tokenize(text) for text in texts], k1=self.k1, b=self.b)
        return _bm25.score_query(index, tokenize(query))


class CrossEncoderReranker:
    """Score ``(query, text)`` pairs with a sentence-transformers cross-encoder.

    ``model_name`` is any cross-encoder the package can load, e.g.
    ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``; there is no default, so no
    download happens without you naming the model. Loading a model can run
    code shipped with it, so choose ``model_name`` yourself -- never from a
    request -- and prefer a vetted, pinned local path. This adapter never
    passes ``trust_remote_code``.

    Raises:
        ValueError: If ``batch_size < 1``.
        ImportError: If ``sentence_transformers`` is not installed.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise ImportError(
                "CrossEncoderReranker needs the optional sentence-transformers package "
                "(it pulls in torch): pip install sentence-transformers. "
                "BM25Reranker needs nothing extra."
            ) from exc
        self.model_name = model_name
        self._batch_size = batch_size
        self._model = module.CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        """One model score per text, aligned with ``texts``.

        Raises:
            ValueError: If the model returns anything but one score per text
                (a multi-label model, say), so the caller cannot misalign them.
        """
        if not texts:
            return ()
        pairs = [(query, text) for text in texts]
        raw = np.asarray(self._model.predict(pairs, batch_size=self._batch_size), dtype=float)
        if raw.shape != (len(texts),):
            raise ValueError(
                f"{self.model_name} returned shape {raw.shape} for {len(texts)} texts; "
                "expected one score per text"
            )
        return tuple(raw.tolist())


def _checked(scores: Sequence[object], expected: int) -> tuple[float, ...]:
    """Reject anything that would silently corrupt the order."""
    values = tuple(scores)
    if len(values) != expected:
        raise ValueError(f"reranker returned {len(values)} scores for {expected} texts")
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"reranker scores must be real numbers, got {type(value).__name__}")
        try:
            number = float(value)
        except OverflowError:  # an int too large for a float, e.g. 10**400
            number = math.inf
        if not math.isfinite(number):
            raise ValueError(f"reranker scores must be finite, got {number!r}")
        checked.append(number)
    return tuple(checked)


def apply(
    engine: MiniRAG,
    query: str,
    reranker: Reranker,
    *,
    top_k: int = 5,
    candidates: int = DEFAULT_CANDIDATES,
    mode: SearchMode = "hybrid",
    where: Mapping[str, str] | None = None,
) -> tuple[RerankedHit, ...]:
    """Search, then re-order the top ``candidates`` hits with ``reranker``.

    The first stage is ``engine.search(query, top_k=candidates, mode=mode,
    where=where)``, unchanged. Each hit's chunk text is scored; ties keep the
    first-stage order, so the result is deterministic. A query the engine
    answers with nothing never reaches the reranker.

    Raises:
        ValueError: If ``top_k < 1``, ``candidates < top_k``, or the reranker
            returns the wrong number of scores or a non-finite one.
        TypeError: If the reranker returns a non-numeric score.
        RuntimeError: If ``engine`` has not been indexed.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if candidates < top_k:
        raise ValueError(f"candidates ({candidates}) must be >= top_k ({top_k})")
    hits = engine.search(query, top_k=candidates, mode=mode, where=where)
    if not hits:
        return ()
    scores = _checked(reranker.score(query, [hit.text for hit in hits]), len(hits))
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    return tuple(
        RerankedHit(hit=hits[i], score=scores[i], first_stage_rank=i + 1) for i in order[:top_k]
    )
