"""The engine: chunk, index twice, search three ways.

``"bm25"`` matches exact terms -- unbeatable on names, codes and rare
words, blind to paraphrase. ``"dense"`` matches vector geometry: robust to
wording (with a real model), prone to something topically close but
factually wrong. ``"hybrid"`` fuses their *rankings*; :mod:`minirag.fusion`
explains why ranks and not scores; ``python -m minirag.demo`` runs all three.

Retrieval runs over chunks, but results collapse to one hit per document
(its best chunk) -- otherwise a long, heavily overlapped document fills the
whole top-5 with near-identical text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from minirag import bm25, fusion
from minirag.chunker import Chunk, chunk_document
from minirag.tokenize import tokenize
from minirag.vectors import DEFAULT_DIM, DenseIndex, Embedder, HashedTfidfEmbedder

SearchMode = Literal["bm25", "dense", "hybrid"]
_MODES: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")


@dataclass(frozen=True, slots=True)
class Document:
    """An input document. ``metadata`` rides on every hit; ``where`` filters it."""

    doc_id: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result. ``score`` is mode-dependent and NOT comparable
    across modes -- BM25 is unbounded, dense a cosine in ``[0, 1]``, hybrid
    an RRF score near ``1/k``. It orders one result set, nothing more.
    """

    doc_id: str
    chunk_id: str
    text: str
    score: float
    mode: SearchMode
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class _State:
    """Everything :meth:`MiniRAG.search` needs, swapped in atomically."""

    chunks: tuple[Chunk, ...]
    metadata: Mapping[str, Mapping[str, str]]
    lexical: bm25.BM25Index
    embedder: Embedder
    dense: DenseIndex


class MiniRAG:
    """A complete retrieval engine over an in-memory corpus."""

    def __init__(
        self,
        *,
        max_tokens: int = 120,
        overlap_tokens: int = 30,
        dim: int = DEFAULT_DIM,
        rrf_k: int = 60,
        weights: Mapping[str, float] | None = None,
    ) -> None:
        """Configure chunking, embedding width and fusion.

        ``weights`` (keys ``"bm25"``, ``"dense"``, equal by default): raise
        ``"bm25"`` for identifier-heavy corpora, ``"dense"`` for paraphrases.
        """
        self._window = {"max_tokens": max_tokens, "overlap_tokens": overlap_tokens}
        self._dim = dim
        self._rrf_k = rrf_k
        self._weights = dict(weights or {"bm25": 1.0, "dense": 1.0})
        self._state: _State | None = None

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks (``0`` before :meth:`index`)."""
        return 0 if self._state is None else len(self._state.chunks)

    def index(self, documents: Sequence[Document]) -> None:
        """Chunk, tokenise and build both indexes, replacing any prior index.

        Nothing the caller passed is modified or retained: metadata is copied.

        Raises:
            ValueError: On a duplicate ``doc_id`` (results would be ambiguous).
        """
        chunks: list[Chunk] = []
        metadata: dict[str, Mapping[str, str]] = {}
        for document in documents:
            if document.doc_id in metadata:
                raise ValueError(f"duplicate doc_id: {document.doc_id!r}")
            metadata[document.doc_id] = dict(document.metadata)
            chunks.extend(chunk_document(document.text, document.doc_id, **self._window))

        tokens = tuple(tokenize(chunk.text) for chunk in chunks)
        embedder = HashedTfidfEmbedder.fit(tokens, dim=self._dim)
        self._state = _State(
            chunks=tuple(chunks),
            metadata=metadata,
            lexical=bm25.build_index(tokens),
            embedder=embedder,
            dense=DenseIndex.build(embedder, tokens),
        )

    def search(
        self, query: str, *, top_k: int = 5, mode: SearchMode = "hybrid",
        where: Mapping[str, str] | None = None,
    ) -> tuple[Hit, ...]:
        """Retrieve the best chunks for ``query``, at most one per document.

        ``where={"field": value}`` keeps documents whose metadata has every
        pair, *before* ranking: the rest never take a rank or a ``top_k`` slot.
        Empty or all-stopword queries return ``()`` rather than raising.

        Raises:
            RuntimeError: If called before :meth:`index`.
            ValueError: If ``mode`` is unknown or ``top_k < 1``.
        """
        state = self._state
        if state is None:
            raise RuntimeError("call index() before search()")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if mode not in _MODES:
            raise ValueError(f"unknown mode: {mode!r}, expected one of {_MODES}")

        query_tokens = tokenize(query)
        if not query_tokens:
            return ()
        # Over-fetch: fusion needs depth, and collapsing shrinks the list.
        ranked = self._rank(state, query_tokens, mode, max(top_k * 4, 20), dict(where or {}))
        return self._collapse(state, ranked, mode, top_k)

    def _rank(
        self, state: _State, query_tokens: tuple[str, ...], mode: SearchMode, depth: int,
        where: Mapping[str, str],
    ) -> Sequence[tuple[int, float]]:
        """Run the requested retriever(s), returning ``(chunk_index, score)``."""
        wide = len(state.chunks) if where else depth  # filtering needs the full list
        ok = {d for d, m in state.metadata.items() if where.items() <= m.items()}  # subset test

        def keep(pairs: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
            return [p for p in pairs if state.chunks[p[0]].doc_id in ok][:depth]

        lexical = keep(bm25.search(state.lexical, query_tokens, top_k=wide))
        if mode == "bm25":
            return lexical
        dense = keep(state.dense.search(state.embedder.encode([query_tokens]), top_k=wide))
        if mode == "dense":
            return dense
        ranks = {"bm25": [i for i, _ in lexical], "dense": [i for i, _ in dense]}
        return fusion.reciprocal_rank_fusion(ranks, weights=self._weights, k=self._rrf_k)

    @staticmethod
    def _collapse(
        state: _State, ranked: Sequence[tuple[int, float]], mode: SearchMode, top_k: int
    ) -> tuple[Hit, ...]:
        """Keep the best-scoring chunk per document, preserving rank order."""
        hits: dict[str, Hit] = {}
        for chunk_index, score in ranked:
            chunk = state.chunks[chunk_index]
            if chunk.doc_id in hits:
                continue
            hits[chunk.doc_id] = Hit(
                doc_id=chunk.doc_id, chunk_id=chunk.chunk_id, text=chunk.text,
                score=float(score), mode=mode, metadata=state.metadata[chunk.doc_id],
            )
            if len(hits) == top_k:
                break
        return tuple(hits.values())
