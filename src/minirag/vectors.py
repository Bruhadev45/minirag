"""Dense vectors without a model download, plus brute-force cosine search.

Dense retrieval has two halves that people usually conflate: ``text ->
vector`` (the embedder -- a model, in the real world) and ``vector ->
neighbours`` (the index -- an ANN structure). minirag implements both in
numpy so the repo runs offline, then marks the seam where you would swap in
something better.

**The embedder** builds a vector space arithmetically instead of learning
one. Hash each term to a column with ``blake2b`` -- not Python's ``hash()``,
which is salted per process, so vectors would differ between runs. Any term
maps into ``dim`` columns, so the vocabulary grows forever without resizing
anything; that is the "hashing trick", and the price is collisions (rare at
``dim=4096``). Weight each column by ``tf * idf`` -- rare terms carry the
meaning, as in BM25 -- then L2-normalise so cosine is a plain dot product.
That is genuinely a dense space but still *lexical underneath*: "car" and
"automobile" hash to unrelated columns, so it cannot match synonyms, the
selling point of learned embeddings. Collisions also mean an
out-of-vocabulary query still scores above zero against something.

**The seam** is :class:`Embedder`, a ``Protocol`` with one method: anything
returning an ``(n, dim)`` array of L2-normalised rows works, be it
sentence-transformers, an embeddings API call or a local ONNX model.
:class:`DenseIndex` and :class:`~minirag.engine.MiniRAG` never name
``HashedTfidfEmbedder``; they only need the protocol.

**The index is brute force on purpose.** ``matrix @ query`` compares
against every vector: exact, three lines, and faster on a few thousand
chunks than any approximate structure once you count build time. HNSW/IVF
only win in the millions, by *trading away recall*.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_DIM = 4096


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns token sequences into unit-norm row vectors."""

    dim: int

    def encode(self, documents: Sequence[Sequence[str]]) -> np.ndarray:
        """Return an ``(len(documents), dim)`` float64 array."""
        ...


def _column(term: str, dim: int) -> int:
    """Map a term to a column index, stably across processes and machines."""
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


@dataclass(frozen=True, slots=True, eq=False)
class HashedTfidfEmbedder:
    """A fitted hashed TF-IDF embedder. Immutable; :meth:`fit` returns new.

    ``dim`` is the vector width -- larger means fewer hash collisions.
    ``idf`` is a ``(dim,)`` array of per-column weights learned by
    :meth:`fit`; pass all-ones to degrade gracefully to plain TF.
    """

    dim: int
    idf: np.ndarray

    @classmethod
    def fit(cls, corpus: Sequence[Sequence[str]], *, dim: int = DEFAULT_DIM) -> HashedTfidfEmbedder:
        """Learn column IDF weights from a tokenised corpus.

        Uses the smoothed form ``ln((N + 1) / (df + 1)) + 1``, which is
        always positive -- unlike BM25's IDF we are building a vector to be
        normalised, and negative weights would flip a term's direction in
        the space rather than merely discount it.

        Raises:
            ValueError: If ``dim < 1``.
        """
        if dim < 1:
            raise ValueError("dim must be >= 1")
        n_docs = len(corpus)
        doc_freq = np.zeros(dim, dtype=np.float64)
        for tokens in corpus:
            for column in {_column(t, dim) for t in tokens}:
                doc_freq[column] += 1.0
        idf = np.log((n_docs + 1.0) / (doc_freq + 1.0)) + 1.0
        return cls(dim=dim, idf=idf)

    def encode(self, documents: Sequence[Sequence[str]]) -> np.ndarray:
        """Embed token sequences as L2-normalised rows.

        Sub-linear term frequency (``1 + log f``) is used for the same
        reason BM25 has ``k1``: the tenth occurrence of a word says far less
        than the second. Empty documents produce an all-zero row, a valid
        vector with cosine 0 to everything.
        """
        matrix = np.zeros((len(documents), self.dim), dtype=np.float64)
        for row, tokens in enumerate(documents):
            for term, freq in Counter(tokens).items():
                matrix[row, _column(term, self.dim)] += 1.0 + math.log(freq)
        matrix *= self.idf
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0.0, 1.0, norms)


@dataclass(frozen=True, slots=True, eq=False)
class DenseIndex:
    """Brute-force cosine index over a matrix of unit-norm row vectors."""

    matrix: np.ndarray

    @classmethod
    def build(cls, embedder: Embedder, corpus: Sequence[Sequence[str]]) -> DenseIndex:
        """Encode ``corpus`` once and keep the matrix. The input is not kept."""
        return cls(matrix=embedder.encode(corpus))

    def search(self, query_vector: np.ndarray, *, top_k: int = 10) -> tuple[tuple[int, float], ...]:
        """Return the ``top_k`` ``(row_index, cosine)`` pairs, best first.

        Because every row and the query are unit-norm, the whole search is
        one matrix-vector product; ``argpartition`` then finds the top k in
        O(n) and only the survivors get sorted.

        Raises:
            ValueError: If ``query_vector`` does not match the index width.
        """
        if self.matrix.size == 0:
            return ()
        if query_vector.shape[-1] != self.matrix.shape[1]:
            raise ValueError(
                f"query has {query_vector.shape[-1]} dims, index has {self.matrix.shape[1]}"
            )
        sims = self.matrix @ query_vector.reshape(-1)
        k = min(top_k, sims.shape[0])
        candidates = np.argpartition(-sims, k - 1)[:k]
        ranked = sorted(candidates, key=lambda i: (-sims[i], i))
        return tuple((int(i), float(sims[i])) for i in ranked if sims[i] > 0.0)
