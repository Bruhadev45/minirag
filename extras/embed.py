"""Embeddings that carry their identity, so two vector spaces never mix.

A dense index only means something against queries embedded the same way:
the same model, the same weights, the same output width. Vectors from a
learned model and from minirag's hashed TF-IDF can have the same width and
still mean nothing to each other; a cosine between them is just a number.
So every embedder here names its :class:`Space`, every :class:`VectorIndex`
remembers the space it was built in, and searching it with an embedder from
any other space raises :class:`MixedSpaceError` rather than ranking anything.
A space is what an embedder *declares*: the check stops honest mistakes
(another model, revision, width or fit), not an embedder that lies about
itself, so use only embedders you trust.

Both embedders read **raw text**. The core's ``Embedder`` protocol reads
tokens (lowercased, stopwords dropped), which is right for hashing terms
and wrong for a model trained on text as people write it. So these take
strings and hand the learned model each text unchanged.

- :class:`HashedTfidfText` wraps the core's ``HashedTfidfEmbedder``. Its
  space has ``kind="hashed-tfidf"``: dense vectors, *lexical underneath*, not
  a learned semantic embedding. Its IDF weights come from the texts it was
  fitted on, so its ``revision`` fingerprints those weights (rounded, so the
  last-bit noise of ``log`` on another machine does not rename them), and fits
  over different corpora are different spaces.
- :class:`SentenceTransformerEmbedder` wraps a ``sentence_transformers``
  ``SentenceTransformer``, from the optional ``embeddings`` extra (it pulls
  in torch). Its space has ``kind="learned"`` and records the model, the
  revision you asked for and the width the model reports. It loads only
  files already on disk unless you pass ``allow_download=True``, and it never
  passes ``trust_remote_code``. A revision that is not a 40-character commit
  hash (a branch such as ``"main"``) can move, so :attr:`Space.pinned` is
  false for it. The library ignores ``revision`` for a local directory, so a
  commit hash is refused there: it would claim a pin nothing checks.

What a space does not record: the library and torch versions, the device and
settings such as ``max_seq_length``. Each can move a vector by a little, which
is why scores are rounded to ``SCORE_DECIMALS`` before ties break on the ID.

The tests drive the adapter with a fake model, so they check the plumbing,
not any model's quality.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, runtime_checkable

import numpy as np

from minirag.tokenize import tokenize
from minirag.vectors import DEFAULT_DIM, HashedTfidfEmbedder

Kind = Literal["hashed-tfidf", "learned"]
KINDS: Final = ("hashed-tfidf", "learned")
LABELS: Final = {
    "hashed-tfidf": "hashed TF-IDF (lexical, not a learned embedding)",
    "learned": "learned embedding",
}
HASHED_MODEL: Final = "minirag.HashedTfidfEmbedder"
SCORE_DECIMALS: Final = 12
"""Cosines are rounded to this many decimals, so floating-point noise cannot break a tie."""
_IDF_DECIMALS: Final = 9
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")


class MixedSpaceError(ValueError):
    """Vectors from two embedding spaces met; a cosine between them would mean nothing."""


@dataclass(frozen=True, slots=True)
class Space:
    """What makes two vectors comparable: kind, model, revision and width."""

    kind: Kind
    model: str
    revision: str
    dim: int

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        for name in ("model", "revision"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string without outer spaces")
        if type(self.dim) is not int or self.dim < 1:
            raise ValueError("dim must be an int >= 1")

    @property
    def label(self) -> str:
        """``"hashed TF-IDF (...)"`` or ``"learned embedding"``: never one for the other."""
        return LABELS[self.kind]

    @property
    def pinned(self) -> bool:
        """True if the revision names fixed weights: a fit fingerprint or a Hub commit hash."""
        return self.kind == "hashed-tfidf" or bool(_COMMIT.fullmatch(self.revision))

    def describe(self) -> str:
        """One line naming everything that identifies the space."""
        return f"{self.label}: {self.model} @ {self.revision}, {self.dim} dims"


@runtime_checkable
class TextEmbedder(Protocol):
    """Anything that embeds raw text into one named :class:`Space`."""

    @property
    def space(self) -> Space: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """An ``(len(texts), space.dim)`` array, one row per text."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """A ``(space.dim,)`` array; queries may be embedded unlike documents."""
        ...


# A ``learned`` embedder is trusted to embed each text on its own. Only the
# ``hashed-tfidf`` kind is fitted on a corpus, and only that kind's fit is
# checked by callers that care where corpus statistics come from.


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    """A tuple of plain strings; a lone string is refused rather than read as characters."""
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    values = tuple(values)
    if not all(type(v) is str for v in values):
        raise TypeError(f"{name} must all be plain strings")
    return values


def _unit_rows(raw: object, rows: int, dim: int) -> np.ndarray:
    """Check an embedder's output; return a read-only copy with L2-normalised rows.

    A zero row stays zero: a valid vector with cosine 0 to everything.
    """
    try:
        matrix = np.array(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("embedder output is not a numeric array") from exc
    if matrix.shape != (rows, dim):
        raise ValueError(f"embedder returned shape {matrix.shape}, expected {(rows, dim)}")
    if not np.isfinite(matrix).all():
        raise ValueError("embedder output has a non-finite value")
    scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    matrix = matrix / np.where(scale == 0.0, 1.0, scale)  # largest entry 1: no over/underflow
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0.0, 1.0, norms)
    matrix.flags.writeable = False
    return matrix


def _fingerprint(idf: np.ndarray) -> str:
    """Names a fit by its IDF weights, rounded so machine-level ``log`` noise keeps the name."""
    rounded = np.round(np.asarray(idf, dtype=np.float64), _IDF_DECIMALS)
    return "idf-" + hashlib.blake2b(rounded.tobytes(), digest_size=8).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class HashedTfidfText:
    """The core's hashed TF-IDF over raw text. Lexical underneath: no synonyms.

    Build it with :meth:`fit`. Built by hand, the space must describe the
    weights it is paired with, or construction fails.
    """

    space: Space
    core: HashedTfidfEmbedder

    def __post_init__(self) -> None:
        core, idf = self.core, getattr(self.core, "idf", None)
        if not (type(core) is HashedTfidfEmbedder and type(idf) is np.ndarray
                and idf.shape == (core.dim,) and type(self.space) is Space
                and self.space == Space("hashed-tfidf", HASHED_MODEL, _fingerprint(idf), core.dim)):
            raise ValueError("this space does not describe these IDF weights; use fit()")

    @classmethod
    def fit(cls, texts: Sequence[str], *, dim: int = DEFAULT_DIM) -> HashedTfidfText:
        """Fit IDF weights on ``texts``; the space's revision fingerprints them."""
        if type(dim) is not int:
            raise ValueError("dim must be an int >= 1")
        tokens = [tokenize(t) for t in _strings(texts, "texts")]
        idf = HashedTfidfEmbedder.fit(tokens, dim=dim).idf.copy()
        idf.flags.writeable = False
        return cls(Space("hashed-tfidf", HASHED_MODEL, _fingerprint(idf), dim),
                   HashedTfidfEmbedder(dim=dim, idf=idf))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.core.encode([tokenize(t) for t in _strings(texts, "texts")])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def _reported_dim(model: object) -> int:
    """The model's output width: ``get_embedding_dimension``, else its deprecated old name."""
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, name, None)
        if callable(method):
            dim = method()
            if type(dim) is int and dim >= 1:
                return dim
            break
    raise ValueError("the model does not report a fixed embedding width")


class SentenceTransformerEmbedder:
    """A sentence-transformers model as a :class:`TextEmbedder` in a ``learned`` space.

    ``model`` is a Hub ID or a local directory that you choose, never one from
    a request: loading a model reads files shipped with it. ``revision`` is
    required so the space says which weights made the vectors; a commit hash
    pins them. A local directory ignores ``revision``, so give it a label that
    is not a commit hash; its space is then reported unpinned. With
    ``allow_download=False`` (the default) a model missing from the local
    cache fails to load instead of downloading. Documents go
    through ``encode_document`` and queries through ``encode_query``, so a
    model's own query/document prompts apply; both normalise the vectors.

    Raises:
        ValueError: A bad ``batch_size``, model name or revision (including a
            commit hash for a local directory), or a model that does not
            report a fixed output width.
        ImportError: ``sentence_transformers`` is not installed; an import
            error from inside it (a broken torch, say) is raised unchanged.
    """

    __slots__ = ("_batch_size", "_model", "_space")

    def __init__(
        self, model: str, *, revision: str, allow_download: bool = False,
        batch_size: int = 32, device: str | None = None,
    ) -> None:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be an int >= 1")
        Space("learned", model, revision, 1)  # check the names before importing anything
        if os.path.isdir(model) and _COMMIT.fullmatch(revision):  # the library's own test
            raise ValueError("a local model directory ignores revision, so a commit hash "
                             "would claim a pin nothing checks; use another label")
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            if exc.name != "sentence_transformers":
                raise
            raise ImportError(
                "SentenceTransformerEmbedder needs the optional embeddings extra, which pulls "
                'in torch: pip install -e ".[embeddings]". HashedTfidfText needs nothing extra.'
            ) from exc
        self._model = module.SentenceTransformer(
            model, revision=revision, device=device,
            local_files_only=not allow_download, trust_remote_code=False,
        )
        self._batch_size = batch_size
        self._space = Space("learned", model, revision, _reported_dim(self._model))

    @property
    def space(self) -> Space:
        return self._space

    def _encode(self, method: str, texts: list[str]) -> np.ndarray:
        return getattr(self._model, method)(
            texts, batch_size=self._batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Unit rows, checked: the wrong shape or a non-finite value raises ``ValueError``."""
        texts = _strings(texts, "texts")
        if not texts:
            return np.zeros((0, self._space.dim))
        return _unit_rows(self._encode("encode_document", list(texts)), len(texts), self._space.dim)

    def embed_query(self, text: str) -> np.ndarray:
        """One unit vector, checked like :meth:`embed_documents`."""
        (query,) = _strings([text], "query")
        return _unit_rows(self._encode("encode_query", [query]), 1, self._space.dim)[0]


@dataclass(frozen=True, slots=True, eq=False)
class VectorIndex:
    """Unit vectors for named passages, tagged with the space that made them.

    Build it with :meth:`build`; the matrix is a read-only copy, row ``i``
    belonging to ``ids[i]``. Search is brute force, like the core's
    ``DenseIndex``, but ties break on the ID rather than on row position.
    """

    space: Space
    ids: tuple[str, ...]
    matrix: np.ndarray

    @classmethod
    def build(cls, embedder: TextEmbedder, ids: Sequence[str], texts: Sequence[str]) -> VectorIndex:
        """Embed ``texts`` as given, one row per ID.

        Raises:
            TypeError: If the IDs or texts are not plain strings, or the
                embedder has no :class:`Space`.
            ValueError: If IDs repeat, do not pair up with the texts, or the
                embedder's output has the wrong shape or a non-finite value.
        """
        ids, texts = _strings(ids, "ids"), _strings(texts, "texts")
        space = getattr(embedder, "space", None)
        if type(space) is not Space:
            raise TypeError("the embedder must name its Space")
        if len(set(ids)) != len(ids) or len(ids) != len(texts):
            raise ValueError("ids must be unique, one per text")
        raw = embedder.embed_documents(texts) if texts else np.zeros((0, space.dim))
        return cls(space, ids, _unit_rows(raw, len(texts), space.dim))

    def search(
        self, embedder: TextEmbedder, query: str, *, top_k: int
    ) -> tuple[tuple[str, float], ...]:
        """The ``top_k`` best ``(id, cosine)`` pairs, ties broken by ID.

        Cosines are rounded to :data:`SCORE_DECIMALS`, so scores equal up to
        floating-point noise tie. As in the core, a cosine of zero or below is
        no match and is dropped. The embedder's space is read once.

        Raises:
            TypeError: If ``query`` is not a plain string.
            MixedSpaceError: If ``embedder`` is from another space.
            ValueError: If ``top_k`` is not an int >= 1 or the query vector
                has the wrong shape or a non-finite value.
        """
        if type(query) is not str:
            raise TypeError("query must be a plain string")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be an int >= 1")
        space = getattr(embedder, "space", None)
        if type(space) is not Space or space != self.space:
            theirs = space.describe() if type(space) is Space else "not a Space"
            raise MixedSpaceError(f"index is {self.space.describe()}; query embedder is {theirs}")
        if not self.ids:
            return ()
        raw = np.asarray(embedder.embed_query(query))
        query_vector = _unit_rows(raw.reshape(1, -1) if raw.ndim == 1 else raw, 1, space.dim)[0]
        sims = np.round(self.matrix @ query_vector, SCORE_DECIMALS)
        order = sorted(range(len(self.ids)), key=lambda i: (-sims[i], self.ids[i]))
        return tuple((self.ids[i], float(sims[i])) for i in order[:top_k] if sims[i] > 0.0)
