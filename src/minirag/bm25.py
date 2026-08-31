"""BM25 Okapi, built from an inverted index, term by term.

For a query Q and document D, BM25 sums one contribution per query term::

    score(D, Q) = SUM over q in Q of  IDF(q) * TF_sat(q, D)

                                  f(q, D) * (k1 + 1)
    TF_sat(q, D) = -------------------------------------------------
                   f(q, D) + k1 * (1 - b + b * |D| / avgdl)

``f(q, D)`` is the raw count of ``q`` in ``D``, ``|D|`` the document length
in tokens, ``avgdl`` the corpus average. Three ideas, multiplied:

**IDF -- how surprising is this term?** ``ln(1 + (N - n(q) + 0.5) /
(n(q) + 0.5))``, with ``n(q)`` documents containing ``q`` out of ``N``. A
term in every document scores near 0, a term in one document scores high:
this makes "the" free and "perihelion" expensive. The ``1 +`` wrapper keeps
it non-negative, unlike the textbook form -- without it a term in over half
the corpus scores *negatively*, so deleting a word could improve rank.

**TF saturation (k1) -- how much does repetition help?** A document saying
"Mars" 50 times is not 50x more about Mars than one saying it once. The
ratio is a hyperbola in ``f``: steep from 0, then flattening toward a
ceiling of ``k1 + 1``, with ``k1`` setting the knee. ``k1=0`` makes it
binary, ``k1 -> inf`` linear; ``1.5`` is our default.

**Length normalisation (b) -- is this document just long?** A long document
has more chances to contain any term, so ``|D| / avgdl`` divides it down.
``b`` interpolates: ``0`` ignores length, ``1`` normalises fully, ``0.75``
is the default. It sits in the *denominator* beside ``k1``, so a long
document does not merely lose score -- it also saturates more slowly.

The inverted index makes this fast: scoring only needs documents holding at
least one query term, and a ``term -> [(doc, count), ...]`` map jumps
straight to them.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BM25Index:
    """An immutable inverted index plus the statistics BM25 needs.

    ``postings`` maps ``term -> ((doc_index, term_frequency), ...)``;
    ``doc_lengths`` is positionally aligned with the corpus passed to
    :func:`build_index`; ``avg_doc_length`` is ``avgdl`` above.
    """

    postings: Mapping[str, tuple[tuple[int, int], ...]]
    doc_lengths: tuple[int, ...]
    avg_doc_length: float
    k1: float
    b: float

    @property
    def n_docs(self) -> int:
        """Number of indexed documents."""
        return len(self.doc_lengths)


def build_index(corpus: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """Build an inverted index over already-tokenised documents.

    ``corpus`` is one token sequence per document; it is neither mutated
    nor retained. An empty corpus is legal and yields an index that scores
    everything as nothing.

    Raises:
        ValueError: If ``k1 < 0`` or ``b`` is outside ``[0, 1]``.
    """
    if k1 < 0:
        raise ValueError("k1 must be >= 0")
    if not 0.0 <= b <= 1.0:
        raise ValueError("b must be in [0, 1]")

    accumulator: dict[str, list[tuple[int, int]]] = {}
    lengths: list[int] = []
    for doc_index, tokens in enumerate(corpus):
        lengths.append(len(tokens))
        for term, freq in Counter(tokens).items():
            accumulator.setdefault(term, []).append((doc_index, freq))

    return BM25Index(
        postings={term: tuple(plist) for term, plist in accumulator.items()},
        doc_lengths=tuple(lengths),
        avg_doc_length=(sum(lengths) / len(lengths)) if lengths else 0.0,
        k1=k1,
        b=b,
    )


def idf(index: BM25Index, term: str) -> float:
    """Inverse document frequency of ``term``, smoothed and non-negative.

    Unknown terms return ``0.0`` rather than raising: a query word absent
    from the corpus should contribute nothing, not crash the search.
    """
    n_docs = index.n_docs
    if n_docs == 0:
        return 0.0
    containing = len(index.postings.get(term, ()))
    if containing == 0:
        return 0.0
    return math.log(1.0 + (n_docs - containing + 0.5) / (containing + 0.5))


def score_query(index: BM25Index, query_tokens: Sequence[str]) -> tuple[float, ...]:
    """Score every document against ``query_tokens``.

    Walks one postings list per *distinct* query term, so cost scales with
    the number of matching documents, not corpus size. Repeated query terms
    count once -- BM25 has no query-side term frequency component. Returns
    one score per document, positionally aligned with the corpus; a
    document sharing no term with the query scores exactly ``0.0``.
    """
    scores = [0.0] * index.n_docs
    if index.n_docs == 0 or index.avg_doc_length == 0.0:
        return tuple(scores)

    k1, b = index.k1, index.b
    for term in set(query_tokens):
        postings = index.postings.get(term)
        if not postings:
            continue
        term_idf = idf(index, term)
        for doc_index, freq in postings:
            rel_len = index.doc_lengths[doc_index] / index.avg_doc_length
            denominator = freq + k1 * (1.0 - b + b * rel_len)
            scores[doc_index] += term_idf * freq * (k1 + 1.0) / denominator
    return tuple(scores)


def search(
    index: BM25Index, query_tokens: Sequence[str], *, top_k: int = 10
) -> tuple[tuple[int, float], ...]:
    """Return the ``top_k`` ``(doc_index, score)`` pairs, best first.

    Zero-scoring documents are dropped: in a lexical system a score of 0
    means "shares no term with the query", not a weak match but no match at
    all. Ties break on ascending document index, so results are
    deterministic across runs.
    """
    scored = [(i, s) for i, s in enumerate(score_query(index, query_tokens)) if s > 0.0]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return tuple(scored[:top_k])
