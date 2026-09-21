"""Split documents into overlapping, sentence-aligned chunks.

**Why chunk at all?** Retrieval returns whole units. If the unit is a
40-page document the answer is buried in 39 pages of noise -- one matching
sentence barely moves BM25's length-normalised score -- and if it is a
single sentence you retrieve fragments too small to use.

**Why overlap?** (the part people skip and then regret) A hard boundary at
token 120 will eventually land inside the one passage that answers a
query::

    ... Europa is covered in water ice. | Beneath that crust lies a
    salty ocean twice the volume of Earth's. ...

Split at ``|`` and no chunk holds "Europa" *and* "ocean" together, so the
query matches each half weakly instead of one chunk strongly -- and dense
retrieval fares worse, because the second chunk's embedding has no idea
what "that crust" refers to. Overlap makes every boundary fall *inside*
some chunk, so no adjacent pair of sentences is separated in every chunk.
The cost is duplication: a 120-token window with 30-token overlap (typical
is 10-25%) stores ~25% more text and can return two near-identical chunks,
which is why :mod:`minirag.engine` keeps only the best chunk per document.

**Why sentence-aligned?** Cutting at a fixed token count severs clauses,
which reads badly when the chunk is shown to a user or handed to a
generator. Packing whole sentences up to a budget costs a few lines and
makes every chunk quotable, at the price of oversized chunks whenever a
sentence is longer than the budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split after . ! or ? before whitespace and a likely sentence start. Naive
# ("Dr. Sagan" splits); a real system uses a trained segmenter.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text. Immutable by construction.

    ``chunk_id`` is ``"{doc_id}#{ordinal}"``, unique across the corpus.
    ``sentence_span`` is a half-open ``(start, end)`` pair of indices into
    the parent document's sentence list, so you can widen context at answer
    time without re-chunking.
    """

    doc_id: str
    chunk_id: str
    text: str
    sentence_span: tuple[int, int]


def split_sentences(text: str) -> tuple[str, ...]:
    """Split ``text`` into sentences, dropping empties and stray whitespace."""
    return tuple(s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip())


def chunk_document(
    text: str, doc_id: str, *, max_tokens: int = 120, overlap_tokens: int = 30
) -> tuple[Chunk, ...]:
    """Chunk one document with a sentence-aware sliding window.

    Token counts are whitespace word counts, not tokeniser output: sizing
    need only be roughly right, and it keeps :mod:`minirag.tokenize` out.

    ``max_tokens`` is a soft bound -- a window always holds at least one
    sentence. ``overlap_tokens`` is the trailing context repeated at the
    start of the next chunk; ``0`` disables it. Blank input yields ``()``.

    Raises:
        ValueError: If ``max_tokens < 1``, or ``overlap_tokens`` is negative
            or not smaller than ``max_tokens`` -- equal would never advance
            the window, an infinite loop caught here rather than at 3am.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must satisfy 0 <= overlap < max_tokens")

    sentences = split_sentences(text)
    if not sentences:
        return ()
    lengths = [len(s.split()) for s in sentences]

    chunks: list[Chunk] = []
    start = 0
    while start < len(sentences):
        # Grow the window one sentence at a time until the budget is spent.
        end, budget = start, 0
        while end < len(sentences) and (end == start or budget + lengths[end] <= max_tokens):
            budget += lengths[end]
            end += 1

        chunks.append(
            Chunk(doc_id, f"{doc_id}#{len(chunks)}", " ".join(sentences[start:end]), (start, end))
        )
        if end >= len(sentences):
            break
        start = _overlap_start(lengths, start, end, overlap_tokens)
    return tuple(chunks)


def _overlap_start(lengths: list[int], start: int, end: int, overlap_tokens: int) -> int:
    """Walk back from ``end`` until ``overlap_tokens`` words are covered.

    Guaranteed to return a value ``> start`` so the window always advances,
    even when the last sentence alone exceeds the overlap budget.
    """
    back, carried = end, 0
    while back > start + 1 and carried + lengths[back - 1] <= overlap_tokens:
        back -= 1
        carried += lengths[back]
    return back


def chunk_corpus(
    documents: dict[str, str], *, max_tokens: int = 120, overlap_tokens: int = 30
) -> tuple[Chunk, ...]:
    """Chunk a ``{doc_id: text}`` mapping. The input is never modified."""
    window = {"max_tokens": max_tokens, "overlap_tokens": overlap_tokens}
    return tuple(c for i, t in documents.items() for c in chunk_document(t, i, **window))
