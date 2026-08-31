# 1. Chunking: choosing the unit of retrieval

> Module: [`src/minirag/chunker.py`](../../src/minirag/chunker.py) — 130 lines

Before you can search anything, you have to decide *what a result is*. This is the
first design decision in a RAG system and the one people spend the least time on,
which is odd, because everything downstream inherits it.

## The unit problem

Retrieval returns whole units. Pick the wrong size and no amount of clever scoring
saves you.

**Whole documents are too big.** Say a 40-page manual mentions "certificate
rotation" in one paragraph. BM25 divides a document's score by its relative length
(see [stage 2](02-lexical-search.md)), so that single match is drowned out by
thirty-nine pages of unrelated text — and even if it ranks first, you hand the
reader forty pages and hope.

**Single sentences are too small.** "It must be rotated every 90 days." Rotated
*what*? A sentence-level index retrieves individually meaningless fragments, and
with fewer terms per unit the scores get noisy.

**Chunks are the compromise:** long enough to carry their own context, short
enough that matching means something. minirag defaults to a 120-word budget.

## Why overlap is not optional

Here is the failure that overlap exists to prevent. Take this text:

```
... Europa is covered in water ice. | Beneath that crust lies a salty
ocean twice the volume of Earth's. ...
```

Chunk on a hard boundary at `|` and you get two chunks, neither of which contains
"Europa" *and* "ocean". A user searching `Europa subsurface ocean` now matches
each half weakly instead of one chunk strongly.

Dense retrieval does worse, not better. The second chunk's embedding has to
represent "Beneath that crust lies a salty ocean..." with no idea whose crust —
the referent was in the chunk before it. Pronouns and demonstratives are exactly
what a boundary tends to sever.

Overlap fixes this by making every boundary appear *inside* some chunk: chunk N+1
starts a few sentences before chunk N ended, so no adjacent pair of sentences is
ever separated in every chunk. The tests assert exactly that:

```python
def test_every_adjacent_sentence_pair_shares_a_chunk() -> None:
    chunks = chunk_document(PARAGRAPH, "doc", max_tokens=21, overlap_tokens=7)
    covered = {(i, i + 1) for c in chunks
               for i in range(c.sentence_span[0], c.sentence_span[1] - 1)}
    assert covered == {(i, i + 1) for i in range(n_sentences - 1)}
```

The cost is duplication. A 120-token window with 30-token overlap stores about 25%
more text, and a query can retrieve two near-identical chunks from the same
document. minirag handles that at the other end of the pipeline: the engine
collapses results to the best-scoring chunk per document. Typical overlap is
10–25% of the window; past ~40% you are mostly paying to index the same words.

## Why sentences, not tokens

You could slice at exactly 120 tokens — simpler, and perfectly even. It also cuts
through the middle of clauses, so a chunk reads like "...the atmospheric pressure
at the surface is about ninety times", which is bad shown to a user and worse
pasted into a prompt. Packing whole sentences costs about ten lines and makes
every chunk quotable:

```python
end, budget = start, 0
while end < len(sentences) and (end == start or budget + lengths[end] <= max_tokens):
    budget += lengths[end]
    end += 1
```

The `end == start` clause is the interesting part: a window always takes at least
one sentence, even one longer than the whole budget. That means an oversized
sentence becomes an oversized chunk rather than being hard-split — we chose the
occasional big chunk over the mid-clause cut we were trying to avoid.

## What you get

`Chunk` is a frozen dataclass, so nothing downstream can quietly rewrite an index:

```python
Chunk(doc_id="europa", chunk_id="europa#1", text="Beneath a crust of water ice...",
      sentence_span=(2, 4))
```

`sentence_span` is a half-open range into the parent document's sentence list.
Search does not use it; it is what lets you widen context at answer time — "this
chunk plus the sentence before it" — without re-chunking.

## Honest limits

Sentence splitting is one regex. `Dr. Sagan` becomes two sentences. Token counting
is whitespace words, not the tokeniser's output, because chunk sizing only needs to
be approximately right and this keeps the module independent of `tokenize.py`.
Both are deliberate: a trained segmenter is a dependency, and neither would teach
you anything about retrieval.

**Next:** [2. Lexical search](02-lexical-search.md) — now that we have units, how
do we score them?
