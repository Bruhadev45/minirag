# 2. Lexical search: the inverted index and BM25

> Modules: [`src/minirag/tokenize.py`](../../src/minirag/tokenize.py) (66 lines),
> [`src/minirag/bm25.py`](../../src/minirag/bm25.py) (147 lines)

We have chunks. Now we need to score them against a query. Fifty years on, the
answer is still BM25 — and unlike most of modern retrieval, you can hold all of it
in your head.

## One tokeniser, used by both sides

Before scoring, text becomes terms. The rule that matters more than any choice
inside `tokenize()`: **the indexer and the query must use the same function.**
Index `Earth's` as `earth` but query it as `earth's` and you match nothing, with no
error to tell you why. minirag has exactly one tokeniser and both sides call it.

```python
>>> tokenize("Beneath Europa's crust lies a salty ocean.")
('beneath', 'europa', 'crust', 'lies', 'salty', 'ocean')
```

Note what did *not* happen: `lies` was not stemmed to `lie`, and the duplicates in
`("ice", "ice", "ice")` are kept. Term frequency is the entire basis of the scoring
below, so a tokeniser that deduplicates would silently destroy it.

## The inverted index

The naive way to score is to walk every chunk. The inverted index flips the map
around — instead of `document -> terms`, store `term -> [(document, count), ...]`:

```python
>>> build_index([("ice", "water"), ("ice", "ice", "rock")]).postings
{'ice': ((0, 1), (1, 2)), 'water': ((0, 1),), 'rock': ((1, 1),)}
```

Now a three-term query touches three postings lists instead of the whole corpus,
and documents sharing no term with the query are never visited at all. In our
20-document corpus that is a 556-entry map; the principle is identical at web
scale, which is why this one data structure carried search for fifty years.

## BM25, three parts

```
                              f(q, D) * (k1 + 1)
score = SUM  IDF(q) * --------------------------------------
        q in Q         f(q, D) + k1 * (1 - b + b * |D|/avgdl)
```

**IDF — how surprising is this term?** In our corpus `planet` appears in 11 of 20
documents and scores 0.60; `europa` appears in one and scores 2.64. Rare terms
carry the meaning, so they dominate the sum. This is why you rarely need to strip
stopwords for correctness — IDF already makes them nearly free.

One wrinkle worth knowing: the textbook IDF is `ln((N - n + 0.5) / (n + 0.5))`,
which goes **negative** for a term in more than half the corpus. A document could
then improve its rank by deleting a word. minirag wraps it in `ln(1 + ...)`, which
is the standard fix, and there is a test pinning it:

```python
def test_idf_is_never_negative_even_for_very_common_terms() -> None:
    index = build_index([("a", "x"), ("a", "y"), ("a", "z")])
    assert idf(index, "a") >= 0.0
```

**Saturation (`k1`) — how much does repetition help?** A document that says "Mars"
fifty times is not fifty times more about Mars. The fraction is a hyperbola in
`f`: steep at first, then flattening toward a ceiling of `k1 + 1`. `k1` sets where
the knee is — `k1=0` makes term frequency binary, large `k1` makes it linear.
Going from 1 to 2 occurrences must help more than going from 8 to 16:

```python
scores = score_query(build_index(corpus), ["ice"])   # n = 1, 2, 8, 16
assert scores[1] - scores[0] > scores[3] - scores[2] > 0.0
```

**Length normalisation (`b`) — is this document just long?** `|D|/avgdl` is the
document's length relative to the corpus average (44.8 tokens here). A long
document has more chances to contain any term, so it gets divided down. `b=0`
ignores length entirely, `b=1` normalises fully, `0.75` is the long-standing
default. Note where it sits: in the *denominator*, next to `k1`. A long document
does not merely lose score — it also saturates more slowly.

## Zero means zero

`search()` drops documents scoring 0.0 rather than returning them at the bottom:

```python
>>> [i for i, _ in search(build_index(CORPUS), ["rock"])]
[1]
```

In a lexical system, 0 is not a weak match. It means the document shares no term
with the query at all — an entirely different statement from "scored badly", and
one that matters in [stage 4](04-fusion.md), where a retriever's *absence* from a
list is meaningful.

This is also lexical search's whole weakness in one line. Ask for `automobile` in
a corpus that only ever says `car` and BM25 returns nothing. That is the problem
dense retrieval is supposed to solve.

**Next:** [3. Dense search](03-dense-search.md).
