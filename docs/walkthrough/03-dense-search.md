# 3. Dense search: vectors without a model

> Module: [`src/minirag/vectors.py`](../../src/minirag/vectors.py) — 146 lines

Dense retrieval is usually presented as one thing. It is two, and separating them
is the most useful idea in this stage:

```
text   -> vector       the embedder   (a model, in the real world)
vector -> neighbours   the index      (an ANN structure, in the real world)
```

minirag implements both in numpy, so the repo runs with no download and no network
call, and then marks precisely where you would swap in something better.

## An embedder built with arithmetic

`HashedTfidfEmbedder` constructs a vector space instead of learning one. Three
steps:

**1. Hash every term to a column.** `blake2b(term) % dim`. Any term maps into a
fixed `dim` columns, so the vocabulary can grow forever and the matrix never
resizes. That is the *hashing trick*, and it is the reason there is no vocabulary
file anywhere in this repo.

The choice of hash is load-bearing. Python's built-in `hash()` is salted per
process — the same text would embed differently after a restart, silently
invalidating any persisted index. `blake2b` is stable across processes and
machines, and there is a test that says so.

**2. Weight each column by `tf * idf`.** Same intuition as [stage
2](02-lexical-search.md): rare terms carry meaning. Term frequency is sublinear
(`1 + log f`) for the same reason BM25 has `k1` — the tenth occurrence of a word
says far less than the second.

**3. L2-normalise the rows.** Once every vector has unit length, cosine similarity
*is* the dot product, and the entire search becomes one matrix multiply.

```python
embedder = HashedTfidfEmbedder.fit(corpus_tokens)   # learns per-column IDF
matrix = embedder.encode(corpus_tokens)             # (n_chunks, 4096), rows unit-norm
```

## What this is honestly not

It is a real dense vector space — fixed width, real-valued, cosine comparable, and
near-duplicate texts genuinely land near each other. But it is **lexical
underneath**: `car` and `automobile` hash to unrelated columns and have cosine 0.
It cannot do synonym matching, which is the entire selling point of a learned
embedding model.

There is a second, sharper consequence of the hashing trick. Unknown terms still
land in *some* column, so they can collide with real ones:

```python
>>> q = "quantum chromodynamics lagrangian"
>>> engine.search(q, mode="bm25")
()
>>> [(h.doc_id, round(h.score, 3)) for h in engine.search(q, mode="dense")]
[('earth', 0.092), ('jupiter', 0.084), ('solar_wind', 0.071)]
```

BM25 correctly says "nothing here". The dense index returns three documents about
nothing of the sort, at low but nonzero scores. This is pinned by a test rather
than hidden, because it is the honest character of the method: **no lexical match
does not imply no dense match.** Real embedding models have a stronger version of
the same failure — they always return *something*, confidently.

## The seam

`Embedder` is a `Protocol` with one method:

```python
class Embedder(Protocol):
    dim: int
    def encode(self, documents: Sequence[Sequence[str]]) -> np.ndarray: ...
```

`DenseIndex` and `MiniRAG` never name `HashedTfidfEmbedder`. Anything returning an
`(n, dim)` array of L2-normalised rows drops in — sentence-transformers, an
embeddings API call, a local ONNX model. The test suite exercises the seam with a
five-line stand-in to prove the boundary is real and not aspirational.

That is the intended upgrade path: keep the chunker, the BM25 index, the fusion
and the engine exactly as they are, and replace one class.

## Brute force, on purpose

```python
sims = self.matrix @ query_vector.reshape(-1)
candidates = np.argpartition(-sims, k - 1)[:k]
```

Two lines, exact results. It compares against every vector, which sounds
embarrassing next to HNSW or IVF until you notice that at a few thousand chunks
brute force is *faster* once you count index build time — and that approximate
structures buy their speed by trading away recall. Start exact. Add ANN when a
profiler, not a blog post, tells you to.

`argpartition` finds the top `k` in O(n) without sorting the rest; only the `k`
survivors get sorted. Rows with similarity 0 are dropped for the same reason BM25
drops zero scores.

**Next:** [4. Fusion](04-fusion.md) — we now have two ranked lists that disagree.
