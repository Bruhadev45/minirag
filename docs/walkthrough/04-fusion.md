# 4. Fusion: combining rankings you cannot compare

> Modules: [`src/minirag/fusion.py`](../../src/minirag/fusion.py) (65 lines),
> [`src/minirag/engine.py`](../../src/minirag/engine.py) (183 lines)

Two retrievers, two ranked lists, one answer required. This stage is short, and
the interesting part is why the obvious approaches fail.

## Why you cannot just add the scores

BM25 returns unbounded scores — 0 to roughly 30, depending on corpus statistics.
Cosine returns `[-1, 1]`. Add them and the lexical list wins by choice of units
alone. Multiply one by a constant and you have invented a hyperparameter that
needs retuning every time the corpus changes.

Normalising per query is worse, and subtly so. Min-max rescaling maps the best
result in each list to 1.0 — including when every result was terrible. "Everything
matched badly" becomes "the least bad result is perfect", and the retriever that
found nothing useful gets an equal vote.

## Reciprocal rank fusion

RRF sidesteps the problem by throwing the scores away and keeping only the ranks,
which are comparable by construction:

```
RRF(d) = SUM  w_r / (k + rank_r(d))
          r
```

`rank` is 1-based. `k` defaults to 60, the value from Cormack et al. (2009). It
flattens the curve near the top: at `k=60`, rank 1 scores `1/61` and rank 2 scores
`1/62` — a gap of about 1.6%. Being first is barely better than being second.

That flatness is the whole design. It makes **agreement across retrievers worth
more than confidence within one**:

```python
fused = reciprocal_rank_fusion(
    {"a": ["lone", "p", "consensus"],
     "b": ["q",    "r", "consensus"]}, k=60)
assert fused[0][0] == "consensus"       # ranked 3rd twice beats ranked 1st once
```

Lower `k` inverts this. At `k=1`, rank 1 scores `1/2` and rank 3 scores `1/4`, so a
single strong opinion wins — and the test suite asserts both behaviours, because
`k` is the one knob here that actually changes what the system believes.

## A real disagreement

From the test corpus, query `water on other worlds`:

| | rank 1 | rank 2 | rank 3 |
| --- | --- | --- | --- |
| **bm25** | mars | pluto | uranus |
| **dense** | uranus | mars | titan |
| **hybrid** | **mars** | uranus | pluto |

Neither retriever's first choice wins automatically. Mars scores
`1/61 + 1/62 = 0.0325`; Uranus scores `1/63 + 1/61 = 0.0323`. Mars takes it by
consensus — first for one retriever, second for the other — against Uranus's first
place plus a weak third. That margin is deliberately small; RRF is a tie-breaker,
not a re-scorer.

Weights let you lean: `MiniRAG(weights={"bm25": 2.0, "dense": 1.0})`. Raise the
lexical weight when your corpus is full of identifiers, part numbers or names;
raise the dense weight when your users paraphrase. A weight of `0.0` silences a
retriever entirely, which is a useful way to A/B one against the pair.

## Putting it together

`MiniRAG.search()` is the whole pipeline in a dozen lines: tokenise the query, run
one or both retrievers, fuse, collapse. Two details in there earn their keep.

**Over-fetching.** Fusion needs depth to work with — if each list is 5 long, there
is nothing to reconcile — so the engine retrieves `max(top_k * 4, 20)` candidates
per retriever before fusing and trimming.

**Collapsing to one hit per document.** Retrieval runs over chunks, but chunks
overlap by design ([stage 1](01-chunking.md)). Without this step, a long document
can fill your entire top-5 with near-identical text and crowd out every other
result. The engine keeps the best-scoring chunk per document and drops the rest.

One caveat on what comes back: `Hit.score` means something different in each mode
— unbounded for `bm25`, a cosine for `dense`, a number near `1/k` for `hybrid`. It
orders results within one result set and is meaningless across them. The docstring
says so, because someone will otherwise threshold on it.

## What is deliberately missing

**Reranking.** The highest-value addition to any real pipeline is a cross-encoder
over the fused top-50 — a model that reads query and chunk *together* rather than
comparing two independently-built vectors. It would roughly double this repo's
size and require a model download, so it is not here. If you build on minirag, it
is the first thing to add.

← [Back to the README](../../README.md)
