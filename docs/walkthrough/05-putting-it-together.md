# 5. Putting it together: the engine, and where it stops

> Modules: [`src/minirag/engine.py`](../../src/minirag/engine.py) (175 lines),
> [`src/minirag/demo.py`](../../src/minirag/demo.py) (44 lines)

Four stages built the parts: chunks, a lexical index, a vector index, and a way to
combine two rankings. This stage is the assembly — and an honest account of what
the assembled thing cannot do.

## Run it

```bash
python -m minirag.demo
```

Six one-line documents, two questions, three modes each:

```
indexed 6 documents -> 6 chunks

query: 'which moon hides an ocean beneath the ice'
  expect: europa, and all three modes agree
    bm25   1. europa 3.5001   2. pluto 1.0099   3. mars 0.9549
   dense   1. europa 0.3938   2. pluto 0.1167   3. mars 0.1090
  hybrid   1. europa 0.0328   2. pluto 0.0323   3. mars 0.0317

query: 'volcanic eruptions on a moon'
  expect: io missing everywhere -- no stemming, no synonyms
    bm25   1. enceladus 0.7214   2. titan 0.7214   3. europa 0.6428
   dense   1. titan 0.1006   2. enceladus 0.0994   3. europa 0.0850
  hybrid   1. enceladus 0.0325   2. titan 0.0325   3. europa 0.0317
```

The first query is the happy path. The second one is the point of the demo: the
corpus contains *"Io is the most volcanically active body in the solar system"*,
the query asks for `volcanic eruptions`, and **no mode returns it**. `volcanic` and
`volcanically` are different strings, so BM25 has nothing to match; the hashed
embedder is lexical underneath ([stage 3](03-dense-search.md)), so they land in
unrelated columns and dense mode cannot bridge the gap either. Fusion then does
what fusion does — it reconciles two rankings, and neither contains the answer.

Both behaviours are pinned in `tests/test_demo.py`. Add a stemmer or a real
embedder and that second test fails, which is the intent: the failure is a claim
this repo makes in prose, so it is tested like any other claim.

## What `index()` actually does

```
documents --chunk--> chunks --tokenize--> tokens --+--> BM25 index
                                                   +--> dense index
```

Three things worth noticing in `MiniRAG.index()`:

**Tokenise once, index twice.** The token tuples are built one time and handed to
both `bm25.build_index()` and the embedder — if the two retrievers normalised text
differently they would disagree for reasons unrelated to retrieval.

**Duplicate `doc_id` is an error, not a merge.** Two documents under one id makes
every later result ambiguous, and that ambiguity would surface as a confusing hit
rather than an exception. Fail at index time.

**The index is replaced atomically.** Everything search needs lives in one frozen
`_State` object, swapped in at the end, so a failed `index()` leaves the previous
index intact and searchable rather than half-rebuilt.

## What `search()` actually does

Tokenise the query, run one or both retrievers, fuse if asked, collapse. Two
details earn their keep, both covered in [stage 4](04-fusion.md): the engine
**over-fetches** `max(top_k * 4, 20)` candidates per retriever, because fusion
needs depth to reconcile, and it **collapses to one hit per document**, because
overlapping chunks would otherwise fill the top-5 with near-identical text.

One edge case is a design decision rather than an oversight: a query that
tokenises to nothing (`""`, or `"the who"` with stopwords removed) returns `()`.
Empty is a legitimate answer. Callers handle it; the engine does not raise.

## The trade-offs, stated plainly

| Choice | Buys | Costs |
| --- | --- | --- |
| In-memory, rebuilt whole | No storage layer, no staleness | Re-index to change one document |
| Brute-force cosine | Exact results, ~3 lines | O(n) per query; dies in the millions |
| Hashed TF-IDF embedder | Runs offline, no model | Not semantic — no synonym matching |
| Fusion over rankings | No score calibration to tune | Throws away margin information |
| Regex sentence splitting | One line | `Dr. Sagan` is two sentences |
| Under 800 lines | Readable in an afternoon | Every feature below is missing |

## Where a real system diverges

Roughly in the order the additions pay off:

1. **A learned embedder.** The single change that alters what the system can
   answer. Implement `Embedder` ([stage 3](03-dense-search.md)) with
   sentence-transformers or an API call and the `volcanic`/`volcanically` miss
   above goes away — nothing else in the repo changes.
2. **Reranking.** A cross-encoder over the fused top-50, reading query and chunk
   together rather than comparing two independently-built vectors. The highest
   precision-per-line addition to any real pipeline.
3. **Persistence and incremental updates.** A real corpus changes daily: a stored
   index, deletes, a rebuild strategy. This is where most of a production
   retrieval system's actual code ends up living.
4. **An ANN index.** Only past roughly a million vectors, and knowingly: HNSW and
   IVF buy latency by giving back recall.
5. **Metadata filtering.** "Only search documents from this tenant, after this
   date." Cheap to add, and the constraint that most often decides whether a
   result is usable at all.
6. **Measurement.** Recall@k over a labelled query set, run on every change.
   Without it every tuning decision is a guess — `k1`, `b`, `rrf_k`, the fusion
   weights and the chunk size all trade against each other.

Generation is the deliberate omission. minirag hands you ranked chunk text; what
you put in the prompt, and how you cite it, is a different problem.

← [Back to the README](../../README.md) · [Stage 4: fusion](04-fusion.md)
