# minirag

[![CI](https://github.com/Bruhadev45/minirag/actions/workflows/ci.yml/badge.svg)](https://github.com/Bruhadev45/minirag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

**A complete RAG engine in <800 lines. No frameworks. Read it in an afternoon.**

No LangChain. No LlamaIndex. No vector database. No model download. No network
calls. Just Python's standard library and numpy, implementing the retrieval half
of RAG end to end: sentence-aware chunking, BM25 from an inverted index, dense
vectors, brute-force cosine search, and reciprocal rank fusion to combine them.

Every module is meant to be read top to bottom. The docstrings explain *why*, not
just what — the BM25 module walks the formula term by term, the chunker argues for
overlap, the vector module marks the exact seam where you would plug in a real
embedding model.

```python
from minirag import Document, MiniRAG

engine = MiniRAG()
engine.index([
    Document(doc_id="europa", text="Europa is a moon of Jupiter. Beneath its "
                                   "icy crust lies a salty subsurface ocean."),
    Document(doc_id="io",     text="Io is the most volcanically active body in "
                                   "the solar system, heated by tidal forces."),
])

for hit in engine.search("which moon has an ocean under the ice?", mode="hybrid"):
    print(f"{hit.score:.4f}  {hit.doc_id}  {hit.text[:60]}...")
```

## Line count

| Module | Lines | What it does |
| --- | ---: | --- |
| `src/minirag/tokenize.py` | 64 | Lowercase, strip punctuation, optional stopword removal |
| `src/minirag/chunker.py` | 125 | Sentence-aware sliding window with overlap |
| `src/minirag/bm25.py` | 147 | Inverted index, IDF, TF saturation (`k1`), length norm (`b`) |
| `src/minirag/vectors.py` | 145 | Hashed TF-IDF embedder + brute-force cosine index |
| `src/minirag/fusion.py` | 65 | Weighted reciprocal rank fusion |
| `src/minirag/engine.py` | 183 | `index()` / `search()`, three modes, metadata filter |
| `src/minirag/demo.py` | 44 | `python -m minirag.demo`, the runnable tour |
| `src/minirag/__init__.py` | 26 | Public surface |
| **Total** | **799** | Budget **800**, enforced by `tests/test_line_budget.py` |

Physical lines, docstrings and blanks included — the lines you actually scroll
past. 192 tests cover it at a 95% coverage floor (measured 99%), `ruff check`
is clean, and CI runs all of it on Python 3.10–3.13 plus macOS. The budget test
asserts `< 800`, so there is no headroom left: metadata filtering only fit after
trimming prose elsewhere, which is the trade the budget exists to force.

## Quickstart

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # the full suite, coverage floor, and the line budget
.venv/bin/ruff check src/ tests/ extras/
.venv/bin/python -m minirag.demo   # six documents, two queries, three modes
```

The demo is the 30-second version: it indexes six one-line documents and prints
each mode's top three for two questions — one every mode answers, one every mode
gets wrong. [Stage 5](docs/walkthrough/05-putting-it-together.md) unpacks why.

Then run it against the test corpus — 20 short documents about the solar system,
shipped in `tests/fixtures/corpus.json`:

```python
import json
from minirag import Document, MiniRAG

data = json.load(open("tests/fixtures/corpus.json"))
engine = MiniRAG(max_tokens=60, overlap_tokens=20)
engine.index([Document(d["doc_id"], d["text"], {"title": d["title"]}) for d in data["documents"]])

for query in [q["query"] for q in data["queries"]]:
    top = engine.search(query, top_k=1)[0]
    print(f"{query!r}\n  -> {top.doc_id} ({top.score:.4f})\n")
```

Try the same queries with `mode="bm25"` and `mode="dense"` to watch the two
retrievers disagree, then `mode="hybrid"` to watch fusion settle it.

## How it fits together

```
documents --chunk--> chunks --tokenize--> tokens --+--> BM25 index
                                                   +--> dense index

query --tokenize--> +--> BM25 search --+
                    +--> dense search -+--> RRF --> hits
```

Three search modes:

- **`bm25`** — exact term matching. Unbeatable on names, codes and rare words;
  blind to paraphrase.
- **`dense`** — cosine similarity in vector space. Robust to wording (with a real
  model); occasionally confidently wrong.
- **`hybrid`** — fuses the two *rankings*, not their scores. Ranks are comparable;
  an unbounded BM25 score and a cosine in `[0, 1]` are not.

Every mode takes `where={"field": value}` to search only documents whose
metadata matches every pair. It filters *before* ranking, so a matching document
is never crowded out by higher-ranked ones that do not match:

```python
engine.search("ocean", where={"title": "Europa"})
```

## The walkthrough

Five staged documents that build the system up, in `docs/walkthrough/`:

1. [Chunking](docs/walkthrough/01-chunking.md) — why documents are the wrong unit,
   and why overlap is not optional.
2. [Lexical search](docs/walkthrough/02-lexical-search.md) — the inverted index and
   the three moving parts of BM25.
3. [Dense search](docs/walkthrough/03-dense-search.md) — building a vector space
   with arithmetic, and where a real model plugs in.
4. [Fusion](docs/walkthrough/04-fusion.md) — why you cannot add the scores, and
   what to do instead.
5. [Putting it together](docs/walkthrough/05-putting-it-together.md) — the
   engine, the trade-offs it made, and where a real system diverges.

## Extras

Code that is worth reading but does not belong inside the budget lives in
`extras/` — beyond the 800 lines, same spirit: no new required dependencies,
every parameter explained, every claim measured or labelled as unmeasured.

- **[`extras/hnsw.py`](extras/hnsw.py)** — a minimal HNSW index, the approximate
  neighbour search this repo argues you do not need yet. It is here so the
  argument stays checkable: same input matrix and same return shape as
  `DenseIndex`, so `tests/test_hnsw.py` can put the two side by side and report
  what the graph loses.

Recall@10 against exact brute-force search, measured by that test file
(`m` edges per node, `ef` beam width):

| Vectors | `m` | `ef` | recall@10 |
| --- | ---: | ---: | ---: |
| 40 fixture chunks, 25 queries | 8 | 16 | 1.0000 |
| 40 fixture chunks, 25 queries | 8 | 32 | 1.0000 |
| 800 random unit vectors (64-d), 60 queries | 8 | 16 | 0.6700 |
| 800 random unit vectors (64-d), 60 queries | 8 | 32 | 0.8250 |
| 800 random unit vectors (64-d), 60 queries | 8 | 64 | 0.9467 |
| 800 random unit vectors (64-d), 60 queries | 16 | 64 | 0.9983 |

Read that top to bottom: on the fixture corpus the graph loses nothing, because
40 chunks is nowhere near where ANN pays — which is exactly why the shipped
engine does not use it. On unclustered Gaussian vectors, the hardest case for a
navigable graph, the default settings silently drop a third of the true
neighbours, and you buy them back with `ef` at query time or `m` at build time.
That silence is the point: an ANN index never reports the neighbours it missed.

```bash
.venv/bin/pytest tests/test_hnsw.py -q
```

- **[`extras/eval.py`](extras/eval.py)** — recall@k and MRR for `bm25`, `dense`
  and `hybrid` over the fixture's labelled queries, indexed exactly as the
  quickstart above does. `tests/test_eval.py` checks the metric math on
  hand-worked rankings and pins the measured table.

### Measured on the toy corpus

Output of `python -m extras.eval`, pasted unedited:

```
5 queries over 20 documents (fixture corpus)

| mode | recall@1 | recall@3 | recall@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `bm25` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `dense` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `hybrid` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
```

Every mode is at the ceiling, and that is the honest result: each of the five
fixture queries shares distinctive words with its answer ("methane",
"greenhouse", "red storm"), so exact matching alone ranks it first, and the
hashed embedder — TF-IDF in disguise — agrees. This set cannot tell the modes
apart, so it shows no gain from fusion; separating them takes paraphrased
queries, which need a real embedding model to answer. What the table does do
is act as a tripwire: a tokeniser, chunker or fusion change that loses one of
these answers drops a cell to 0.8 and fails the pinned test.

```bash
.venv/bin/python -m extras.eval
```

- **[`extras/rerank.py`](extras/rerank.py)** — a second pass over the first
  stage's shortlist. `apply(engine, query, reranker)` runs `engine.search` for
  the top `candidates` (20 by default), scores each hit's chunk text and
  re-orders them. It wraps the engine instead of adding `search(..., rerank=...)`
  because the core has no line left for the hook. It only re-orders what
  `search` returned, so a `where` filter still decides which documents can come
  back. The reranker is still trusted code that reads every candidate's text,
  and it sees only each document's best first-stage chunk. Two rerankers share
  the one-method `Reranker` protocol:
  - `BM25Reranker` — no dependency. BM25 over the shortlist alone, so a word
    every candidate shares counts for little and a word that separates them
    counts for a lot. Useful after `hybrid`, whose RRF kept only ranks.
  - `CrossEncoderReranker("<model name>")` — wraps a sentence-transformers
    `CrossEncoder`, imported only when you build one
    (`pip install sentence-transformers`; it pulls in torch, and the model
    downloads on first use). Loading a model can run code shipped with it, so
    pick the model yourself, never from a request; the adapter never sets
    `trust_remote_code`. The tests drive it with a fake model, so nothing in
    this repo measures a real cross-encoder yet.

With `engine` indexed over the fixture corpus as in the quickstart, from the
repo root:

```python
from extras.rerank import BM25Reranker, apply

query = "which moon has an ocean under the ice"
print("hybrid:  ", [hit.doc_id for hit in engine.search(query, top_k=3)])
for r in apply(engine, query, BM25Reranker(), top_k=3):
    print(f"{r.score:.4f}  {r.doc_id:<7} was #{r.first_stage_rank}")
```

```
hybrid:   ['moon', 'pluto', 'europa']
1.7158  moon    was #1
1.5657  europa  was #3
1.3522  pluto   was #2
```

Europa's chunk matches "ocean" and "ice" and now beats Pluto's, which matches
only "ice". It still trails the Moon's ("moon", "ocean") because Europa's text
says "moons", and with no stemmer that never matches "moon". A lexical second
opinion rewards shared words and nothing else. On the five labelled fixture
queries it keeps every answer first, which `tests/test_rerank.py` pins, so it
loses none of them; at the ceiling already, it cannot show a gain either.

```bash
.venv/bin/pytest tests/test_rerank.py -q
```

## Honest limitations

This is a teaching repository. It is correct, tested, and deliberately incomplete.

- **The embeddings are not semantic.** `HashedTfidfEmbedder` builds a real dense
  vector space arithmetically, so the whole dense path is exercised and testable —
  but "car" and "automobile" hash to unrelated columns. It cannot match synonyms,
  which is the entire selling point of a learned embedding model. Swap one in at
  the `Embedder` protocol; nothing else changes.
- **Hash collisions make dense mode over-eager.** Every term lands in *some*
  column, so an out-of-vocabulary query that BM25 correctly answers with nothing
  still scores ~0.09 against a few unrelated documents. Pinned by a test rather
  than hidden — see [stage 3](docs/walkthrough/03-dense-search.md).
- **There is no generation.** minirag is the *retrieval* in RAG. It hands you the
  chunk text; prompting a model with it is your job.
- **Everything is in memory, and rebuilt from scratch.** No persistence, no
  incremental updates. Re-index to change anything. Expect it to be comfortable up
  to roughly 10<sup>4</sup>–10<sup>5</sup> chunks on a laptop, then not.
- **Search is exact, not approximate.** Brute-force cosine is O(n) per query, which
  beats ANN structures at this scale and loses badly beyond it. The measured
  version of that claim is [`extras/hnsw.py`](extras/hnsw.py) and the recall
  table above.
- **The tokeniser is naive.** No stemming, no lemmatisation, no subwords. `C++`
  becomes `c`, and a query for `volcanic` misses a document saying `volcanically`
  — `python -m minirag.demo` shows exactly that. Non-English text will tokenise
  but the stopword list won't apply.
- **Sentence splitting is a regex.** `Dr. Sagan` splits into two sentences. A
  trained segmenter would not.
- **Reranking is lexical unless you bring a model.** `extras/rerank.py` ships the
  hook, a BM25 reranker and a cross-encoder adapter, but the adapter has only run
  against a fake model in the tests. No cross-encoder is downloaded, run or
  measured in this repo.
- **Single-process, no concurrency story.** The index is immutable once built, so
  reads are safe; writes are a full replacement.

## Licence

MIT © 2026 Kandimalla Bruhadev
