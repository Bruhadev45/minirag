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

**What it does, and what it does not.** You give minirag documents; it gives
you back ranked chunks of their text with scores. It never calls a language
model and never writes an answer — passing those chunks to a model is your job.
The `dense` mode runs on a hashed TF-IDF embedder: arithmetic over word counts,
no model and no download. It matches shared words, not meanings ("car" never
finds "automobile"). A learned embedding model plugs in at the `Embedder`
protocol; this repo does not ship or measure one.

## First result in a minute

You need Python 3.10 or newer (CI tests 3.10–3.13) and `git`. minirag is not
published on PyPI (checked 2026-10-29), so install it from GitHub. If a package
named `minirag` appears on PyPI, it is not this project unless this README says
so. In a new, empty directory (macOS/Linux shown; on Windows the venv's
programs are in `.venv\Scripts\`):

```bash
python3 --version   # must print 3.10 or newer
python3 -m venv .venv
.venv/bin/pip install "git+https://github.com/Bruhadev45/minirag.git"
.venv/bin/python -m minirag.demo
```

If `python3` is older, name a newer interpreter instead, e.g.
`python3.12 -m venv .venv`. The `/usr/bin/python3` that Apple's developer tools
install can be too old: it was 3.9.6 on the Mac where these steps were checked.
The only tagged release, v0.1.0, predates both the demo and the `where=`
filter, so install from `main`; `main` still reports version 0.1.0 because the
version number has not been bumped since. To pin a revision, append
`@<commit>` to the URL. The demo indexes six one-line documents and asks two
questions in all three modes:

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

Scores compare only within a mode; the ranking is the point.
```

The first question works in every mode. The second fails in every mode, and
that is deliberate: "volcanic" never matches Io's "volcanically" without a
stemmer, and the hashed embedder does not bridge that gap here either.
[Stage 5](docs/walkthrough/05-putting-it-together.md) unpacks why. Each mode
scores on its own scale: BM25 is unbounded, `dense` is a cosine, and `hybrid`
is a reciprocal rank fusion score, where 0.0328 (1/61 + 1/61) means first in
both lists.

### Use it from Python

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

```
0.0328  europa  Europa is a moon of Jupiter. Beneath its icy crust lies a sa...
```

One hit, not two. Once stopwords are dropped the question is `moon ocean ice`;
Io's text contains none of those words, so it scores zero in both retrievers,
and a zero-score document is left out of the results.

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
past. The test suite covers the core and the extras, with a 95% coverage floor
on the core (measured 99%), and `tests/test_readme.py` re-runs this README's
demo, Python and corpus examples from a clone and compares them with their
pasted output (the Git install itself is not run in the tests). `ruff check` is
clean, and CI runs all of it on Python 3.10–3.13 plus macOS. The budget test
asserts `< 800`, so there is no headroom left: metadata filtering only fit
after trimming prose elsewhere, which is the trade the budget exists to force.

## Develop: clone, test, and run the extras

The installed package holds only `src/minirag`. The tests, the fixture corpus
and `extras/` live in the repository, so everything from here on runs from the
root of a clone:

```bash
git clone https://github.com/Bruhadev45/minirag.git
cd minirag
python3.12 -m venv .venv          # any Python 3.10 or newer
.venv/bin/pip install -e ".[dev]"  # editable install plus pytest, pytest-cov and ruff
.venv/bin/pytest                   # the full suite, coverage floor, and the line budget
.venv/bin/ruff check src/ tests/ extras/
```

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

```
'which planet has a giant red storm bigger than Earth'
  -> jupiter (0.0328)

'moon with lakes of liquid methane and a thick nitrogen atmosphere'
  -> titan (0.0328)

'runaway greenhouse effect makes this the hottest planet'
  -> venus (0.0328)

'subsurface saltwater ocean under an icy crust'
  -> europa (0.0328)

'spectacular ring system made of water ice particles'
  -> saturn (0.0328)
```

All five top hits are the labelled answers, and each is first in both lists.
These queries share distinctive words with their answers, so they are easy;
[`extras/eval.py`](extras/eval.py) below says the same in numbers.

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
  corpus example above does. `tests/test_eval.py` checks the metric math on
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

With `engine` indexed over the fixture corpus as in the corpus example, from the
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

- **[`extras/persist.py`](extras/persist.py)** — `save(engine, "index.npz")` and
  `load("index.npz")`, so you index once and search later. Numpy only, no
  pickle. The `.npz` holds the dense matrix and the embedder's IDF weights as
  raw float64, which come back bit for bit. A JSON sidecar beside it holds the
  settings, metadata and chunk text. BM25 is rebuilt from that text at load,
  one tokenise pass. `tests/test_persist.py` checks that every mode, `top_k`
  and `where` filter returns identical hits after a round trip.
  - Loading fails with `PersistError` if the format version or the minirag
    version differs, because a changed tokeniser would quietly break saved
    vectors. The sidecar also stores a fingerprint of what the tokeniser and
    term hash do to one fixed probe sentence. That catches an edit made without
    a version bump, as long as the probe exercises it.
  - Loading also fails if the two files come from different saves: the sidecar
    records the `.npz`'s SHA-256. Each file is written to a temporary file and
    renamed into place. A crash between the two renames is caught at load, but
    leaves neither the old index nor the new one loadable, so save to a new
    path if the old one must survive.
  - Metadata must be `str` to `str` to be saved; anything else is refused
    before a byte is written.
  - Every field is checked on load and compressed archives are refused, so a
    corrupted file gives a clear error. That is not authentication: whoever can
    write the sidecar can recompute the hash, and a crafted index decides which
    chunk text ranks first and reaches your prompt. Load only files you wrote.
  - The matrix is stored dense and uncompressed: about 32 KB per chunk at the
    default `dim=4096`, most of it zeros.

With `engine` indexed over the fixture corpus as in the corpus example, from the
repo root:

```python
from extras.persist import load, save

npz_path, sidecar = save(engine, "fixture-index.npz")
print(f"{npz_path} {npz_path.stat().st_size:,} bytes, {sidecar} {sidecar.stat().st_size:,} bytes")

restored = load(npz_path)
query = "which moon has an ocean under the ice"
print([hit.doc_id for hit in restored.search(query, top_k=3)])
print(restored.search(query, top_k=20) == engine.search(query, top_k=20))
```

```
fixture-index.npz 1,343,992 bytes, fixture-index.json 12,924 bytes
['moon', 'pluto', 'europa']
True
```

```bash
.venv/bin/pytest tests/test_persist.py -q
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
- **Everything is in memory, and rebuilt from scratch.** `extras/persist.py` saves
  and loads a whole index, but there are no incremental updates: re-index and save
  again to change anything. Expect it to be comfortable up to roughly
  10<sup>4</sup>–10<sup>5</sup> chunks on a laptop, then not.
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
