"""Recall@k and MRR for the three modes: ``python -m extras.eval``.

The README says hybrid "settles" the disagreements between BM25 and dense
search. That is a claim about numbers, so this module produces them: it
indexes the fixture corpus exactly as the README corpus example does, runs every
fixture query in each mode, and scores the rankings against the labelled
answer. It lives outside the 800-line budget, like ``hnsw.py``, because
measuring the engine is a different job from being the engine.

**The metrics.** Each query has a set of relevant documents (the fixture
labels exactly one). *Recall@k* is the fraction of that set found in the top
``k`` hits, averaged over queries -- with one relevant document it is simply
"did it make the top ``k``". *MRR* (mean reciprocal rank) is ``1 / rank`` of
the first relevant hit, averaged, with ``0`` for a miss; it rewards putting
the answer first rather than merely somewhere. Rankings run over the whole
corpus (``top_k`` = number of documents), so MRR is not truncated.

**Read the numbers for what they are.** Five queries over twenty documents
is a smoke test, not a benchmark: one query moves recall by 0.2. The value
is that a change to the tokeniser, the chunker or the fusion constant shows
up here -- and in ``tests/test_eval.py``, which pins the measured table
-- instead of as search that quietly got worse.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from minirag import Document, MiniRAG, SearchMode

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus.json"
"""The 20-document solar-system corpus and its labelled queries."""

MODES: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")
KS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class Query:
    """A query and the doc_ids that count as a correct answer."""

    text: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class ModeResult:
    """One row of the table: a mode's mean recall at each ``k``, and its MRR."""

    mode: SearchMode
    recall: Mapping[int, float]
    mrr: float


def load_fixture(path: Path = FIXTURE) -> tuple[tuple[Document, ...], tuple[Query, ...]]:
    """Parse ``corpus.json`` into documents and labelled queries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    documents = tuple(
        Document(doc_id=d["doc_id"], text=d["text"], metadata={"title": d["title"]})
        for d in data["documents"]
    )
    queries = tuple(
        Query(text=q["query"], relevant=frozenset({q["expected_doc_id"]})) for q in data["queries"]
    )
    return documents, queries


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    """Fraction of ``relevant`` that appears in the first ``k`` of ``ranked``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        raise ValueError("a query needs at least one relevant doc_id")
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    """``1 / rank`` of the first relevant doc_id in ``ranked``; ``0.0`` if absent."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    engine: MiniRAG,
    queries: Sequence[Query],
    mode: SearchMode,
    *,
    depth: int,
    ks: Sequence[int] = KS,
) -> ModeResult:
    """Score ``mode`` over ``queries``, ranking ``depth`` documents per query.

    Raises:
        ValueError: If ``queries`` is empty or any ``k`` exceeds ``depth``.
    """
    if not queries:
        raise ValueError("need at least one query")
    if max(ks) > depth:
        raise ValueError(f"k={max(ks)} exceeds the ranking depth {depth}")
    judged = [
        ([hit.doc_id for hit in engine.search(q.text, top_k=depth, mode=mode)], q.relevant)
        for q in queries
    ]
    recall = {k: sum(recall_at_k(r, rel, k) for r, rel in judged) / len(judged) for k in ks}
    mrr = sum(reciprocal_rank(r, rel) for r, rel in judged) / len(judged)
    return ModeResult(mode=mode, recall=MappingProxyType(recall), mrr=mrr)


def run(path: Path = FIXTURE, ks: Sequence[int] = KS) -> tuple[ModeResult, ...]:
    """Index the fixture as the README corpus example does and score every mode."""
    documents, queries = load_fixture(path)
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)
    engine.index(documents)
    return tuple(evaluate(engine, queries, m, depth=len(documents), ks=ks) for m in MODES)


def format_table(results: Sequence[ModeResult]) -> str:
    """Render results as a Markdown table, four decimals, one row per mode."""
    ks = sorted(results[0].recall)
    header = ["mode", *(f"recall@{k}" for k in ks), "MRR"]
    rows = [[f"`{r.mode}`", *(f"{r.recall[k]:.4f}" for k in ks), f"{r.mrr:.4f}"] for r in results]
    align = ["---", *("---:" for _ in header[1:])]
    return "\n".join("| " + " | ".join(line) + " |" for line in (header, align, *rows))


def main() -> None:
    """Print the table for the fixture corpus."""
    documents, queries = load_fixture()
    print(f"{len(queries)} queries over {len(documents)} documents (fixture corpus)\n")
    print(format_table(run()))


if __name__ == "__main__":
    main()
