"""Why hybrid ranked what it ranked: ``python -m extras.explain``.

``hybrid`` search prints one number per hit, an RRF score, and hides where
it came from. This module shows the working: each retriever's own ranking,
and what each one added to every hybrid hit's score. Run with no arguments
it explains two fixture queries on which BM25 and dense search disagree;
pass your own queries to explain those instead.

**It recomputes nothing.** The two lists come from the engine's own ranking
step, at the depth ``search`` over-fetches to and under the same ``where``
filter, and the hybrid hits come from the engine's own fusion and collapse.
So :attr:`Explanation.hits` holds exactly the hits that
``engine.search(query, mode="hybrid")`` returns, in the same order with the
same scores; ``tests/test_explain.py`` checks that on every fixture query.
Like ``persist.py`` it reads private engine state, because the core has no
line left for a public hook.

**Ranks are chunk ranks.** RRF fuses chunks; ``search`` then keeps each
document's best chunk. A retriever adds ``weight / (k + rank)``, where
``rank`` is the chunk's 1-based place in that retriever's list, or adds 0
when the chunk is not in its list: BM25 drops a chunk that shares no word
with the query, dense drops a cosine of 0 or less, and both stop at the
over-fetch depth. The contributions sum exactly to the hybrid score.

**What the disagreement is.** Both retrievers here match words. BM25 and the
hashed TF-IDF embedder see the same tokens but weigh them differently: BM25
saturates repeated terms and only partly normalises for length, while a
cosine divides by the whole vector's length, which can favour short chunks.
This is not lexical-versus-semantic disagreement; that needs a learned
embedding model. :data:`QUERIES` were chosen because the two retrievers
disagree on them. They show the mechanism and measure nothing.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from extras.eval import load_fixture
from minirag import Hit, MiniRAG, tokenize

RETRIEVERS: tuple[str, ...] = ("bm25", "dense")
"""The engine's two retrievers, in the order it fuses them."""

QUERIES: tuple[str, ...] = (
    "dwarf planet beyond neptune",
    "which moon has an ocean under the ice",
)
"""Fixture queries on which the two retrievers put different chunks first."""


@dataclass(frozen=True, slots=True)
class Ranked:
    """One chunk in one retriever's list; ``score`` is that retriever's own."""

    rank: int
    chunk_id: str
    doc_id: str
    score: float


@dataclass(frozen=True, slots=True)
class Contribution:
    """What one retriever added to a hybrid hit: ``weight / (k + rank)``, or 0."""

    retriever: str
    rank: int | None
    weight: float
    rrf: float


@dataclass(frozen=True, slots=True)
class ExplainedHit:
    """A hybrid hit, untouched, and the contributions that sum to its score."""

    rank: int
    hit: Hit
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True, slots=True)
class Explanation:
    """Both retrievers' lists (to the over-fetch depth) and the hybrid hits."""

    query: str
    k: int
    lists: Mapping[str, tuple[Ranked, ...]]
    hits: tuple[ExplainedHit, ...]


def explain(
    engine: MiniRAG, query: str, *, top_k: int = 5, where: Mapping[str, str] | None = None
) -> Explanation:
    """Explain ``engine.search(query, top_k=top_k, mode="hybrid", where=where)``.

    Raises:
        RuntimeError: If ``engine`` has not been indexed.
        ValueError: If ``top_k < 1``.
    """
    state = engine._state
    if state is None:
        raise RuntimeError("call index() before explain()")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    k, weights = engine._rrf_k, engine._weights
    tokens = tokenize(query)
    if not tokens:  # search() returns () here too
        return Explanation(query, k, MappingProxyType({name: () for name in RETRIEVERS}), ())

    depth, kept = max(top_k * 4, 20), dict(where or {})  # exactly what search() passes
    pairs = {name: engine._rank(state, tokens, name, depth, kept) for name in RETRIEVERS}
    fused = engine._rank(state, tokens, "hybrid", depth, kept)
    hits = MiniRAG._collapse(state, fused, "hybrid", top_k)

    chunk_index = {chunk.chunk_id: i for i, chunk in enumerate(state.chunks)}
    rank_of = {name: {i: r for r, (i, _) in enumerate(pairs[name], 1)} for name in RETRIEVERS}
    # Same default and same term as minirag.fusion, so the sums are exact.
    weight_of = {name: float(weights.get(name, 1.0)) for name in RETRIEVERS}
    explained = []
    for position, hit in enumerate(hits, 1):
        contributions = []
        for name in RETRIEVERS:
            rank, weight = rank_of[name].get(chunk_index[hit.chunk_id]), weight_of[name]
            rrf = 0.0 if rank is None else weight / (k + rank)
            contributions.append(Contribution(name, rank, weight, rrf))
        explained.append(ExplainedHit(position, hit, tuple(contributions)))

    lists = {
        name: tuple(
            Ranked(r, state.chunks[i].chunk_id, state.chunks[i].doc_id, float(score))
            for r, (i, score) in enumerate(pairs[name], 1)
        )
        for name in RETRIEVERS
    }
    return Explanation(query, k, MappingProxyType(lists), tuple(explained))


def _table(rows: Sequence[Sequence[str]], right: frozenset[int]) -> list[str]:
    """Pad each column to its widest cell; columns in ``right`` align right."""
    widths = [max(len(row[c]) for row in rows) for c in range(len(rows[0]))]
    return [
        "  "
        + "  ".join(
            cell.rjust(widths[c]) if c in right else cell.ljust(widths[c])
            for c, cell in enumerate(row)
        ).rstrip()
        for row in rows
    ]


def _verdict(explanation: Explanation) -> str:
    """One computed sentence about the two lists' first places."""
    tops = {name: ranked[0].chunk_id for name, ranked in explanation.lists.items() if ranked}
    if len(tops) < len(RETRIEVERS):
        empty = [name for name in RETRIEVERS if name not in tops]
        return f"{' and '.join(empty)} ranked nothing."
    if len(set(tops.values())) == 1:
        return "bm25 and dense agree on the top chunk."
    first = explanation.hits[0].hit.chunk_id
    return f"bm25 and dense disagree on the top chunk; hybrid puts {first} first."


def format_explanation(explanation: Explanation, *, shown: int = 3) -> str:
    """Render the first ``shown`` places of each list, then every hybrid hit's sum."""
    lists = [explanation.lists[name] for name in RETRIEVERS]
    rows = [["rank", "bm25", "score", "dense", "cosine"]]
    for place in range(min(shown, max(map(len, lists)))):
        row = [str(place + 1)]
        for ranked in lists:
            if place < len(ranked):
                row += [ranked[place].chunk_id, f"{ranked[place].score:.4f}"]
            else:
                row += ["", ""]
        rows.append(row)
    lines = [f"query: {explanation.query!r}", ""]
    if len(rows) > 1:
        lines += _table(rows, right=frozenset({0, 2, 4}))
    lines.append(f"  {_verdict(explanation)}")
    if not explanation.hits:
        return "\n".join([*lines, "", "  hybrid: no hits."])

    rows = [["rank", "hybrid", *(f"{name} adds" for name in RETRIEVERS), "score"]]
    for item in explanation.hits:
        cells = [str(item.rank), item.hit.chunk_id]
        for c in item.contributions:
            if c.rank is None:
                cells.append("not ranked = 0")
            else:
                cells.append(f"{c.weight:g}/({explanation.k}+{c.rank}) = {c.rrf:.6f}")
        rows.append([*cells, f"{item.hit.score:.6f}"])
    return "\n".join([*lines, "", *_table(rows, right=frozenset({0, 4}))])


def main(argv: Sequence[str] | None = None) -> None:
    """Index the fixture corpus as the README does and explain each query."""
    parser = argparse.ArgumentParser(
        prog="python -m extras.explain",
        description="Show each retriever's ranking and what it adds to every hybrid hit.",
    )
    parser.add_argument("queries", nargs="*", help="queries to explain (default: two built in)")
    parser.add_argument("--top-k", type=int, default=3, help="hits and list places shown")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    documents, _ = load_fixture()
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)
    engine.index(documents)
    print(f"fixture corpus: {len(documents)} documents -> {engine.chunk_count} chunks")
    for query in args.queries or QUERIES:
        print()
        print(format_explanation(explain(engine, query, top_k=args.top_k), shown=args.top_k))


if __name__ == "__main__":
    main()
