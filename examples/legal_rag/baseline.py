"""A hybrid baseline inside one scope: BM25, dense and their fusion, side by side.

:func:`~examples.legal_rag.eligibility.search` runs one mode through the core
engine. :func:`compare` runs all three over the same eligible passages and
shows where they agree. Its dense side comes from :mod:`extras.embed`, so it can
be minirag's hashed TF-IDF or a learned model, and the result says which.

- **Eligible candidates only.** Everything is built from a
  :class:`~examples.legal_rag.eligibility.ScopedView`'s passages: the BM25
  statistics, the hashed TF-IDF weights and the vectors. Every ID that can come
  back was taken from the view. A hashed TF-IDF embedder must have been fitted
  on exactly those passages, because its weights are corpus statistics: one
  fitted on a wider corpus would carry other tenants' text into the scores, so
  it is refused. A learned model embeds each text on its own, so one loaded
  model can serve every scope.
- **Raw text.** A learned model gets each passage's exact text and the query
  as typed. BM25 and hashed TF-IDF read the core tokeniser's tokens.
- **Deterministic ties.** BM25 and dense ties break on chunk ID. Hybrid is the
  core's weighted RRF, whose ties break on best rank, then BM25-first order.
- **Labelled spaces.** The result carries the dense side's
  :class:`~extras.embed.Space`, so a hashed TF-IDF run is never reported as a
  semantic one.

With the default hashed embedder the three rankings match the core engine's
modes over the same passages; a test checks that. This is a baseline to
compare against on 13 fixture passages, not a measurement. Held-out
comparisons belong to rag-eval-lab. Run
``python -m examples.legal_rag.baseline --help``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from examples.legal_rag.eligibility import (
    EligibilityError,
    Scope,
    ScopedView,
    _iso_day,
    scoped_view,
)
from examples.legal_rag.ingest import ingest
from examples.legal_rag.provenance import FIXTURE, ProvenanceError, load
from extras.embed import (
    HashedTfidfText,
    SentenceTransformerEmbedder,
    Space,
    TextEmbedder,
    VectorIndex,
)
from minirag import bm25
from minirag.fusion import reciprocal_rank_fusion
from minirag.tokenize import tokenize

DEPTH: Final = 20
"""Candidates each retriever hands to fusion, as the core engine over-fetches."""
RETRIEVERS: Final = ("bm25", "dense")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked passage and where each retriever put it (``None``: not in its top ``depth``)."""

    chunk_id: str
    score: float
    bm25_rank: int | None
    dense_rank: int | None


@dataclass(frozen=True, slots=True)
class Baseline:
    """The three rankings of one query in one scope; scores order one list only."""

    space: Space
    bm25: tuple[Candidate, ...]
    dense: tuple[Candidate, ...]
    hybrid: tuple[Candidate, ...]


def _check(top_k: object, depth: object, weights: Mapping[str, float] | None) -> dict[str, float]:
    if type(top_k) is not int or type(depth) is not int or not 1 <= top_k <= depth:
        raise ValueError("top_k and depth must be ints with 1 <= top_k <= depth")
    weights = dict(weights if weights is not None else {"bm25": 1.0, "dense": 1.0})
    if not set(weights) <= set(RETRIEVERS) or not all(
            type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights.values()):
        raise ValueError("weights map 'bm25'/'dense' to finite numbers >= 0")
    return weights


def _embedder(given: TextEmbedder | None, texts: tuple[str, ...]) -> TextEmbedder:
    """The dense side: the given embedder, or hashed TF-IDF fitted on ``texts`` alone."""
    if given is None:
        return HashedTfidfText.fit(texts)
    space = getattr(given, "space", None)
    if type(space) is not Space:
        raise TypeError("the embedder must name its Space")
    if space.kind == "hashed-tfidf" and space != HashedTfidfText.fit(texts, dim=space.dim).space:
        raise EligibilityError("a hashed TF-IDF embedder must be fitted on this scope's passages")
    return given


def compare(
    view: ScopedView, query: str, embedder: TextEmbedder | None = None, *,
    top_k: int = 5, depth: int = DEPTH, weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
) -> Baseline:
    """Rank ``view``'s passages for ``query`` by BM25, dense and their RRF fusion.

    A blank query, or a scope without passages, gives three empty lists.

    Raises:
        TypeError: If ``view`` is not a :class:`ScopedView`, ``query`` is not a
            string, or the embedder names no :class:`~extras.embed.Space`.
        ValueError: If ``top_k``/``depth``/``weights``/``rrf_k`` are invalid,
            or the embedder's output is malformed.
        EligibilityError: If a hashed TF-IDF embedder was fitted on other text.
    """
    if not isinstance(view, ScopedView):
        raise TypeError("compare needs a ScopedView from scoped_view()")
    if type(query) is not str:
        raise TypeError("query must be a plain string")
    weights = _check(top_k, depth, weights)
    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be an int >= 1")
    ids = tuple(sorted(view.children))
    texts = tuple(view.children[cid].text for cid in ids)
    embedder = _embedder(embedder, texts)
    if not ids or not query.strip():
        return Baseline(embedder.space, (), (), ())
    lexical = tuple((ids[i], score) for i, score in
                    bm25.search(bm25.build_index([tokenize(t) for t in texts]),
                                tokenize(query), top_k=depth))
    dense = VectorIndex.build(embedder, ids, texts).search(embedder, query, top_k=depth)
    ranks = {name: {cid: r for r, (cid, _) in enumerate(pairs, start=1)}
             for name, pairs in (("bm25", lexical), ("dense", dense))}
    fused = reciprocal_rank_fusion({name: list(ranks[name]) for name in RETRIEVERS},
                                   weights=weights, k=rrf_k, top_k=top_k)

    def rows(pairs: Sequence[tuple[str, float]]) -> tuple[Candidate, ...]:
        return tuple(Candidate(cid, float(score), ranks["bm25"].get(cid), ranks["dense"].get(cid))
                     for cid, score in pairs[:top_k])

    return Baseline(embedder.space, rows(lexical), rows(dense), rows(fused))


# --- the command line -----------------------------------------------------------


def _print(baseline: Baseline, view: ScopedView, top_k: int) -> None:
    print(f"dense side: {baseline.space.describe()}")
    if not baseline.space.pinned:
        print("  warning: the revision is not a commit hash, so these weights can change")
    print(f"eligible: {len(view.sources)} source(s), {len(view.children)} passage(s)")
    for mode in ("bm25", "dense", "hybrid"):
        rows = getattr(baseline, mode)
        print(f"{mode} top {top_k}:" + ("" if rows else " (no match)"))
        width = max(len(row.chunk_id) for row in rows) if rows else 0
        for i, row in enumerate(rows, start=1):
            seen = ", ".join(f"{name} #{rank}" if rank else f"{name} -" for name, rank in
                             (("bm25", row.bm25_rank), ("dense", row.dense_rank)))
            print(f"  {i}. {row.chunk_id:<{width}}  {row.score:.4f}  ({seen})")


def main(argv: Sequence[str] | None = None) -> int:
    """Compare BM25, dense and hybrid over one requester's passages."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.legal_rag.baseline",
        description="Rank one scope's passages three ways. Scope flags stand in for what a "
                    "server derives from an authenticated session. Without --model the dense "
                    "side is hashed TF-IDF, which is lexical, not a learned embedding.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, help="public or tenant-...")
    parser.add_argument("--matter", help="matter-... (omit for none)")
    parser.add_argument("--jurisdiction", required=True, help="fic-...")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--as-of", required=True, type=_iso_day, help="YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    learned = parser.add_argument_group("learned embeddings (needs the embeddings extra)")
    learned.add_argument("--model", help="sentence-transformers model ID or local path")
    learned.add_argument("--revision", help="the model's commit hash (required with --model)")
    learned.add_argument("--allow-download", action="store_true",
                         help="fetch the model if it is not already in the local cache")
    learned.add_argument("--device", help="e.g. cpu; the library picks one if omitted")
    args = parser.parse_args(argv)
    if (args.model is None) != (args.revision is None):
        parser.error("--model and --revision go together")
    try:
        scope = Scope(args.tenant, args.matter, args.jurisdiction, args.regime, args.as_of)
        records = load(args.fixture)
        view = scoped_view(records, ingest(records), scope)
    except (OSError, UnicodeDecodeError, ProvenanceError) as exc:
        print(f"{args.fixture}: cannot load: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # EligibilityError
        parser.error(str(exc))
    embedder = None
    if args.model is not None:
        try:
            embedder = SentenceTransformerEmbedder(
                args.model, revision=args.revision, allow_download=args.allow_download,
                device=args.device)
        except (ImportError, OSError, ValueError) as exc:
            print(f"cannot load model {args.model!r}: {exc}", file=sys.stderr)
            return 1
    try:
        baseline = compare(view, args.query, embedder, top_k=args.top_k)
    except ValueError as exc:  # a bad --top-k
        parser.error(str(exc))
    _print(baseline, view, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
