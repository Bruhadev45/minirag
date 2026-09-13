"""Weighted Reciprocal Rank Fusion: combine rankings you cannot compare.

BM25 returns unbounded scores (0 to ~30, corpus-dependent); cosine returns
[-1, 1]. Adding them means the lexical list wins by unit choice alone, and
min-max rescaling per query turns "everything matched badly" into "the
least bad result is perfect". RRF instead throws the scores away and keeps
only the ranks, which are comparable by construction::

    RRF(d) = SUM over retrievers r of  w_r / (k + rank_r(d))

``rank`` is 1-based. ``k`` (60 by convention, from Cormack et al. 2009)
flattens the curve near the top -- at k=60 rank 1 beats rank 2 by about
1.6% -- so a document ranked 3rd by both retrievers beats one ranked 1st by
one and absent from the other. Consensus over confidence; lower k inverts
it and rewards a single strong opinion.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[T]],
    *,
    weights: Mapping[str, float] | None = None,
    k: int = 60,
    top_k: int | None = None,
) -> tuple[tuple[T, float], ...]:
    """Fuse named ranked lists into one.

    Args:
        rankings: ``{retriever_name: [best_item, ...]}``. Lists may differ
            in length and need not contain the same items.
        weights: Per-retriever multiplier, default ``1.0`` each. Names not
            in ``rankings`` are ignored.
        k: Rank-smoothing constant. Must be >= 1.
        top_k: Truncate the result. ``None`` returns everything seen.

    Returns:
        ``(item, fused_score)`` pairs, best first. Ties break on the item's
        best rank across retrievers, then first-seen order, deterministically.

    Raises:
        ValueError: If ``k < 1``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    weights = weights or {}

    scores: dict[T, float] = {}
    best_rank: dict[T, int] = {}
    order: dict[T, int] = {}
    for name, ranked in rankings.items():
        weight = float(weights.get(name, 1.0))
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            order.setdefault(item, len(order))

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], order[kv[0]]))
    return tuple(fused if top_k is None else fused[:top_k])
