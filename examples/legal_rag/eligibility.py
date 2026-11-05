"""Eligibility before ranking: a request only ever sees what its scope allows.

A legal retriever must not rank a passage its requester may not read, cite a
version that was not in force, or answer from a jurisdiction nobody asked
about. Filtering *after* ranking is not enough: the core engine's ``where``
filter drops ineligible hits, but their text has already shaped BM25 and
TF-IDF statistics, so another tenant's documents still move this tenant's
scores. So here eligibility comes first:

1. The server builds a :class:`Scope` (tenant, matter, jurisdiction, regime,
   as-of date) from what it has authenticated, never from the query text or
   model output. A scope with a missing, look-alike or contradictory field is
   refused with :class:`EligibilityError`.
2. :func:`scoped_view` checks that the ingested passages match their source
   records, then keeps only the sources :func:`ineligibility` passes. The
   :class:`ScopedView` it returns holds nothing else.
3. :func:`search` builds a fresh index from that view's passages alone, so
   ineligible text never takes a rank, a ``top_k`` slot or a share of corpus
   statistics. Each hit is checked against the view again on the way out.
4. Expansion reads through the same view: :meth:`ScopedView.parent_of` for
   parent context, :meth:`ScopedView.resolve` for citations and, later, graph
   edges. An ID outside the view raises :class:`NotAvailable` with one fixed
   message, whether the ID belongs to someone else or to nothing at all.

The rule is fail-closed. ``"unknown"`` never matches; a source with a matter
suits only that matter; effective intervals are half-open, so a version is in
force on its ``effective_from`` and not on its ``effective_to``. The reason
names match rag-eval-lab's legal-demo labels. Error messages never echo IDs or
field values. This shows the mechanics on fictional data; it is not an access
control system, and it says nothing about whether any rule is legally right.
Run ``python -m examples.legal_rag.eligibility --help``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from examples.legal_rag.ingest import Child, Ingestion, Parent, ingest
from examples.legal_rag.provenance import (
    FIXTURE,
    LOOKALIKES,
    MAX_ID,
    PUBLIC,
    UNKNOWN,
    ProvenanceError,
    SourceRecord,
    load,
)
from minirag.engine import Document, MiniRAG, SearchMode

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_PATTERNS: Final = {
    "tenant": re.compile(rf"{PUBLIC}|tenant-{_SLUG}"),
    "matter": re.compile(rf"matter-{_SLUG}"),
    "jurisdiction": re.compile(rf"fic-{_SLUG}"),
    "regime": re.compile(_SLUG),
}
NOT_AVAILABLE: Final = "not available in this scope"


class EligibilityError(ValueError):
    """A scope or corpus that cannot be judged safely, so nothing is served."""


class NotAvailable(LookupError):
    """An ID outside the view; the message is the same whether it exists elsewhere or not."""

    def __init__(self) -> None:
        super().__init__(NOT_AVAILABLE)


def _known(value: object) -> bool:
    """A usable value: a trimmed str that is no look-alike of "unknown", nor ends in one."""
    if not isinstance(value, str) or value != value.strip() or value.lower() in LOOKALIKES:
        return False
    _, dash, rest = value.partition("-")
    return not (dash and rest.lower() in LOOKALIKES)


def _day(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


@dataclass(frozen=True, slots=True)
class Scope:
    """Who asks, about which place and regime, as of when. Set by the server, not the query."""

    tenant: str
    matter: str | None
    jurisdiction: str
    regime: str
    as_of: date

    def __post_init__(self) -> None:
        fields = {"tenant": self.tenant, "jurisdiction": self.jurisdiction, "regime": self.regime}
        if self.matter is not None:
            fields["matter"] = self.matter
        bad = [name for name, value in fields.items() if not (
            _known(value) and len(value) <= MAX_ID and _PATTERNS[name].fullmatch(value))]
        if not _day(self.as_of):
            bad.append("as_of")
        if bad:
            names = ", ".join(bad)
            raise EligibilityError(f"scope field(s) missing, unknown or malformed: {names}")
        if self.tenant == PUBLIC and self.matter is not None:
            raise EligibilityError("a public requester has no matter")


def ineligibility(record: SourceRecord, scope: Scope) -> tuple[str, ...]:
    """Every rule ``record`` fails for ``scope``; empty means eligible.

    Fails closed on its own, without trusting either constructor: a scope or
    place field that is not :func:`_known` gives an ``unknown-*`` reason,
    never a match.

    Raises:
        EligibilityError: A date is not a ``datetime.date`` or ``"unknown"``.
    """
    start, end = record.effective_from, record.effective_to
    if not (_day(scope.as_of) and (_day(start) or start == UNKNOWN)
            and (end is None or end == UNKNOWN or _day(end))):
        raise EligibilityError("eligibility dates must be datetime.date values or 'unknown'")
    reasons = []
    matters = (record.matter, scope.matter)
    if not (_known(record.tenant) and _known(scope.tenant)) or not all(
            m is None or _known(m) for m in matters):
        reasons.append("unknown-scope")
    elif record.tenant != PUBLIC and record.tenant != scope.tenant:
        reasons.append("other-tenant")
    elif record.matter is not None and record.matter != scope.matter:
        reasons.append("other-matter")
    for name in ("jurisdiction", "regime"):
        value, asked = getattr(record, name), getattr(scope, name)
        if not (_known(value) and _known(asked)):
            reasons.append(f"unknown-{name}")
        elif value != asked:
            reasons.append(f"other-{name}")
    if start == UNKNOWN:
        reasons.append("unknown-in-force")
    elif start > scope.as_of:
        reasons.append("not-yet-in-force")
    elif end == UNKNOWN:
        reasons.append("unknown-in-force")
    elif end is not None and end <= scope.as_of:
        reasons.append("no-longer-in-force")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ScopedView:
    """The sources, parents and children one scope may see, in read-only maps; nothing else."""

    scope: Scope
    sources: Mapping[str, SourceRecord]
    parents: Mapping[str, Parent]
    children: Mapping[str, Child]

    def resolve(self, ref: object) -> SourceRecord | Parent | Child:
        """A cited source, parent or child ID, or :class:`NotAvailable`."""
        if isinstance(ref, str):
            for table in (self.children, self.parents, self.sources):
                if ref in table:
                    return table[ref]
        raise NotAvailable

    def parent_of(self, chunk_id: object) -> Parent:
        """The section around a visible child, or :class:`NotAvailable`."""
        child = self.resolve(chunk_id)
        if not isinstance(child, Child) or child.parent_id not in self.parents:
            raise NotAvailable
        return self.parents[child.parent_id]


def _slice_ok(p: Parent | Child, records: Mapping[str, SourceRecord]) -> bool:
    """True if ``p`` is an exact, current slice of the record it names."""
    record = records.get(p.source_id) if isinstance(p.source_id, str) else None
    return (record is not None and p.content_hash == record.content_hash
            and type(p.start) is int and type(p.end) is int
            and 0 <= p.start < p.end <= len(record.text)
            and p.text == record.text[p.start : p.end])


def _bad_passages(records: Mapping[str, SourceRecord], ingestion: Ingestion) -> int:
    """How many passages name no record, are stale, or have IDs that do not fit."""
    parents = {p.parent_id: p for p in ingestion.parents
               if isinstance(p, Parent) and isinstance(p.parent_id, str)}
    bad = len(ingestion.parents) - len(parents)
    for p in parents.values():
        bad += not (_slice_ok(p, records) and p.parent_id == f"{p.source_id}#{p.section_id}")
    for c in ingestion.children:
        ok = isinstance(c, Child) and isinstance(c.parent_id, str)
        parent = parents.get(c.parent_id) if ok else None
        bad += not (parent is not None and _slice_ok(c, records)
                    and parent.source_id == c.source_id
                    and parent.start <= c.start and c.end <= parent.end
                    and c.chunk_id == f"{c.parent_id}:{c.start}-{c.end}")
    return bad


def scoped_view(records: Iterable[SourceRecord], ingestion: Ingestion, scope: Scope) -> ScopedView:
    """The part of ``ingestion`` that ``scope`` may see.

    Raises:
        TypeError: If an argument has the wrong type.
        EligibilityError: If two records share an ID, a matter has two tenants
            or the scope's matter belongs to another tenant, any passage does
            not match the record it names, or a date is malformed.
    """
    records = tuple(records)
    if not (isinstance(scope, Scope) and isinstance(ingestion, Ingestion)
            and all(isinstance(r, SourceRecord) for r in records)):
        raise TypeError("expected SourceRecords, an Ingestion and a Scope")
    Scope.__post_init__(scope)  # re-check, in case the frozen fields were forced
    by_id = {r.source_id: r for r in records}
    if len(by_id) != len(records):
        raise EligibilityError("two records share a source_id")
    owner: dict[str, str] = {}
    for r in records:
        if r.matter is not None and owner.setdefault(r.matter, r.tenant) != r.tenant:
            raise EligibilityError("a matter belongs to two tenants")
    if scope.matter is not None and owner.get(scope.matter, scope.tenant) != scope.tenant:
        raise EligibilityError("the scope's matter belongs to another tenant")
    bad = _bad_passages(by_id, ingestion)
    if bad:
        raise EligibilityError(f"{bad} passage(s) do not match their source record")
    visible = {sid for sid, r in by_id.items() if not ineligibility(r, scope)}
    return ScopedView(
        scope=scope,
        sources=MappingProxyType({sid: by_id[sid] for sid in sorted(visible)}),
        parents=MappingProxyType({p.parent_id: p for p in ingestion.parents
                                  if p.source_id in visible}),
        children=MappingProxyType({c.chunk_id: c for c in ingestion.children
                                   if c.source_id in visible}),
    )


@dataclass(frozen=True, slots=True)
class Retrieved:
    """One ranked child passage; ``score`` orders one result set only."""

    child: Child
    score: float
    mode: SearchMode


def search(
    view: ScopedView, query: str, *, top_k: int = 5, mode: SearchMode = "hybrid"
) -> tuple[Retrieved, ...]:
    """Rank ``view``'s children for ``query`` with an index built from them alone.

    The index is rebuilt on every call, which is fine for a fixture and
    linear in the view's text; caching it per scope is a later step.

    Raises:
        TypeError: If ``view`` is not a :class:`ScopedView`.
        ValueError: From the core engine, for a bad ``top_k`` or ``mode``.
        EligibilityError: If a hit falls outside the view (an invariant breach).
    """
    if not isinstance(view, ScopedView):
        raise TypeError("search needs a ScopedView from scoped_view()")
    if not view.children:
        return ()
    engine = MiniRAG()
    engine.index([Document(cid, view.children[cid].text) for cid in sorted(view.children)])
    results = []
    for hit in engine.search(query, top_k=top_k, mode=mode):
        child = view.children.get(hit.doc_id)
        if child is None:
            raise EligibilityError("retrieval returned a passage outside the scope")
        results.append(Retrieved(child, hit.score, mode))
    return tuple(results)


# --- the command line -----------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Show which sources one scope may see, why the rest are out, and the top passages."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.legal_rag.eligibility",
        description="Search only the passages one requester's scope allows. The flags "
                    "stand in for what a server derives from an authenticated session.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, help="public or tenant-...")
    parser.add_argument("--matter", help="matter-... (omit for none)")
    parser.add_argument("--jurisdiction", required=True, help="fic-...")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)
    try:
        scope = Scope(args.tenant, args.matter, args.jurisdiction, args.regime, args.as_of)
        records = load(args.fixture)
        view = scoped_view(records, ingest(records), scope)
        hits = search(view, args.query, top_k=args.top_k, mode=args.mode)
    except (OSError, UnicodeDecodeError, ProvenanceError) as exc:
        print(f"{args.fixture}: cannot load: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # EligibilityError, or a bad --top-k
        parser.error(str(exc))
    print(f"eligible: {len(view.sources)} of {len(records)} sources")
    for sid in view.sources:
        print(f"  {sid}")
    print("excluded (operator view; a requester must not see this list):")
    for r in sorted(records, key=lambda r: r.source_id):
        if r.source_id not in view.sources:
            print(f"  {r.source_id:<36}{', '.join(ineligibility(r, scope))}")
    print(f"{args.mode} top {args.top_k}:")
    width = max((len(hit.child.chunk_id) for hit in hits), default=0)
    for i, hit in enumerate(hits, start=1):
        c = hit.child
        preview = c.text if len(c.text) <= 40 else c.text[:37] + "..."
        print(f"  {i}. {c.chunk_id:<{width}}  {hit.score:.4f}  {preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
