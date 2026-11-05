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
2. :func:`scoped_view` checks that every ingested passage is an exact, current
   slice of the record it names, then keeps only the sources
   :func:`ineligibility` passes. Only a :class:`ScopedView` that
   :func:`scoped_view` returned is trusted; one built by hand is not checked.
3. :func:`search` builds a fresh index from that view's passages alone, so
   ineligible text never takes a rank, a ``top_k`` slot or a share of corpus
   statistics. Each hit is checked against the view again on the way out.
4. Expansion reads through the same view: :meth:`ScopedView.parent_of` for
   parent context, :meth:`ScopedView.resolve` for citations and, later, graph
   edges. An ID outside the view raises :class:`NotAvailable` with one fixed
   message, whether the ID belongs to someone else or to nothing at all.

The rule is fail-closed. ``"unknown"`` never matches; a source with a matter
suits only that matter; effective intervals are half-open, so a version is in
force on its ``effective_from`` and not on its ``effective_to``. Reason names
and their order match rag-eval-lab's legal-demo labels; this module is a little
stricter, treating every loader look-alike (``"n/a"``, ``"tbd"``,
``"tenant-unknown"`` ...) as unknown, which only matters for values the loader
already rejects. A visible record's ``supersedes`` may name a version outside
the view. Versions of a family share their scope, so that is never another
tenant's source, and resolving it still fails.

Error messages never echo IDs or field values. A corpus error (a stale passage,
a matter with two tenants) refuses every request until it is fixed, and a scope
error can reveal facts about other tenants' matters, so a service should keep
both operator-side. This shows the mechanics on fictional data; it is not an
access-control system, and it says nothing about whether any rule is legally
right. Run ``python -m examples.legal_rag.eligibility --help``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeGuard, get_args

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
_ISO_DAY: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
MODES: Final = get_args(SearchMode)
NOT_AVAILABLE: Final = "not available in this scope"


class EligibilityError(ValueError):
    """A scope or corpus that cannot be judged safely, so nothing is served."""


class NotAvailable(LookupError):
    """An ID outside the view; the message is the same whether it exists elsewhere or not."""

    def __init__(self) -> None:
        super().__init__(NOT_AVAILABLE)


def _known(value: object) -> TypeGuard[str]:
    """A usable value: a plain, trimmed str that is no look-alike of "unknown", nor ends in one."""
    if type(value) is not str or value != value.strip() or value.lower() in LOOKALIKES:
        return False
    _, dash, rest = value.partition("-")
    return not (dash and rest.lower() in LOOKALIKES)


def _day(value: object) -> bool:
    """A plain ``date``: not a ``datetime``, nor a subclass that could redefine comparison."""
    return type(value) is date


def _field_ok(name: str, value: object) -> bool:
    return _known(value) and len(value) <= MAX_ID and bool(_PATTERNS[name].fullmatch(value))


@dataclass(frozen=True, slots=True)
class Scope:
    """Who asks, about which place and regime, as of when. Set by the server, not the query."""

    tenant: str
    matter: str | None
    jurisdiction: str
    regime: str
    as_of: date

    def __post_init__(self) -> None:
        _check_scope(self)


def _check_scope(scope: Scope) -> None:
    """Raise :class:`EligibilityError` naming each bad field; run again by :func:`scoped_view`."""
    fields = {"tenant": scope.tenant, "jurisdiction": scope.jurisdiction, "regime": scope.regime}
    if scope.matter is not None:
        fields["matter"] = scope.matter
    bad = [name for name, value in fields.items() if not _field_ok(name, value)]
    if not _day(scope.as_of):
        bad.append("as_of")
    if bad:
        names = ", ".join(bad)
        raise EligibilityError(f"scope field(s) missing, unknown or malformed: {names}")
    if scope.tenant == PUBLIC and scope.matter is not None:
        raise EligibilityError("a public requester has no matter")


def ineligibility(record: SourceRecord, scope: Scope) -> tuple[str, ...]:
    """Every rule ``record`` fails for ``scope``; empty means eligible.

    Fails closed on its own, without trusting either constructor: a scope or
    place field that is not a plain, known string gives an ``unknown-*``
    reason, never a match.

    Raises:
        EligibilityError: A date is not a ``datetime.date`` or ``"unknown"``
            (rag-eval-lab raises ``TypeError`` here).
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
        if type(ref) is str:
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
    record = records.get(p.source_id) if type(p.source_id) is str else None
    return (record is not None and type(p.text) is str and type(p.content_hash) is str
            and p.content_hash == record.content_hash
            and type(p.start) is int and type(p.end) is int
            and 0 <= p.start < p.end <= len(record.text)
            and p.text == record.text[p.start : p.end])


def _parent_ok(p: Parent, records: Mapping[str, SourceRecord], kids: Sequence[str]) -> bool:
    """A parent is its record's section, field for field, and lists exactly its children."""
    if not _slice_ok(p, records):
        return False
    section = next((s for s in records[p.source_id].sections
                    if s.section_id == p.section_id), None)
    return (section is not None and type(p.heading) is str
            and p.parent_id == f"{p.source_id}#{p.section_id}"
            and (section.heading, section.start, section.end, section.pages)
            == (p.heading, p.start, p.end, p.pages)
            and p.child_ids == tuple(kids))


def _child_ok(c: Child, parent: Parent | None, records: Mapping[str, SourceRecord]) -> bool:
    """A child lies inside a parent of its own source, under the ID its offsets give."""
    return (parent is not None and _slice_ok(c, records)
            and parent.source_id == c.source_id
            and parent.start <= c.start and c.end <= parent.end
            and c.chunk_id == f"{c.parent_id}:{c.start}-{c.end}"
            and c.words == len(c.text.split()))  # ingest counts words the same way


def _bad_passages(records: Mapping[str, SourceRecord], ingestion: Ingestion) -> int:
    """How many passages name no record, are stale, repeat, or have fields that do not fit."""
    parents = {p.parent_id: p for p in ingestion.parents
               if isinstance(p, Parent) and type(p.parent_id) is str}
    children = [c for c in ingestion.children if isinstance(c, Child)]
    kids: dict[str, list[str]] = {}
    for c in children:
        if type(c.parent_id) is str:
            kids.setdefault(c.parent_id, []).append(c.chunk_id)
    bad = len(ingestion.parents) - len(parents) + len(ingestion.children) - len(children)
    bad += sum(not _parent_ok(p, records, kids.get(pid, ())) for pid, p in parents.items())
    bad += sum(not _child_ok(c, parents.get(c.parent_id) if type(c.parent_id) is str else None,
                             records) for c in children)
    ids = [c.chunk_id for c in children if type(c.chunk_id) is str]
    return bad + len(ids) - len(set(ids))


def scoped_view(records: Iterable[SourceRecord], ingestion: Ingestion, scope: Scope) -> ScopedView:
    """The part of ``ingestion`` that ``scope`` may see.

    Raises:
        TypeError: If an argument has the wrong type.
        EligibilityError: If the scope is invalid, a record's ID or scope is
            not a string, two records share an ID, a matter has two tenants or
            the scope's matter belongs to another tenant, any passage does not
            match the record it names, or a date is malformed.
    """
    records = tuple(records)
    if not (type(scope) is Scope and isinstance(ingestion, Ingestion)
            and all(isinstance(r, SourceRecord) for r in records)):
        raise TypeError("expected SourceRecords, an Ingestion and a Scope")
    _check_scope(scope)  # again, in case the frozen fields were forced
    if not all(type(r.source_id) is str and type(r.tenant) is str
               and (r.matter is None or type(r.matter) is str) for r in records):
        raise EligibilityError("a record's source_id, tenant or matter is not a string")
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

    Arguments are checked before the view is looked at, so a bad call fails
    the same way whether or not the scope has any passages. The index is
    rebuilt on every call: linear in the view's text, fine for a fixture.
    Caching it per scope, and limits on query size and ``top_k``, belong to the
    later session and service steps.

    Raises:
        TypeError: If ``view`` is not a :class:`ScopedView`.
        ValueError: If ``top_k`` is not an int of at least 1 or ``mode`` is unknown.
        EligibilityError: If a hit falls outside the view (an invariant breach).
    """
    if not isinstance(view, ScopedView):
        raise TypeError("search needs a ScopedView from scoped_view()")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be an int >= 1")
    if mode not in MODES:
        raise ValueError(f"unknown mode, expected one of {MODES}")
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


def _iso_day(text: str) -> date:
    """``YYYY-MM-DD`` only; newer Pythons' ``fromisoformat`` also takes other ISO forms."""
    if not _ISO_DAY.fullmatch(text):
        raise ValueError(text)
    return date.fromisoformat(text)


def main(argv: Sequence[str] | None = None) -> int:
    """Show the top passages one scope may see; with ``--show-excluded``, what it may not."""
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
    parser.add_argument("--as-of", required=True, type=_iso_day, help="YYYY-MM-DD")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--show-excluded", action="store_true",
                        help="operator view: also list every excluded source and why")
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
    print(f"eligible: {len(view.sources)} source(s)")
    for sid in view.sources:
        print(f"  {sid}")
    if args.show_excluded:
        print(f"excluded: {len(records) - len(view.sources)} of {len(records)} "
              "(operator view; a requester must not see this list)")
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
