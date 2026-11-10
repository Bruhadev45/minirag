"""Bounded graph-assisted retrieval: follow explicit links, and only through the view.

A passage can depend on another one it names: an exemption "from section 1"
means little without section 1. :func:`expand` starts from the sections around
ranked passages and follows links from them, under three rules.

- **Explicit edges only.** There are two kinds, and both are written down in
  the fixtures. Nothing is guessed from similar wording or legal reasoning.

  - A *reference*: ``fixtures/references.json`` says that the words at
    ``[start, end)`` of one section name an instrument (a source family) or
    one of its sections. The loader checks that those words are exactly there.
  - *Supersession*: a record's own ``supersedes`` field, followed both ways,
    ``"supersedes"`` to the older version and ``"superseded-by"`` to the newer.

  An edge records only that the words name the target. It says nothing about
  precedence, agreement or authority. The bundled guidance note even says
  outright that it does not say how it relates to the act it names.
- **The same eligibility policy.** A reference names an instrument, not a
  version, so it resolves to whichever versions the scoped view holds: the one
  the scope's as-of date puts in force, which need not be the one the citing
  text meant. Every section reached is read from the view and checked again
  with :func:`~examples.legal_rag.eligibility.ineligibility`, as seeds are. An
  edge whose target the view does not hold is not followed, and it is counted,
  never named: :attr:`Expansion.not_available` counts references (including one
  whose named section is missing from the version in view), and
  :attr:`Expansion.versions_not_available` counts supersession edges. Valid
  versions of one family are never in force on the same day, so within one
  view a supersession edge is always refused and only counted: the graph cannot
  bring back a superseded version or serve one that is not yet in force.
- **Bounded work.** At most :data:`MAX_HOPS` hops, :data:`MAX_CANDIDATES`
  added sections, :data:`MAX_SEEDS` seeds per call and :data:`MAX_REFERENCES`
  references per file, read from at most :data:`MAX_BYTES` bytes. Each section
  is visited once, so cycles end. The walk is breadth-first in a fixed order
  (seeds as given; then citing words by position, then supersession; targets by
  version, then section), so the same inputs give the same result. When the cap
  stops the walk, ``truncated`` is True and the counts cover the walk so far.

The loader also refuses a reference into a narrower scope: a public source
naming a tenant's memo, or a tenant-wide source naming one matter's memo. So a
counted edge can only lead to a source that is out of force, or in another
jurisdiction or regime. Only a :class:`Graph` from :func:`build_graph` or
:func:`load_graph` is trusted that far; at expansion its citing words are
checked against the view's records again. This shows the mechanics on
fictional data. Run ``python -m examples.legal_rag.graph --help``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from examples.legal_rag.context import MAX_RANKED, _printable
from examples.legal_rag.eligibility import (
    MODES,
    EligibilityError,
    NotAvailable,
    Scope,
    ScopedView,
    _iso_day,
    _slice_ok,
    ineligibility,
    scoped_view,
    search,
)
from examples.legal_rag.ingest import Parent, ingest
from examples.legal_rag.provenance import (
    FIXTURE,
    PUBLIC,
    ProvenanceError,
    SourceRecord,
    _is_int,
    _keys,
    _no_duplicate_keys,
    _show,
    load,
)

REFERENCES: Final = Path(__file__).parent / "fixtures" / "references.json"
SCHEMA: Final = "minirag-legal-references"
SCHEMA_VERSION: Final = 1
MAX_REFERENCES: Final = 1000
MAX_BYTES: Final = 1_000_000
MAX_QUOTE: Final = 200
MAX_HOPS: Final = 3
MAX_CANDIDATES: Final = 50
MAX_SEEDS: Final = MAX_RANKED
EdgeKind = Literal["reference", "supersedes", "superseded-by"]

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_FROM: Final = re.compile(rf"({_SLUG}@v[1-9][0-9]*)#(s[1-9][0-9]*)")
_TO: Final = re.compile(rf"({_SLUG})(?:#(s[1-9][0-9]*))?")
_Key = tuple[object, ...]  # identifies one edge, so each is counted once
_TOP_KEYS: Final = frozenset({"schema", "schema_version", "fictional", "notice", "references"})
_KEYS: Final = frozenset({"from", "start", "end", "quote", "to"})


@dataclass(frozen=True, slots=True)
class Reference:
    """``quote == text[start:end]`` in section ``from_id`` names ``family_id`` or one section."""

    from_id: str
    source_id: str
    content_hash: str  # of the citing record, so a stale graph is caught at expansion
    start: int
    end: int
    quote: str
    family_id: str
    section_id: str | None  # None: the whole instrument


@dataclass(frozen=True, slots=True)
class Graph:
    """Checked references, and ``(newer, older)`` supersession pairs taken from the records."""

    references: tuple[Reference, ...]
    supersessions: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Hop:
    """One followed edge; for a reference, the citing words and their span, else None."""

    kind: EdgeKind
    from_id: str
    to_id: str
    quote: str | None
    start: int | None
    end: int | None


@dataclass(frozen=True, slots=True)
class Reached:
    """A section in an expansion: a seed's own (``hops == 0``) or one reached along ``path``."""

    parent: Parent
    hops: int
    seeds: tuple[str, ...]  # the seed chunk IDs inside it, for hops == 0
    path: tuple[Hop, ...]  # the edges from a seed's section to this one


@dataclass(frozen=True, slots=True)
class Expansion:
    """Seed sections first, then reached ones in walk order, and what was not followed."""

    sections: tuple[Reached, ...]
    not_available: int  # distinct references whose target is not in the view
    versions_not_available: int  # distinct supersession edges whose version is not in the view
    truncated: bool  # the walk stopped at max_candidates
    max_hops: int
    max_candidates: int


# --- loading ----------------------------------------------------------------------


def _reference(
    raw: object, where: str, by_id: Mapping[str, SourceRecord],
    families: Mapping[str, Sequence[SourceRecord]], out: list[str],
) -> Reference | None:
    """One checked reference, or None after appending its problems to ``out``."""
    if not _keys(raw, _KEYS, where, out) or not isinstance(raw, Mapping):
        return None
    before = len(out)
    start, end, quote = raw["start"], raw["end"], raw["quote"]
    origin = _FROM.fullmatch(raw["from"]) if type(raw["from"]) is str else None
    record = by_id.get(origin[1]) if origin else None
    section = next((s for s in record.sections if s.section_id == origin[2]), None) \
        if record and origin else None
    if section is None:
        out.append(f"{where}.from: {_show(raw['from'])} is not '<source_id>#<section_id>' "
                   "of a record here")
    if not (_is_int(start, 0) and _is_int(end, 1) and start < end):
        out.append(f"{where}: start and end must be ints with start < end")
    elif section is not None and not section.start <= start < end <= section.end:
        out.append(f"{where}: [start, end) must lie inside the citing section")
    elif record is not None and (type(quote) is not str or not 0 < len(quote) <= MAX_QUOTE
                                 or record.text[start:end] != quote):
        out.append(f"{where}.quote: must be exactly the text at [start, end), "
                   f"at most {MAX_QUOTE} characters")
    target = _TO.fullmatch(raw["to"]) if type(raw["to"]) is str else None
    versions = families.get(target[1], ()) if target else ()
    if not target or not versions or (target[2] is not None and not any(
            s.section_id == target[2] for v in versions for s in v.sections)):
        out.append(f"{where}.to: {_show(raw['to'])} is not a family or "
                   "'<family_id>#<section_id>' here")
    elif record is not None and origin is not None:
        if (target[1], target[2]) == (record.family_id, origin[2]):
            out.append(f"{where}: a section cannot reference itself")
        if not all(v.tenant == PUBLIC or (v.tenant == record.tenant and (
                v.matter is None or v.matter == record.matter)) for v in versions):
            out.append(f"{where}: points into a narrower scope than the citing source")
    if len(out) > before or record is None or target is None:
        return None
    return Reference(raw["from"], record.source_id, record.content_hash, start, end, quote,
                     target[1], target[2])


def build_graph(records: Iterable[SourceRecord], raw: object) -> Graph:
    """Check a parsed references file against ``records``; :class:`ProvenanceError` if broken."""
    records = tuple(records)
    if not all(isinstance(r, SourceRecord) for r in records):
        raise TypeError("expected SourceRecords")
    out: list[str] = []
    if not _keys(raw, _TOP_KEYS, "references file", out) or not isinstance(raw, Mapping):
        raise ProvenanceError(out)
    version = raw["schema_version"]
    if raw["schema"] != SCHEMA or type(version) is not int or version != SCHEMA_VERSION:
        out.append(f"references file: expected schema {SCHEMA!r} version {SCHEMA_VERSION}")
    if raw["fictional"] is not True:
        out.append("references file: 'fictional' must be true")
    if type(raw["notice"]) is not str or "fictional" not in raw["notice"].lower():
        out.append("references file.notice: must say the sources are fictional")
    items = raw["references"]
    if not isinstance(items, list) or len(items) > MAX_REFERENCES:
        out.append(f"references file.references: expected a list of at most {MAX_REFERENCES}")
        raise ProvenanceError(out)
    by_id = {r.source_id: r for r in records}
    if len(by_id) != len(records):
        out.append("records: two share a source_id")
        raise ProvenanceError(out)
    families: dict[str, list[SourceRecord]] = {}
    for r in sorted(records, key=lambda r: r.version):
        families.setdefault(r.family_id, []).append(r)
    refs: dict[_Key, Reference] = {}
    for i, item in enumerate(items):
        ref = _reference(item, f"references[{i}]", by_id, families, out)
        key = None if ref is None else (ref.from_id, ref.start, ref.end, ref.family_id,
                                        ref.section_id or "")
        if key in refs:
            out.append(f"references[{i}]: repeats an earlier reference")
        elif key is not None and ref is not None:
            refs[key] = ref
    if out:
        raise ProvenanceError(out)
    pairs = sorted((r.source_id, r.supersedes) for r in records
                   if r.supersedes is not None and r.supersedes in by_id)
    return Graph(tuple(refs[key] for key in sorted(refs)), tuple(pairs))


def load_graph(records: Iterable[SourceRecord], path: str | Path = REFERENCES) -> Graph:
    """Read a references file and check it; ``OSError``/``UnicodeDecodeError`` if unreadable."""
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ProvenanceError([f"references file is larger than {MAX_BYTES} bytes"])
    text = data.decode("utf-8")
    try:
        raw = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ProvenanceError:
        raise
    except (ValueError, RecursionError) as exc:  # bad JSON, over-long ints, deep nesting
        raise ProvenanceError([f"not valid JSON: {str(exc)[:200]}"]) from exc
    return build_graph(records, raw)


# --- expansion --------------------------------------------------------------------


def _admit(view: ScopedView, parent: object) -> Parent:
    """``parent`` if the view holds it and the eligibility policy passes its record.

    Typed ``object``: it also checks what a hand-built view's ``parent_of`` returns.
    """
    record = view.sources.get(parent.source_id) if isinstance(parent, Parent) \
        and type(parent.source_id) is str and type(parent.parent_id) is str else None
    if (record is None or not isinstance(parent, Parent)
            or view.parents.get(parent.parent_id) is not parent
            or not _slice_ok(parent, view.sources) or ineligibility(record, view.scope)):
        raise EligibilityError("an expanded section is outside the scope or does not match "
                               "its source record")
    return parent


_Edge = tuple[_Key, EdgeKind, Reference | None, tuple[Parent, ...]]


class _Edges:
    """The explicit edges out of a section, with targets looked up in the view alone."""

    def __init__(self, view: ScopedView, graph: Graph) -> None:
        self.view = view
        self.refs: dict[str, list[Reference]] = {}
        for ref in graph.references:
            self.refs.setdefault(ref.from_id, []).append(ref)
        self.older: dict[str, list[str]] = {}
        self.newer: dict[str, list[str]] = {}
        for new, old in graph.supersessions:  # lists: a hand-built Graph may repeat a version
            self.older.setdefault(new, []).append(old)
            self.newer.setdefault(old, []).append(new)
        self.sections: dict[str, list[Parent]] = {}
        for p in sorted(view.parents.values(), key=lambda p: p.start):
            self.sections.setdefault(p.source_id, []).append(p)
        self.versions: dict[str, list[str]] = {}
        for r in sorted(view.sources.values(), key=lambda r: r.version):
            self.versions.setdefault(r.family_id, []).append(r.source_id)

    def out(self, parent: Parent) -> Iterator[_Edge]:
        """Each edge from ``parent``'s section with its targets in the view, maybe none."""
        record = self.view.sources[parent.source_id]
        for ref in self.refs.get(parent.parent_id, ()):
            if (ref.source_id, ref.content_hash) != (record.source_id, record.content_hash) \
                    or not parent.start <= ref.start < ref.end <= parent.end \
                    or record.text[ref.start : ref.end] != ref.quote:
                raise EligibilityError("a reference does not match its source record")
            targets = tuple(p for sid in self.versions.get(ref.family_id, ())
                            for p in self.sections.get(sid, ())
                            if ref.section_id in (None, p.section_id))
            key = ("reference", ref.from_id, ref.start, ref.end, ref.family_id, ref.section_id)
            yield key, "reference", ref, targets
        kinds: tuple[tuple[EdgeKind, dict[str, list[str]]], ...] = (
            ("supersedes", self.older), ("superseded-by", self.newer))
        for kind, other in kinds:
            for sid in other.get(record.source_id, ()):
                if sid in self.view.sources:  # a version the view holds is never counted
                    yield (kind, record.source_id, sid), kind, None, tuple(
                        self.sections.get(sid, ()))
                else:
                    yield (kind, record.source_id, sid), kind, None, ()


def _walk(view: ScopedView, graph: Graph, found: dict[str, Reached], max_hops: int,
          max_candidates: int) -> tuple[set[_Key], bool]:
    """Breadth-first from ``found``'s seeds, adding to it; ``(edges not followed, truncated)``."""
    edges = _Edges(view, graph)
    frontier = list(found.values())
    blocked: set[_Key] = set()
    added = 0
    for hops in range(1, max_hops + 1):
        reached: list[Reached] = []
        for here in frontier:
            for key, kind, ref, targets in edges.out(here.parent):
                if not targets:
                    blocked.add(key)
                for target in targets:
                    if target.parent_id in found:  # a cycle, or reached another way first
                        continue
                    if added == max_candidates:
                        return blocked, True
                    words = (None, None, None) if ref is None else (ref.quote, ref.start, ref.end)
                    hop = Hop(kind, here.parent.parent_id, target.parent_id, *words)
                    found[target.parent_id] = Reached(_admit(view, target), hops, (),
                                                      (*here.path, hop))
                    reached.append(found[target.parent_id])
                    added += 1
        frontier = reached
    return blocked, False


def _bounded(name: str, value: object, low: int, high: int) -> int:
    """``value`` if it is an int (not a bool) from ``low`` to ``high``, else ValueError."""
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an int from {low} to {high}")
    return value


def expand(
    view: ScopedView, graph: Graph, seeds: list[str] | tuple[str, ...], *,
    max_hops: int = 1, max_candidates: int = 10,
) -> Expansion:
    """The sections around ``seeds`` (chunk IDs), then those their explicit edges reach.

    Raises:
        TypeError: If ``view``, ``graph`` or ``seeds`` has the wrong type.
        ValueError: If ``max_hops`` is not an int from 0 to :data:`MAX_HOPS`,
            ``max_candidates`` not one from 1 to :data:`MAX_CANDIDATES`, or
            ``seeds`` holds more than :data:`MAX_SEEDS` IDs or repeats one.
        NotAvailable: If a seed is not one of the view's passages. This is
            checked before repeats, so a leak is never masked.
        EligibilityError: If a section fails the eligibility policy or does
            not match its record, or a reference does not match its citing words.
    """
    if not (isinstance(view, ScopedView) and type(view.scope) is Scope):
        raise TypeError("expand needs a ScopedView from scoped_view()")
    if not isinstance(graph, Graph):
        raise TypeError("expand needs a Graph from build_graph() or load_graph()")
    if type(seeds) not in (list, tuple):  # their len() and iteration cannot disagree
        raise TypeError("seeds must be a list or tuple of chunk IDs")
    max_hops = _bounded("max_hops", max_hops, 0, MAX_HOPS)
    max_candidates = _bounded("max_candidates", max_candidates, 1, MAX_CANDIDATES)
    if len(seeds) > MAX_SEEDS:
        raise ValueError(f"seeds may hold at most {MAX_SEEDS} IDs")
    parents = [_admit(view, view.parent_of(chunk_id)) for chunk_id in seeds]  # leaks first
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds repeat a chunk ID")
    found: dict[str, Reached] = {}
    for chunk_id, parent in zip(seeds, parents, strict=True):
        prior = found.get(parent.parent_id)
        found[parent.parent_id] = Reached(parent, 0, (*(prior.seeds if prior else ()), chunk_id),
                                          ())
    blocked, truncated = _walk(view, graph, found, max_hops, max_candidates)
    references = sum(key[0] == "reference" for key in blocked)
    return Expansion(tuple(found.values()), references, len(blocked) - references, truncated,
                     max_hops, max_candidates)


# --- the command line -------------------------------------------------------------


def _line(item: Reached) -> str:
    where = f"hop {item.hops}  {item.parent.parent_id}"
    if not item.path:
        return f"{where}  seed {', '.join(item.seeds)}"
    hop = item.path[-1]
    if hop.quote is None:
        return f"{where}  {hop.kind} from {hop.from_id}"
    quote = _printable(hop.quote).replace("\n", "\\n")  # one printed line per section
    return (f"{where}  reference from {hop.from_id}, source chars {hop.start}-{hop.end}: "
            f'"{quote}"')


def main(argv: Sequence[str] | None = None) -> int:
    """Search one scope, then follow explicit edges from the sections around the hits."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.legal_rag.graph",
        description="Rank one scope's passages, then follow only explicit reference and "
                    "supersession edges from their sections, through the same scope. The "
                    "flags stand in for what a server derives from an authenticated session.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, help="public or tenant-...")
    parser.add_argument("--matter", help="matter-... (omit for none)")
    parser.add_argument("--jurisdiction", required=True, help="fic-...")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--as-of", required=True, type=_iso_day, help="YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--max-hops", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= MAX_SEEDS:
        parser.error(f"--top-k must be between 1 and {MAX_SEEDS}")
    path = args.fixture
    try:
        scope = Scope(args.tenant, args.matter, args.jurisdiction, args.regime, args.as_of)
        records = load(path)
        path = args.references
        graph = load_graph(records, path)
        path = args.fixture  # ingest re-checks the records
        view = scoped_view(records, ingest(records), scope)
        hits = search(view, args.query, top_k=args.top_k, mode=args.mode)
        result = expand(view, graph, [hit.child.chunk_id for hit in hits],
                        max_hops=args.max_hops, max_candidates=args.max_candidates)
    except (OSError, UnicodeDecodeError, ProvenanceError) as exc:
        print(f"{path}: cannot load: {exc}", file=sys.stderr)
        return 1
    except NotAvailable as exc:  # search only returns the view's own passages
        print(f"cannot expand: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # EligibilityError, or a bad limit
        parser.error(str(exc))
    print(f"eligible: {len(view.sources)} source(s); {args.mode} top {args.top_k}: "
          f"{len(hits)} hit(s); up to {result.max_hops} hop(s), "
          f"{result.max_candidates} added section(s)")
    for item in result.sections:
        print(f"  {_line(item)}")
    print(f"not followed: {result.not_available} reference(s) and "
          f"{result.versions_not_available} supersession edge(s) lead outside this scope"
          + ("; stopped at the section cap" if result.truncated else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
