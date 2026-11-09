"""Parent retrieval and context packing: ranked passages in, quotable evidence out.

Retrieval ranks small child passages, because small passages match a query
sharply. An answer usually needs the whole section around them, and it must be
able to quote what it was shown. :func:`pack` turns a ranking from one
:class:`~examples.legal_rag.eligibility.ScopedView` into a context of numbered
evidence blocks that fits a declared token budget:

- **Parents through the view.** Each ranked child is expanded to its section
  with :meth:`ScopedView.parent_of`, the same scoped lookup every expansion
  uses. A ranked ID the view cannot show (another tenant's passage, a parent or
  source missing from the view, an ID that names nothing) fails the whole call
  with :class:`~examples.legal_rag.eligibility.NotAvailable`. A ranking that
  carries an ID from outside its scope is a bug upstream, and packing around it
  would hide that. A passage that is not an exact, current slice of its record
  fails with :class:`~examples.legal_rag.eligibility.EligibilityError`.
- **Overlap is removed by position, never by text.** Children of one section
  collapse into that section. When only excerpts fit, ranked children that
  overlap merge into one span, so no text is packed twice. Two passages with
  the same words from different sources or versions stay separate, each with
  its own version and content hash.
- **Every block is an exact quote.** A block's text is always
  ``record.text[start:end]`` with edge whitespace left out, the whole section
  or an excerpt of it, never a cut. A block that does not fit is left out and
  listed in :attr:`Packed.omitted` with its reason: ``"over-budget"`` (it fits
  on its own but not with what is already packed) or ``"oversized"`` (it alone
  exceeds the budget). A section that does not fit falls back to its ranked
  excerpts before it is left out.
- **Source diversity.** At most ``max_per_family`` sections of one source
  family (every version of one instrument) are packed; later ones are listed as
  ``"family-cap"``, so one long document cannot crowd out every other source.
- **A declared, labelled budget.** The budget covers the whole rendered
  context, headers and separators included, and is checked on exactly the
  string that is returned. Counting is done by a :class:`TokenCounter`. The
  default :class:`CharEstimate` is ``ceil(chars / 4)``: an estimate, labelled
  as one, which a real model tokenizer can exceed. Only a counter that wraps a
  model's own tokenizer should say ``exact=True``.

Blocks follow the ranking: block 1 holds the best-ranked passage. The rendered
headers are for a model to cite by number; they are not a trust boundary, since
evidence text can contain anything a header can. Run
``python -m examples.legal_rag.context --help``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal, Protocol

from examples.legal_rag.eligibility import (
    MODES,
    EligibilityError,
    NotAvailable,
    Scope,
    ScopedView,
    _iso_day,
    _slice_ok,
    scoped_view,
    search,
)
from examples.legal_rag.ingest import MAX_WORDS, OVERLAP_WORDS, Child, Parent, Span, _trim, ingest
from examples.legal_rag.provenance import (
    FIXTURE,
    MAX_INT,
    UNKNOWN,
    ProvenanceError,
    SourceRecord,
    Unknown,
    load,
)

MAX_PER_FAMILY: Final = 2
SEPARATOR: Final = "\n\n"
Granularity = Literal["section", "excerpt"]
Reason = Literal["family-cap", "over-budget", "oversized"]


class TokenCounter(Protocol):
    """Counts tokens for the budget; ``label`` says how, ``exact`` whether it is a model's own."""

    @property
    def label(self) -> str: ...

    @property
    def exact(self) -> bool: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class CharEstimate:
    """``ceil(len(text) / chars_per_token)``: a rough estimate, not any model's tokenizer."""

    chars_per_token: int = 4

    def __post_init__(self) -> None:
        if type(self.chars_per_token) is not int or self.chars_per_token < 1:
            raise ValueError("chars_per_token must be an int >= 1")

    @property
    def label(self) -> str:
        return f"estimate: ceil(chars / {self.chars_per_token}), not a model tokenizer"

    @property
    def exact(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return -(-len(text) // self.chars_per_token)


@dataclass(frozen=True, slots=True)
class Evidence:
    """One numbered block: ``text == record.text[start:end]``, a section or an excerpt of it."""

    ref: int
    granularity: Granularity
    source_id: str
    family_id: str
    version: int
    parent_id: str
    section_id: str
    pages: tuple[int, int] | Unknown
    start: int
    end: int
    content_hash: str
    chunk_ids: tuple[str, ...]  # the ranked passages it covers, best first
    ranks: tuple[int, ...]  # their 1-based ranks, in the same order
    tokens: int  # this block rendered on its own
    text: str

    def header(self) -> str:
        if self.pages == UNKNOWN:
            pages = "pages unknown"
        else:
            first, last = self.pages
            pages = f"p. {first}" if first == last else f"pp. {first}-{last}"
        part = "excerpt of " if self.granularity == "excerpt" else ""
        return (f"[{self.ref}] {self.source_id} {part}{self.section_id}, {pages}, "
                f"chars {self.start}-{self.end}")

    def render(self) -> str:
        return f"{self.header()}\n{self.text}"


@dataclass(frozen=True, slots=True)
class Omitted:
    """Ranked evidence left out whole; ``tokens`` is its block alone (None for a family cap)."""

    parent_id: str
    chunk_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    reason: Reason
    tokens: int | None


@dataclass(frozen=True, slots=True)
class Packed:
    """The context string, the blocks in it, and what was left out and why."""

    budget: int
    counter: str
    exact: bool
    tokens: int
    evidence: tuple[Evidence, ...]
    omitted: tuple[Omitted, ...]
    text: str


@dataclass(frozen=True, slots=True)
class _Group:
    """One section, its trimmed span and its ranked children as ``(rank, child)``, best first."""

    parent: Parent
    record: SourceRecord
    span: tuple[int, int]
    hits: tuple[tuple[int, Child], ...]


def _positive(name: str, value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_INT:
        raise ValueError(f"{name} must be an int from 1 to {MAX_INT}")
    return value


def _counter(given: object) -> TokenCounter:
    counter = CharEstimate() if given is None else given
    label, exact = getattr(counter, "label", None), getattr(counter, "exact", None)
    if not (type(label) is str and label.strip() and type(exact) is bool
            and callable(getattr(counter, "count", None))):
        raise TypeError("a token counter needs a str label, a bool exact and a count method")
    return counter  # type: ignore[return-value]


def _count(counter: TokenCounter, text: str) -> int:
    n = counter.count(text)
    if type(n) is not int or n < 0:
        raise ValueError("the token counter must return an int >= 0")
    return n


def _resolve(view: ScopedView, chunk_id: object) -> tuple[Child, Parent, SourceRecord, Span]:
    """A ranked child, its section, its record and the section's trimmed span, via ``view``."""
    child = view.children.get(chunk_id) if type(chunk_id) is str else None
    if not isinstance(child, Child) or type(child.parent_id) is not str:
        raise NotAvailable
    parent = view.parent_of(chunk_id)
    if not isinstance(parent, Parent):
        raise EligibilityError("a ranked passage does not match its source record")
    record = view.sources.get(parent.source_id) if type(parent.source_id) is str else None
    if not isinstance(record, SourceRecord):
        raise NotAvailable
    span = None
    if (record.source_id == parent.source_id == child.source_id
            and parent.parent_id == child.parent_id
            and _slice_ok(parent, view.sources) and _slice_ok(child, view.sources)
            and parent.start <= child.start and child.end <= parent.end
            and _trim(record.text, child.start, child.end) == (child.start, child.end)):
        span = _trim(record.text, parent.start, parent.end)
    if span is None:  # a genuine child is non-empty and has no edge whitespace
        raise EligibilityError("a ranked passage does not match its source record")
    return child, parent, record, span


def _groups(view: ScopedView, ranked: Sequence[object]) -> list[_Group]:
    """Ranked children grouped by section, sections in order of their best child."""
    resolved = [_resolve(view, chunk_id) for chunk_id in ranked]  # leaks are reported first
    seen: set[str] = set()
    for i, (child, *_) in enumerate(resolved):
        if child.chunk_id in seen:
            raise ValueError(f"ranked[{i}] repeats an earlier chunk ID")
        seen.add(child.chunk_id)
    groups: dict[str, tuple[Parent, SourceRecord, Span, list[tuple[int, Child]]]] = {}
    for rank, (child, parent, record, span) in enumerate(resolved, start=1):
        groups.setdefault(parent.parent_id, (parent, record, span, []))[3].append((rank, child))
    return [_Group(p, r, s, tuple(hits)) for p, r, s, hits in groups.values()]


Excerpt = tuple[int, int, tuple[int, ...], tuple[str, ...]]


def _excerpts(group: _Group) -> list[Excerpt]:
    """Ranked children merged where they share text, as ``(start, end, ranks, ids)``, best first.

    Merging only bridges overlap, so ``record.text[start:end]`` is still one
    exact slice. Children that merely sit side by side stay separate excerpts,
    so the best-ranked one can be packed even when its neighbour does not fit.
    """
    merged: list[tuple[int, int, list[tuple[int, str]]]] = []
    for rank, child in sorted(group.hits, key=lambda hit: hit[1].start):
        if merged and child.start < merged[-1][1]:
            start, end, members = merged[-1]
            merged[-1] = (start, max(end, child.end), [*members, (rank, child.chunk_id)])
        else:
            merged.append((child.start, child.end, [(rank, child.chunk_id)]))
    spans = [(start, end, tuple(r for r, _ in sorted(members)),
              tuple(cid for _, cid in sorted(members))) for start, end, members in merged]
    return sorted(spans, key=lambda span: span[2][0])


class _Packer:
    """Greedy packing in rank order; every fit is checked on the full rendered context."""

    def __init__(self, budget: int, counter: TokenCounter, max_per_family: int) -> None:
        self.budget, self.counter, self.max_per_family = budget, counter, max_per_family
        self.packed: list[Evidence] = []
        self.omitted: list[Omitted] = []
        self.per_family: dict[str, int] = {}

    def block(self, group: _Group, granularity: Granularity, excerpt: Excerpt) -> Evidence:
        start, end, ranks, ids = excerpt
        record, parent = group.record, group.parent
        block = Evidence(
            ref=len(self.packed) + 1, granularity=granularity, source_id=record.source_id,
            family_id=record.family_id, version=record.version, parent_id=parent.parent_id,
            section_id=parent.section_id, pages=parent.pages, start=start, end=end,
            content_hash=record.content_hash, chunk_ids=ids, ranks=ranks, tokens=0,
            text=record.text[start:end])
        return replace(block, tokens=_count(self.counter, block.render()))

    def fits(self, block: Evidence) -> bool:
        rendered = SEPARATOR.join(b.render() for b in (*self.packed, block))
        return _count(self.counter, rendered) <= self.budget

    def omit(self, block: Evidence) -> None:
        reason: Reason = "oversized" if block.tokens > self.budget else "over-budget"
        self.omitted.append(Omitted(block.parent_id, block.chunk_ids, block.ranks, reason,
                                    block.tokens))

    def take(self, block: Evidence) -> bool:
        """Pack ``block`` if the whole context still fits; otherwise list it as omitted."""
        if self.fits(block):
            self.packed.append(block)
            return True
        self.omit(block)
        return False

    def add(self, group: _Group) -> None:
        """The whole section if it fits, else its ranked excerpts that fit, else nothing."""
        family = group.record.family_id
        ranks = tuple(rank for rank, _ in group.hits)
        ids = tuple(child.chunk_id for _, child in group.hits)
        if self.per_family.get(family, 0) >= self.max_per_family:
            self.omitted.append(Omitted(group.parent.parent_id, ids, ranks, "family-cap", None))
            return
        section = self.block(group, "section", (*group.span, ranks, ids))
        excerpts = [e for e in _excerpts(group) if e[:2] != group.span]
        if self.fits(section):
            self.packed.append(section)
            took = True
        elif not excerpts:  # its one excerpt would be the section again
            self.omit(section)
            took = False
        else:  # each block is built after the one before it was packed or left out
            took = any([self.take(self.block(group, "excerpt", e)) for e in excerpts])
        if took:
            self.per_family[family] = self.per_family.get(family, 0) + 1


def pack(
    view: ScopedView, ranked: Sequence[str], *, budget: int,
    counter: TokenCounter | None = None, max_per_family: int = MAX_PER_FAMILY,
) -> Packed:
    """Pack the sections around ``ranked`` (chunk IDs, best first) into ``budget`` tokens.

    Any ranking of ``view``'s children works: :func:`~examples.legal_rag.eligibility.search`,
    a baseline list, or a fused list from elsewhere. An empty ranking gives an
    empty context.

    Raises:
        TypeError: If ``view`` is not a :class:`ScopedView`, ``ranked`` is not a
            sequence, or ``counter`` lacks a ``label``, ``exact`` or ``count``.
        ValueError: If ``budget`` or ``max_per_family`` is not an int from 1 to
            :data:`~examples.legal_rag.provenance.MAX_INT`, ``ranked`` is longer
            than the view's passages or repeats one, or the counter returns
            something other than an int >= 0.
        NotAvailable: If a ranked ID, its section or its record is outside the view.
        EligibilityError: If a ranked passage is not an exact slice of its record.
    """
    if not isinstance(view, ScopedView):
        raise TypeError("pack needs a ScopedView from scoped_view()")
    if isinstance(ranked, str | bytes) or not isinstance(ranked, Sequence):
        raise TypeError("ranked must be a sequence of chunk IDs, best first")
    budget = _positive("budget", budget)
    max_per_family = _positive("max_per_family", max_per_family)
    counter = _counter(counter)
    if len(ranked) > len(view.children):
        raise ValueError("ranked has more IDs than the view has passages")
    packer = _Packer(budget, counter, max_per_family)
    for group in _groups(view, ranked):
        packer.add(group)
    context = SEPARATOR.join(block.render() for block in packer.packed)
    return Packed(budget, counter.label, counter.exact, _count(counter, context),
                  tuple(packer.packed), tuple(packer.omitted), context)


# --- the command line -----------------------------------------------------------


def _print(packed: Packed, view: ScopedView, hits: int, mode: str, top_k: int) -> None:
    print(f"eligible: {len(view.sources)} source(s), {len(view.children)} passage(s); "
          f"{mode} top {top_k}: {hits} hit(s)")
    kind = "exact" if packed.exact else "estimated"
    print(f"tokens: {packed.tokens} of {packed.budget}, {kind} ({packed.counter})")
    print(f"packed {len(packed.evidence)} block(s):")
    width = max((len(b.parent_id) for b in packed.evidence), default=0)
    for b in packed.evidence:
        print(f"  [{b.ref}] {b.granularity:<8} {b.parent_id:<{width}}  chars {b.start}-{b.end}, "
              f"{b.tokens} tokens, {_ranks(b.ranks)}")
    print(f"omitted {len(packed.omitted)}:" + ("" if packed.omitted else " none"))
    for o in packed.omitted:
        size = "" if o.tokens is None else f", {o.tokens} tokens alone"
        print(f"  {o.parent_id}: {o.reason}{size}, {_ranks(o.ranks)}")
    print("context:")
    print(packed.text)


def _ranks(ranks: tuple[int, ...]) -> str:
    return ("rank " if len(ranks) == 1 else "ranks ") + ", ".join(f"#{r}" for r in ranks)


def main(argv: Sequence[str] | None = None) -> int:
    """Search one scope, then pack the sections around the hits into a token budget."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.legal_rag.context",
        description="Rank one scope's passages, then pack their sections into a declared "
                    "token budget. Scope flags stand in for what a server derives from an "
                    "authenticated session. Token counts are estimates (ceil(chars / 4)).",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant", required=True, help="public or tenant-...")
    parser.add_argument("--matter", help="matter-... (omit for none)")
    parser.add_argument("--jurisdiction", required=True, help="fic-...")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--as-of", required=True, type=_iso_day, help="YYYY-MM-DD")
    parser.add_argument("--budget", type=int, required=True, help="estimated tokens")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--max-per-family", type=int, default=MAX_PER_FAMILY)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)
    try:
        scope = Scope(args.tenant, args.matter, args.jurisdiction, args.regime, args.as_of)
        records = load(args.fixture)
        ingestion = ingest(records, max_words=args.max_words, overlap_words=args.overlap_words)
        view = scoped_view(records, ingestion, scope)
        hits = search(view, args.query, top_k=args.top_k, mode=args.mode)
        packed = pack(view, [hit.child.chunk_id for hit in hits], budget=args.budget,
                      max_per_family=args.max_per_family)
    except (OSError, UnicodeDecodeError, ProvenanceError) as exc:
        print(f"{args.fixture}: cannot load: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # EligibilityError, or a bad limit
        parser.error(str(exc))
    _print(packed, view, len(hits), args.mode, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
