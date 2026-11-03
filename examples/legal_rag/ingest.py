"""Structure-aware ingestion: each section is a *parent*, cut into *child* passages.

minirag's core chunker (:mod:`minirag.chunker`) packs sentences across a whole
document and joins them with single spaces. That is fine for search, but a legal
answer must later quote its evidence exactly and say which section of which
version it came from. So this module chunks differently, and only validated
:class:`~examples.legal_rag.provenance.SourceRecord` objects:

- **Parents follow structure.** Each section of a record becomes one
  :class:`Parent`: the heading-led span ``text[start:end]`` with its page
  range. A passage never crosses a section boundary. Text outside every section
  (the fictional-text notice and the title line) is not ingested.
- **Children are exact slices.** A :class:`Child` is a window of whole
  sentences inside one section, and ``child.text`` is exactly
  ``record.text[child.start:child.end]``: offsets are code-point indices into
  the record's text, the same units as section spans, and no whitespace is
  rewritten. Lines are boundaries too, so a sentence never spans a line break.
- **"Sentence" is a heuristic.** It is the core chunker's naive rule: ``.``,
  ``!`` or ``?``, then whitespace, then a capital, digit, quote or bracket. So
  ``Dr. Lee`` or ``s. 5. (a)`` split mid-sentence. Offsets stay exact; only
  where a passage begins or ends is rough.
- **A heading never stands alone.** It travels with the first sentence of its
  section; a section with only a heading yields one child holding the heading.
- **Windows are bounded in words, not tokens.** ``max_words`` counts
  whitespace-separated words, as the core chunker does; it is not a model
  tokenizer count. It is a soft limit: one sentence longer than it becomes one
  oversized child rather than being cut, so no passage is silently truncated.
  ``overlap_words`` repeats trailing sentences at the start of the next child,
  but only while the next child still has room for a new sentence, so every
  child ends further into the section than the one before it. It may be at
  most half of ``max_words``: a larger overlap would let a run of one-word
  sentences repeat each of them hundreds of times.
- **IDs are stable.** A parent is ``<source_id>#<section_id>`` and depends on
  the record alone. A child is ``<parent_id>:<start>-<end>``, and its offsets
  also depend on the two limits. With the same limits, re-ingesting the same
  records in any order gives equal results, and adding or removing other
  records never renames a passage. IDs are unique within one ingestion; a
  store that mixes passages from separate ingestions must key on more than
  the ID. Every parent and child carries its source's ``content_hash``: if a
  text were edited without a new version, the IDs would survive but the hash
  would not, so a consumer comparing hashes can tell the passage is stale.
- **Scope stays on the record.** Passages carry no tenant, matter,
  jurisdiction or dates. Those stay on the ``SourceRecord``, reached through
  ``source_id`` within the same ingestion, so eligibility has one source of
  truth instead of a copy on every passage.

:func:`ingest` re-checks what the IDs and offsets depend on (the ID formats,
the hash, the section spans and pages), so a hand-built or altered record
fails with :class:`~examples.legal_rag.provenance.ProvenanceError` instead of
yielding wrong offsets or a forged ID. Run
``python -m examples.legal_rag.ingest [path]``.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from examples.legal_rag.provenance import (
    FIXTURE,
    MAX_ID,
    UNKNOWN,
    ProvenanceError,
    Section,
    SourceRecord,
    Unknown,
    content_hash,
    load,
)

MAX_WORDS: Final = 120
OVERLAP_WORDS: Final = 30
# The core chunker's sentence rule: . ! or ? then whitespace then a likely
# sentence start. Naive ("Dr. Sagan" splits), and documented as such there.
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_LINE: Final = re.compile(r"[^\n]+")
# The ID shapes the loader enforces. "#", ":" and "-" can then only appear
# where this module puts them, so an ID cannot be forged from inside a field.
_SOURCE_ID: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*@v[1-9][0-9]*")
_SECTION_ID: Final = re.compile(r"s[1-9][0-9]*")

Span = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Parent:
    """One section of one source version: ``text == record.text[start:end]``."""

    parent_id: str
    source_id: str
    section_id: str
    heading: str
    start: int
    end: int
    pages: tuple[int, int] | Unknown
    content_hash: str
    text: str
    child_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Child:
    """A passage of whole sentences inside one parent: ``text == record.text[start:end]``."""

    chunk_id: str
    parent_id: str
    source_id: str
    start: int
    end: int
    words: int
    content_hash: str
    text: str


@dataclass(frozen=True, slots=True)
class Ingestion:
    """Parents and children of a corpus, in ``source_id`` then offset order."""

    max_words: int
    overlap_words: int
    parents: tuple[Parent, ...]
    children: tuple[Child, ...]


def _trim(text: str, start: int, end: int) -> Span | None:
    """``[start, end)`` without leading or trailing whitespace, or None if nothing is left."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _units(text: str, section: Section) -> list[Span]:
    """The section's packable units: its heading glued to the first sentence, then sentences."""
    heading_end = section.start + len(section.heading)
    body: list[Span] = []
    for line in _LINE.finditer(text, heading_end, section.end):
        cuts = [line.start(), *(m.end() for m in _SENTENCE_END.finditer(text, *line.span()))]
        for cut_start, cut_end in zip(cuts, [*cuts[1:], line.end()], strict=True):
            span = _trim(text, cut_start, cut_end)
            if span is not None:
                body.append(span)
    if not body:
        return [(section.start, heading_end)]
    return [(section.start, body[0][1]), *body[1:]]


def _windows(lengths: Sequence[int], max_words: int, overlap_words: int) -> list[Span]:
    """Half-open ranges of unit indices: packed up to ``max_words``, each reaching further."""
    windows: list[Span] = []
    start = 0
    while start < len(lengths):
        end, used = start, 0
        while end < len(lengths) and (end == start or used + lengths[end] <= max_words):
            used += lengths[end]
            end += 1
        windows.append((start, end))
        if end == len(lengths):
            break
        back, carried = end, 0  # carry trailing units only while the next unit still fits
        while (
            back > start + 1
            and carried + lengths[back - 1] <= overlap_words
            and carried + lengths[back - 1] + lengths[end] <= max_words
        ):
            back -= 1
            carried += lengths[back]
        start = back
    return windows


def _page_range(pages: object) -> bool:
    return (type(pages) is tuple and len(pages) == 2 and all(type(n) is int for n in pages)
            and 1 <= pages[0] <= pages[1])


def _section_ok(text: str, s: object, prev_end: int) -> bool:
    """True if ``s`` has a valid ID and pages and is a heading-led span from ``prev_end`` on."""
    if not isinstance(s, Section):
        return False
    sid, heading = s.section_id, s.heading
    if not (isinstance(sid, str) and len(sid) <= MAX_ID and _SECTION_ID.fullmatch(sid)):
        return False
    if not (s.pages == UNKNOWN or _page_range(s.pages)):
        return False
    if not (isinstance(heading, str) and heading and heading == heading.strip()):
        return False
    if not (type(s.start) is int and type(s.end) is int):
        return False
    if not prev_end <= s.start < s.end <= len(text):
        return False
    if s.start > 0 and text[s.start - 1] != "\n":
        return False
    return text[s.start : s.end].split("\n", 1)[0] == heading


def _check(i: int, record: object) -> list[str]:
    """Problems with what IDs and offsets rely on; field values are never echoed raw."""
    where = f"records[{i}]"
    if not isinstance(record, SourceRecord):
        return [f"{where}: expected a SourceRecord, got {type(record).__name__}"]
    sid = record.source_id
    if not (isinstance(sid, str) and len(sid) <= MAX_ID and _SOURCE_ID.fullmatch(sid)
            and type(record.version) is int and sid == f"{record.family_id}@v{record.version}"):
        return [f"{where}: source_id is not '<family_id>@v<version>' in the loader's format"]
    where, text = f"{where} {sid}", record.text
    try:
        hash_ok = isinstance(text, str) and content_hash(text) == record.content_hash
    except UnicodeEncodeError:  # a lone surrogate cannot be hashed as UTF-8
        hash_ok = False
    if not hash_ok:
        return [f"{where}: text does not match its content_hash"]
    if not isinstance(record.sections, tuple) or not record.sections:
        return [f"{where}: sections must be a non-empty tuple"]
    problems, prev_end = [], 0
    for j, section in enumerate(record.sections):
        if _section_ok(text, section, prev_end):
            prev_end = section.end
        else:
            problems.append(f"{where}: sections[{j}] needs a valid section_id and pages, and a "
                            "span that starts a line with its heading, after the previous one")
    return problems


def _ingest_record(
    record: SourceRecord, max_words: int, overlap_words: int
) -> Iterator[tuple[Parent, list[Child]]]:
    """Each section of ``record`` as a parent and its children."""
    text = record.text
    for section in record.sections:
        parent_id = f"{record.source_id}#{section.section_id}"
        units = _units(text, section)
        lengths = [len(text[a:b].split()) for a, b in units]
        children = []
        for first, last in _windows(lengths, max_words, overlap_words):
            start, end = units[first][0], units[last - 1][1]
            children.append(Child(
                chunk_id=f"{parent_id}:{start}-{end}", parent_id=parent_id,
                source_id=record.source_id, start=start, end=end,
                words=sum(lengths[first:last]), content_hash=record.content_hash,
                text=text[start:end],
            ))
        yield Parent(
            parent_id=parent_id, source_id=record.source_id, section_id=section.section_id,
            heading=section.heading, start=section.start, end=section.end, pages=section.pages,
            content_hash=record.content_hash, text=text[section.start : section.end],
            child_ids=tuple(c.chunk_id for c in children),
        ), children


def ingest(
    records: Iterable[SourceRecord], *, max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> Ingestion:
    """Cut validated records into section parents and sentence-window children.

    An empty corpus gives an empty :class:`Ingestion`. The result depends only
    on the records and the two limits, never on the order the records come in.

    Raises:
        ValueError: If a limit is not an int, ``max_words < 1``, or not
            ``0 <= overlap_words <= max_words // 2``.
        ProvenanceError: If a record is not a :class:`SourceRecord`, an ID field
            is malformed, its text no longer matches its hash, a section span
            or page range does not fit, or two sections would get the same ID.
    """
    for name, value in (("max_words", max_words), ("overlap_words", overlap_words)):
        if type(value) is not int:
            raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if max_words < 1 or not 0 <= overlap_words <= max_words // 2:
        raise ValueError("need max_words >= 1 and 0 <= overlap_words <= max_words // 2")
    corpus = list(records)
    problems = [problem for i, record in enumerate(corpus) for problem in _check(i, record)]
    if problems:
        raise ProvenanceError(problems)
    parents: list[Parent] = []
    children: list[Child] = []
    for record in sorted(corpus, key=lambda r: r.source_id):
        for parent, kids in _ingest_record(record, max_words, overlap_words):
            parents.append(parent)
            children.extend(kids)
    counts = Counter(p.parent_id for p in parents)
    duplicates = sorted(pid for pid, n in counts.items() if n > 1)
    if duplicates:
        raise ProvenanceError([f"{pid}: duplicate parent ID (a repeated source or section)"
                               for pid in duplicates])
    return Ingestion(max_words, overlap_words, tuple(parents), tuple(children))


# --- the command line -----------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Ingest the bundled fixture, or the file given, and list every child passage."""
    parser = argparse.ArgumentParser(
        prog="python -m examples.legal_rag.ingest",
        description="Cut a legal fixture into section parents and child passages.",
    )
    parser.add_argument("fixture", nargs="?", type=Path, default=FIXTURE)
    parser.add_argument("--max-words", type=int, default=MAX_WORDS)
    parser.add_argument("--overlap-words", type=int, default=OVERLAP_WORDS)
    args = parser.parse_args(argv)
    try:
        result = ingest(load(args.fixture), max_words=args.max_words,
                        overlap_words=args.overlap_words)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"{args.fixture}: cannot read: {exc}", file=sys.stderr)
        return 1
    except ProvenanceError as exc:
        print(f"{args.fixture}: {len(exc.problems)} problem(s)", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    except ValueError as exc:
        parser.error(str(exc))
    sources = len({p.source_id for p in result.parents})
    print(f"{sources} sources -> {len(result.parents)} sections -> {len(result.children)} "
          f"passages (max {result.max_words} words, overlap {result.overlap_words})")
    width = max((len(c.chunk_id) for c in result.children), default=0)
    for child in result.children:
        preview = child.text if len(child.text) <= 40 else child.text[:37] + "..."
        print(f"  {child.chunk_id:<{width}}  {child.words:>3} words  {preview!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
