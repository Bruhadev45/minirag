"""The source-record contract for the legal RAG example's fixture.

Every source the legal example will index is one *record*: one version of
one fictional legal-like document, plus the metadata a legal retriever must
not guess. :func:`validate_corpus` turns the raw JSON into frozen
:class:`SourceRecord` objects, or raises :class:`ProvenanceError` listing the
problems it found, so a broken fixture fails before anything is indexed. It
checks every record in one pass; a record too broken to check further (wrong
keys, no usable text) is reported once, and invalid JSON stops at its first
error.

What a record pins down, and why:

- **Identity.** ``source_id`` is ``<family_id>@v<version>``. A family is one
  instrument across its versions; the ID cannot drift from its two parts.
- **Scope.** ``tenant`` is ``"public"`` or ``tenant-...``; ``matter`` is null
  (public, or tenant-wide) or ``matter-...``. Scope is never unknown: a record
  nobody can be shown to own must not load at all. A matter belongs to one
  tenant.
- **Where and when.** ``jurisdiction`` (a ``fic-...`` code, so a real one
  cannot slip in, or ``"unknown"``), ``regime``, ``authority_type`` and the
  half-open effective interval ``[effective_from, effective_to)``.
- **Families.** Every version of a family has the same tenant, matter,
  jurisdiction, regime and authority type.
- **Versions.** A family's versions form one chain: each later version
  supersedes the one just before it. A version that is followed must give an
  end date or ``"unknown"``, never null. Where the dates compared are known,
  versions hand over on one day and never overlap, and no version ends before
  an earlier one starts. An overlap that hinges on an unknown date cannot be
  checked, so a filter must treat that date as unknown rather than as a match.
- **Spans.** Each section is a half-open ``[start, end)`` range of code-point
  offsets into ``text`` (Python ``str`` indices, not bytes). Sections are in
  order, do not overlap, start at the beginning of a line, and the first line
  of each is exactly its heading. ``pages`` is ``[first, last]``, 1-based and
  inclusive.
- **Integrity.** ``content_hash`` is the SHA-256 of the UTF-8 text, which must
  be NFC-normalised so one visible text has one hash. The hash catches drift
  and accidental edits; anyone can recompute it, so it proves nothing about
  who wrote the text.
- **Labelling.** Only ``MIT`` and ``original-fictional`` are accepted, every
  title ends with ``(fictional)``, and every text opens with :data:`NOTICE`.

**Unknown is a value, not an absence.** Every key is required. Where a fact
may be unknown, the record says ``"unknown"``; null means "none" (no matter,
no end date, supersedes nothing), never "we don't know". Look-alikes such as
``"Unknown"``, ``"n/a"``, ``""`` or ``"tenant-unknown"`` are rejected so an
unknown cannot pass as a real value.

This checks that records are complete and consistent. It cannot check that a
text is really fictional, that a scope is right, or what any rule means; see
``DATA_CARD.md``. Run ``python -m examples.legal_rag.provenance [path]``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Final, Literal

SCHEMA: Final = "minirag-legal-fixture"
SCHEMA_VERSION: Final = 1
FIXTURE: Final = Path(__file__).parent / "fixtures" / "sources.json"
NOTICE: Final = "FICTIONAL TEST TEXT. Not law and not legal advice."
UNKNOWN: Final = "unknown"
PUBLIC: Final = "public"
LICENSES: Final = frozenset({"MIT"})
ORIGINS: Final = frozenset({"original-fictional"})
AUTHORITY_TYPES: Final = frozenset(
    {"statute", "regulation", "guidance", "judgment", "contract", "internal-memo"}
)
#: Offsets, pages and versions stay below this, so every number stays printable.
MAX_INT: Final = 10**9
#: IDs, codes and labels stay this short, so a message that quotes one stays short.
MAX_ID: Final = 80
#: Spellings of "unknown" that would otherwise pass as real values (compared lowercased).
LOOKALIKES: Final = frozenset(
    {"", "unknown", "n/a", "n-a", "na", "none", "null", "tbd", "unspecified"}
)

_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_FAMILY = re.compile(_SLUG)
_TENANT = re.compile(rf"tenant-{_SLUG}")
_MATTER = re.compile(rf"matter-{_SLUG}")
_JURISDICTION = re.compile(rf"fic-{_SLUG}")
#: Patterns whose part after the first "-" must not be a look-alike either.
_PREFIXED: Final = (_TENANT, _MATTER, _JURISDICTION)
_SECTION_ID = re.compile(r"s[1-9][0-9]*")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")

_TOP_KEYS = frozenset({"schema", "schema_version", "fictional", "notice", "sources"})
_SECTION_KEYS = frozenset({"section_id", "heading", "start", "end", "pages"})
_RECORD_KEYS = frozenset(
    {"source_id", "family_id", "version", "title", "tenant", "matter", "jurisdiction"}
    | {"regime", "authority_type", "effective_from", "effective_to", "supersedes"}
    | {"license", "origin", "labels", "content_hash", "text", "sections"}
)
_FAMILY_FIELDS = ("tenant", "matter", "jurisdiction", "regime", "authority_type")

#: The type of the ``"unknown"`` value (the constant itself is :data:`UNKNOWN`).
Unknown = Literal["unknown"]


class ProvenanceError(ValueError):
    """A fixture that breaks the contract; ``problems`` lists what was found."""

    def __init__(self, problems: Sequence[str] | str) -> None:
        self.problems = (problems,) if isinstance(problems, str) else tuple(problems)
        super().__init__(self.problems)  # args round-trip through pickle and copy

    def __str__(self) -> str:
        return "\n".join(self.problems)


@dataclass(frozen=True, slots=True)
class Section:
    """A heading-led span ``text[start:end]`` of its record, with its page range."""

    section_id: str
    heading: str
    start: int
    end: int
    pages: tuple[int, int] | Unknown


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One validated version of one fictional source. See the module docstring."""

    source_id: str
    family_id: str
    version: int
    title: str
    tenant: str
    matter: str | None
    jurisdiction: str
    regime: str
    authority_type: str
    effective_from: date | Unknown
    effective_to: date | Unknown | None
    supersedes: str | None
    license: str
    origin: str
    labels: tuple[str, ...]
    content_hash: str
    text: str
    sections: tuple[Section, ...]


def content_hash(text: str) -> str:
    """``"sha256:<hex>"`` of ``text`` encoded as UTF-8, exactly as stored."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- field checks -------------------------------------------------------------
# Each appends any problem to ``out`` and returns the value; the caller builds a
# record only when no problem was added, so a returned value is trusted only then.


def _show(value: object, limit: int = 60) -> str:
    """``repr`` of a file-controlled value, cut short; repr also escapes control characters."""
    try:
        text = repr(value)
    except (ValueError, RecursionError):  # an int past Python's digit limit, absurd nesting
        text = f"<unprintable {type(value).__name__}>"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_int(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= MAX_INT


def _encodable(text: str) -> bool:
    """False for a lone surrogate, which JSON allows but UTF-8 cannot encode."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _dated(*values: object) -> bool:
    return all(isinstance(value, date) for value in values)


def _keys(raw: object, expected: frozenset[str], where: str, out: list[str]) -> bool:
    """True if ``raw`` is an object with exactly ``expected`` keys; reports what is off."""
    if not isinstance(raw, Mapping):
        out.append(f"{where}: expected an object, got {_show(raw)}")
        return False
    missing = sorted(expected - raw.keys())
    unexpected = sorted(_show(k) for k in raw.keys() - expected)
    if missing:
        out.append(f'{where}: missing {", ".join(missing)} (an unknown value is "unknown")')
    if unexpected:
        more = f" and {len(unexpected) - 5} more" if len(unexpected) > 5 else ""
        out.append(f"{where}: unexpected key(s) {', '.join(unexpected[:5])}{more}")
    return not missing and not unexpected


def _string(
    value: object, where: str, out: list[str], pattern: re.Pattern[str] | None = None,
    *, unknown: bool = False,
) -> Any:
    """A string matching ``pattern``; ``"unknown"`` only if ``unknown`` allows it."""
    if unknown and value == UNKNOWN:
        return value
    hint = 'write exactly "unknown"' if unknown else "this field must be known"
    if not isinstance(value, str):
        null = ' (write "unknown", not null)' if value is None and unknown else ""
        out.append(f"{where}: expected a string, got {_show(value)}{null}")
    elif not _encodable(value):
        out.append(f"{where}: is not valid Unicode (it holds a lone surrogate)")
    elif value.strip().lower() in LOOKALIKES:
        out.append(f"{where}: {_show(value)} is not a real value; {hint}")
    elif pattern is not None and not pattern.fullmatch(value):
        out.append(f"{where}: {_show(value)} does not match {pattern.pattern}")
    elif pattern is not None and len(value) > MAX_ID:
        out.append(f"{where}: {_show(value)} is longer than {MAX_ID} characters")
    elif pattern in _PREFIXED and value.partition("-")[2] in LOOKALIKES:
        out.append(f"{where}: {_show(value)} is not a real value; {hint}")
    elif pattern is None and value != value.strip():
        out.append(f"{where}: {_show(value)} has leading or trailing whitespace")
    return value


def _date(value: object, where: str, out: list[str], *, open_end: bool = False) -> Any:
    """A ``YYYY-MM-DD`` date, ``"unknown"``, or (if ``open_end``) null for "no end"."""
    if value == UNKNOWN or (value is None and open_end):
        return value
    if isinstance(value, str) and _DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    allowed = 'a YYYY-MM-DD date or "unknown"' + (", or null for no end" if open_end else "")
    out.append(f"{where}: {_show(value)} is not {allowed}")
    return None


def _int(value: object, where: str, out: list[str], minimum: int) -> Any:
    """An int (bools refused) from ``minimum`` to :data:`MAX_INT`."""
    if not _is_int(value, minimum):
        out.append(f"{where}: expected an integer from {minimum} to {MAX_INT}, got {_show(value)}")
    return value


def _choice(value: object, allowed: frozenset[str], where: str, out: list[str]) -> None:
    """Report ``value`` unless it is one of the ``allowed`` strings (lists cannot crash this)."""
    if not isinstance(value, str) or value not in allowed:
        out.append(f"{where}: {_show(value)} is not one of {sorted(allowed)}")


def _pages(value: object, where: str, out: list[str]) -> Any:
    """``"unknown"`` or ``[first, last]`` with ``1 <= first <= last``, returned as a tuple."""
    if value == UNKNOWN:
        return value
    ints = isinstance(value, list) and len(value) == 2 and all(_is_int(p, 1) for p in value)
    if ints and value[0] <= value[1]:
        return (value[0], value[1])
    out.append(f'{where}: pages must be [first, last] with 1 <= first <= last, or "unknown"')
    return value


def _labels(value: object, where: str, out: list[str]) -> tuple[str, ...]:
    """A non-empty list of distinct slug labels."""
    if not isinstance(value, list) or not value:
        out.append(f"{where}.labels: expected a non-empty list")
        return ()
    for i, label in enumerate(value):
        _string(label, f"{where}.labels[{i}]", out, _FAMILY)
    strings = [label for label in value if isinstance(label, str)]
    if len(set(strings)) != len(strings):
        out.append(f"{where}.labels: labels must be distinct")
    return tuple(strings)


# --- one record ---------------------------------------------------------------


def _sections(value: object, text: str, where: str, out: list[str]) -> tuple[Section, ...]:
    """Parse the sections and check their spans against ``text``."""
    if not isinstance(value, list) or not value:
        out.append(f"{where}.sections: expected a non-empty list")
        return ()
    sections: list[Section] = []
    seen: set[str] = set()
    prev_end, prev_page = 0, 1
    for i, raw in enumerate(value):
        at = f"{where}.sections[{i}]"
        if not _keys(raw, _SECTION_KEYS, at, out) or not isinstance(raw, Mapping):
            continue
        before = len(out)
        sid = _string(raw["section_id"], f"{at}.section_id", out, _SECTION_ID)
        heading = _string(raw["heading"], f"{at}.heading", out)
        start = _int(raw["start"], f"{at}.start", out, 0)
        end = _int(raw["end"], f"{at}.end", out, 1)
        pages = _pages(raw["pages"], f"{at}.pages", out)
        if len(out) > before:
            continue
        if sid in seen:
            out.append(f"{at}: duplicate section_id {_show(sid)}")
        seen.add(sid)
        span_ok = prev_end <= start < end <= len(text)
        if not span_ok:
            out.append(
                f"{at}: span [{start}, {end}) must be non-empty, inside the text "
                f"(length {len(text)}) and start at or after the previous end ({prev_end})"
            )
        elif start > 0 and text[start - 1] != "\n":
            out.append(f"{at}: span must start at the beginning of a line")
        elif text[start:end].split("\n", 1)[0] != heading:
            out.append(f"{at}: the span's first line is not the heading {_show(heading)}")
        if pages != UNKNOWN:
            if pages[0] < prev_page:
                out.append(f"{at}: pages {list(pages)} start before the previous section's")
            prev_page = pages[1]
        if span_ok:  # a bad span is reported once, not again against every later section
            prev_end = end
        sections.append(Section(sid, heading, start, end, pages))
    return tuple(sections)


def _text(fields: Mapping[str, Any], where: str, out: list[str]) -> str | None:
    """The record's text, checked for labelling, encoding, normal form and hash."""
    text = fields["text"]
    if not isinstance(text, str) or not text.strip():
        out.append(f"{where}.text: expected non-empty text")
        return None
    if not _encodable(text):
        out.append(f"{where}.text: is not valid Unicode (it holds a lone surrogate)")
        return None
    expected = content_hash(text)
    if not text.startswith(NOTICE + "\n"):
        out.append(f"{where}.text: must open with the line {NOTICE!r}")
    if not unicodedata.is_normalized("NFC", text):
        out.append(f"{where}.text: is not NFC-normalised, so its hash would be ambiguous")
    claimed = fields["content_hash"]
    if not isinstance(claimed, str) or not _HASH.fullmatch(claimed):
        out.append(f"{where}.content_hash: expected 'sha256:' and 64 lowercase hex digits")
    elif claimed != expected:
        out.append(f"{where}.content_hash: does not match the text (the text hashes to {expected})")
    return text


def _scope(fields: Mapping[str, Any], where: str, out: list[str]) -> None:
    """Tenant and matter, which must be known, and the public/matter rule."""
    tenant, matter = fields["tenant"], fields["matter"]
    if tenant != PUBLIC:
        _string(tenant, f"{where}.tenant", out, _TENANT)
    if matter is not None:
        _string(matter, f"{where}.matter", out, _MATTER)
        if tenant == PUBLIC:
            out.append(f"{where}: a public source cannot belong to a matter")


def _parse_record(raw: object, where: str, out: list[str]) -> SourceRecord | None:
    """One record, or None after appending its problems to ``out``."""
    if not _keys(raw, _RECORD_KEYS, where, out) or not isinstance(raw, Mapping):
        return None
    before = len(out)
    fields = {key: raw[key] for key in _RECORD_KEYS}
    family = _string(fields["family_id"], f"{where}.family_id", out, _FAMILY)
    version = _int(fields["version"], f"{where}.version", out, 1)
    if len(out) == before and fields["source_id"] != f"{family}@v{version}":
        out.append(f"{where}: source_id is not '<family_id>@v<version>' ({family}@v{version})")
    title = _string(fields["title"], f"{where}.title", out)
    if isinstance(title, str) and not title.endswith("(fictional)"):
        out.append(f"{where}.title: must end with '(fictional)'")
    _scope(fields, where, out)
    _string(fields["jurisdiction"], f"{where}.jurisdiction", out, _JURISDICTION, unknown=True)
    _string(fields["regime"], f"{where}.regime", out, _FAMILY, unknown=True)
    _choice(fields["authority_type"], AUTHORITY_TYPES | {UNKNOWN}, f"{where}.authority_type", out)
    start = _date(fields["effective_from"], f"{where}.effective_from", out)
    end = _date(fields["effective_to"], f"{where}.effective_to", out, open_end=True)
    if _dated(start, end) and not start < end:
        out.append(f"{where}: effective_from {start} must be before effective_to {end}")
    if fields["supersedes"] is not None and not isinstance(fields["supersedes"], str):
        out.append(f"{where}.supersedes: expected a source_id or null")
    _choice(fields["license"], LICENSES, f"{where}.license", out)
    _choice(fields["origin"], ORIGINS, f"{where}.origin", out)
    labels = _labels(fields["labels"], where, out)
    text = _text(fields, where, out)
    sections = _sections(fields["sections"], text, where, out) if text is not None else ()
    if len(out) > before or text is None:
        return None
    return SourceRecord(
        fields["source_id"], family, version, title, fields["tenant"], fields["matter"],
        fields["jurisdiction"], fields["regime"], fields["authority_type"], start, end,
        fields["supersedes"], fields["license"], fields["origin"], labels,
        fields["content_hash"], text, sections,
    )


# --- rules across records -----------------------------------------------------


def _check_family(chain: Sequence[SourceRecord], rejected: set[str], out: list[str]) -> None:
    """One family's versions, in version order: shared metadata, one chain, no overlap."""
    first = chain[0]
    for r in chain[1:]:
        for name in _FAMILY_FIELDS:
            if getattr(r, name) != getattr(first, name):
                out.append(f"{r.source_id}: {name} differs from {first.source_id}; "
                           "every version of a family must share it")
    for old, new in pairwise(chain):
        if new.supersedes in rejected:  # its real predecessor is invalid and already reported
            continue
        if new.supersedes != old.source_id:
            out.append(f"{new.source_id}: must supersede the version before it, {old.source_id}")
        if old.effective_to is None:
            out.append(f"{new.source_id}: follows {old.source_id}, whose effective_to is "
                       'null (no end); give its end date or "unknown"')
        elif _dated(old.effective_to, new.effective_from) and \
                old.effective_to != new.effective_from:
            out.append(f"{new.source_id}: starts {new.effective_from} but {old.source_id} ends "
                       f"{old.effective_to}; versions must hand over on one day")
    for i, old in enumerate(chain):
        for j, new in enumerate(chain[i + 1 :], start=i + 1):
            if _dated(old.effective_from, new.effective_from) and \
                    new.effective_from <= old.effective_from:
                out.append(f"{new.source_id}: starts {new.effective_from}, not after "
                           f"{old.source_id} ({old.effective_from})")
            elif j > i + 1 and _dated(old.effective_to, new.effective_from) and \
                    new.effective_from < old.effective_to:
                out.append(f"{new.source_id}: starts {new.effective_from}, before "
                           f"{old.source_id} ends ({old.effective_to})")
            if _dated(old.effective_from, new.effective_to) and \
                    new.effective_to <= old.effective_from:
                out.append(f"{new.source_id}: ends {new.effective_to}, before "
                           f"{old.source_id} starts ({old.effective_from})")


def _check_corpus(records: Sequence[SourceRecord], rejected: set[str], out: list[str]) -> None:
    """Rules that span records: unique IDs, one tenant per matter, the version chains.

    ``rejected`` holds the IDs of records that already failed, so a successor
    is not also blamed for pointing at a record that is present but invalid.
    """
    by_id: dict[str, SourceRecord] = {}
    for r in records:
        if r.source_id in by_id:
            out.append(f"{r.source_id}: duplicate source_id")
        by_id[r.source_id] = r
    matter_tenant: dict[str, str] = {}
    for r in by_id.values():
        if r.matter is not None and matter_tenant.setdefault(r.matter, r.tenant) != r.tenant:
            out.append(f"{r.source_id}: matter {_show(r.matter)} also belongs to another tenant")
        old = by_id.get(r.supersedes) if r.supersedes is not None else None
        if r.supersedes is not None and r.supersedes not in rejected and (
            old is None or old.family_id != r.family_id or old.version >= r.version
        ):
            out.append(f"{r.source_id}: supersedes {_show(r.supersedes)}, which is not a lower "
                       "version of the same family in this fixture")
    families: dict[str, list[SourceRecord]] = {}
    for r in sorted(by_id.values(), key=lambda r: r.version):
        families.setdefault(r.family_id, []).append(r)
    for chain in families.values():
        _check_family(chain, rejected, out)


def validate_corpus(raw: object) -> tuple[SourceRecord, ...]:
    """Every record in a parsed fixture, or :class:`ProvenanceError` with the problems found."""
    out: list[str] = []
    if not _keys(raw, _TOP_KEYS, "fixture", out) or not isinstance(raw, Mapping):
        raise ProvenanceError(out)
    version = raw["schema_version"]
    if raw["schema"] != SCHEMA or type(version) is not int or version != SCHEMA_VERSION:
        out.append(f"fixture: expected schema {SCHEMA!r} version {SCHEMA_VERSION}, got "
                   f"{_show(raw['schema'])} version {_show(version)}")
    if raw["fictional"] is not True:
        out.append("fixture: 'fictional' must be true; this example only loads fictional sources")
    notice = _string(raw["notice"], "fixture.notice", out)
    if isinstance(notice, str) and "fictional" not in notice.lower():
        out.append("fixture.notice: must say the sources are fictional")
    sources = raw["sources"]
    if not isinstance(sources, list) or not sources:
        out.append("fixture.sources: expected a non-empty list")
        raise ProvenanceError(out)
    records: list[SourceRecord] = []
    rejected: set[str] = set()
    for i, item in enumerate(sources):
        label = item.get("source_id") if isinstance(item, Mapping) else None
        where = f"sources[{i}]" + (f" {_show(label)}" if isinstance(label, str) else "")
        record = _parse_record(item, where, out)
        if record is not None:
            records.append(record)
        elif isinstance(label, str):
            rejected.add(label)
    _check_corpus(records, rejected, out)
    if out:
        raise ProvenanceError(out)
    return tuple(records)


# --- files and the command line -----------------------------------------------


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json`` keeps the last of two equal keys; a second ``tenant`` must not hide the first."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key, _ in pairs:
        (duplicates if key in seen else seen).add(key)
    if duplicates:
        names = ", ".join(map(_show, sorted(duplicates)))
        raise ProvenanceError([f"duplicate JSON key(s) {names}"])
    return dict(pairs)


def load(path: str | Path = FIXTURE) -> tuple[SourceRecord, ...]:
    """Read and validate a fixture file.

    Raises :class:`ProvenanceError` for anything wrong with its content, and
    ``OSError`` or ``UnicodeDecodeError`` if the file cannot be read as UTF-8.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except ProvenanceError:
        raise
    except (ValueError, RecursionError) as exc:  # bad JSON, over-long ints, deep nesting
        raise ProvenanceError([f"not valid JSON: {str(exc)[:200]}"]) from exc
    return validate_corpus(raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the bundled fixture, or the file given; exit 1 and list problems if invalid."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1 or args[:1] in (["-h"], ["--help"]):
        print("usage: python -m examples.legal_rag.provenance [fixture.json]", file=sys.stderr)
        return 2
    path = Path(args[0]) if args else FIXTURE
    try:
        records = load(path)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"{path}: cannot read: {exc}", file=sys.stderr)
        return 1
    except ProvenanceError as exc:
        print(f"{path}: {len(exc.problems)} problem(s)", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    families = len({r.family_id for r in records})
    scopes = len({(r.tenant, r.matter) for r in records})
    print(f"{len(records)} valid records: {families} families, {scopes} tenant/matter scopes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
