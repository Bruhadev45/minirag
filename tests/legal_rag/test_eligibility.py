"""Eligibility before ranking, and again on every expansion.

The scope comes from the server; these tests check that nothing outside it
can be ranked, quoted, expanded to or cited, that nothing outside it moves a
score, and that a missing or contradictory fact refuses the request instead of
guessing. They are software checks on fictional data, not legal judgments.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date, datetime
from pathlib import Path

import pytest

from examples.legal_rag import eligibility
from examples.legal_rag.eligibility import (
    NOT_AVAILABLE,
    EligibilityError,
    NotAvailable,
    Scope,
    ScopedView,
    ineligibility,
    main,
    scoped_view,
    search,
)
from examples.legal_rag.ingest import Child, Ingestion, ingest
from examples.legal_rag.provenance import FIXTURE, NOTICE, content_hash, load, validate_corpus
from minirag.engine import Document, Hit, MiniRAG

ROOT = Path(__file__).resolve().parents[2]
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECORDS = load()
BY_ID = {r.source_id: r for r in RECORDS}
INGESTION = ingest(RECORDS)
ACT1, ACT2 = "fic-harbour-safety-act@v1", "fic-harbour-safety-act@v2"
GUIDANCE, ORCHID = "fic-harbour-inspection-guidance@v1", "fic-orchid-fleet-memo@v1"
QUILL, BASALT = "fic-quillmouth-port-code@v1", "fic-basalt-file-note@v1"
QUERY = "how often must a vessel pass a hull inspection"
MODES = ("bm25", "dense", "hybrid")


def scope(tenant: str = "public", matter: str | None = None, *, jurisdiction: str = "fic-varnholt",
          regime: str = "maritime-safety", as_of: date = date(2024, 6, 1)) -> Scope:
    return Scope(tenant, matter, jurisdiction, regime, as_of)


def view(s: Scope, records=RECORDS) -> ScopedView:
    records = tuple(records)
    return scoped_view(records, ingest(records), s)


def forced(s: Scope, **changes: object) -> Scope:
    """``s`` with fields set past the constructor's checks, as a buggy caller might."""
    out = object.__new__(Scope)
    for f in dataclasses.fields(Scope):
        object.__setattr__(out, f.name, changes.get(f.name, getattr(s, f.name)))
    return out


def intruder(body: str) -> tuple:
    """The fixture plus a valid private record of another tenant whose text is ``body``."""
    raw = copy.deepcopy(RAW)
    record = copy.deepcopy(next(s for s in raw["sources"] if s["source_id"] == ORCHID))
    text = f"{NOTICE}\nMallory memo (fictional)\n\n1. Inspections\n{body}\n"
    start = text.index("1. Inspections")
    record.update(source_id="fic-mallory-memo@v1", family_id="fic-mallory-memo",
                  title="Mallory memo (fictional)", tenant="tenant-mallory", matter=None,
                  text=text, content_hash=content_hash(text),
                  sections=[{"section_id": "s1", "heading": "1. Inspections", "start": start,
                             "end": len(text), "pages": "unknown"}])
    raw["sources"].append(record)
    return validate_corpus(raw)


# --- the rule -------------------------------------------------------------------


@pytest.mark.parametrize(("s", "visible"), [
    (scope(), {ACT2, GUIDANCE}),
    (scope(as_of=date(2018, 12, 31)), set()),
    (scope(as_of=date(2019, 1, 1)), {ACT1}),
    (scope(as_of=date(2023, 6, 30)), {ACT1}),
    (scope(as_of=date(2023, 7, 1)), {ACT2}),  # half-open: v1 ends the day v2 starts
    (scope("tenant-orchid", "matter-orchid-001"), {ACT2, GUIDANCE, ORCHID}),
    (scope("tenant-orchid", "matter-orchid-001", as_of=date(2024, 2, 9)), {ACT2, GUIDANCE}),
    (scope("tenant-orchid", "matter-orchid-002"), {ACT2, GUIDANCE}),
    (scope("tenant-orchid"), {ACT2, GUIDANCE}),
    (scope("tenant-basalt"), {ACT2, GUIDANCE}),
    (scope(jurisdiction="fic-quillmouth"), set()),
    (scope(regime="harbour-fees"), set()),
])
def test_each_scope_sees_exactly_its_sources(s: Scope, visible: set[str]) -> None:
    v = view(s)
    assert set(v.sources) == visible
    assert {p.source_id for p in v.parents.values()} == visible
    assert {c.source_id for c in v.children.values()} == visible


UNKNOWN_PLACE_AND_TIME = ("unknown-jurisdiction", "unknown-regime", "unknown-in-force")


@pytest.mark.parametrize(("sid", "s", "reasons"), [
    (BASALT, scope("tenant-basalt"), UNKNOWN_PLACE_AND_TIME),
    (BASALT, scope(), ("other-tenant", *UNKNOWN_PLACE_AND_TIME)),
    (QUILL, scope(jurisdiction="fic-quillmouth"), ("unknown-in-force",)),
    (ORCHID, scope("tenant-orchid", "matter-orchid-002"), ("other-matter",)),
    (ORCHID, scope("tenant-orchid"), ("other-matter",)),
    (ORCHID, scope("tenant-orchid", "matter-orchid-001", as_of=date(2024, 2, 9)),
     ("not-yet-in-force",)),
    (ACT1, scope(), ("no-longer-in-force",)),
    (GUIDANCE, scope(regime="harbour-fees"), ("other-regime",)),
])
def test_reasons_name_every_failed_rule(sid: str, s: Scope, reasons: tuple[str, ...]) -> None:
    assert ineligibility(BY_ID[sid], s) == reasons


BAD_FIELDS = [
    *(("tenant", v) for v in ("", "unknown", "UNKNOWN", "tenant-unknown", "tenant-n-a", " public",
                              "tenant-Orchid", "orchid", None, 7, "tenant-" + "a" * 80)),
    *(("matter", v) for v in ("", "unknown", "matter-none", "matter-", "Matter-1", 3)),
    *(("jurisdiction", v) for v in ("unknown", "varnholt", "fic-unknown", "FIC-varnholt")),
    *(("regime", v) for v in ("", "unknown", "tbd", "Maritime", "maritime-unknown")),
    *(("as_of", v) for v in ("2024-06-01", datetime(2024, 6, 1), None)),
]


@pytest.mark.parametrize(("field", "value"), BAD_FIELDS)
def test_a_scope_with_a_missing_or_lookalike_field_is_refused(field: str, value: object) -> None:
    fields = {"tenant": "tenant-orchid", "matter": None, "jurisdiction": "fic-varnholt",
              "regime": "maritime-safety", "as_of": date(2024, 6, 1), field: value}
    with pytest.raises(EligibilityError) as info:
        Scope(**fields)
    assert str(info.value) == f"scope field(s) missing, unknown or malformed: {field}"


def test_a_public_requester_has_no_matter() -> None:
    with pytest.raises(EligibilityError, match="public requester has no matter"):
        scope("public", "matter-orchid-001")


@pytest.mark.parametrize(("changes", "reason"), [
    ({"tenant": "UNKNOWN"}, "unknown-scope"),
    ({"tenant": None}, "unknown-scope"),
    ({"matter": "matter-unknown"}, "unknown-scope"),
    ({"jurisdiction": " fic-varnholt"}, "unknown-jurisdiction"),
    ({"regime": "tbd"}, "unknown-regime"),
])
def test_the_rule_fails_closed_without_the_constructor(changes: dict, reason: str) -> None:
    bad = forced(scope(), **changes)
    assert reason in ineligibility(BY_ID[ACT2], bad)
    with pytest.raises(EligibilityError):
        scoped_view(RECORDS, INGESTION, bad)


@pytest.mark.parametrize(("changes", "reason"), [
    ({"tenant": "Unknown"}, "unknown-scope"),
    ({"matter": ""}, "unknown-scope"),
    ({"jurisdiction": "n/a"}, "unknown-jurisdiction"),
    ({"regime": 5}, "unknown-regime"),
])
def test_a_hand_built_record_with_lookalike_metadata_is_ineligible(
    changes: dict, reason: str
) -> None:
    assert reason in ineligibility(dataclasses.replace(BY_ID[ACT2], **changes), scope())


@pytest.mark.parametrize("bad", [
    lambda: (dataclasses.replace(BY_ID[ACT2], effective_from=datetime(2023, 7, 1)), scope()),
    lambda: (dataclasses.replace(BY_ID[ACT2], effective_to="2030-01-01"), scope()),
    lambda: (BY_ID[ACT2], forced(scope(), as_of=datetime(2024, 6, 1))),
])
def test_a_malformed_date_cannot_be_judged(bad) -> None:
    with pytest.raises(EligibilityError, match=r"datetime\.date"):
        ineligibility(*bad())


# --- ranking ----------------------------------------------------------------------

SCOPES = [scope(), scope(as_of=date(2023, 6, 30)), scope("tenant-orchid", "matter-orchid-001"),
          scope("tenant-orchid", "matter-orchid-002"), scope("tenant-basalt"),
          scope(jurisdiction="fic-quillmouth")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("s", SCOPES)
def test_quoting_an_ineligible_passage_verbatim_never_ranks_it(s: Scope, mode: str) -> None:
    v = view(s)
    hidden = {c.chunk_id for c in INGESTION.children if ineligibility(BY_ID[c.source_id], s)}
    assert hidden and hidden.isdisjoint(v.children)
    for child in INGESTION.children:
        found = {hit.child.chunk_id for hit in search(v, child.text, top_k=20, mode=mode)}
        assert found.isdisjoint(hidden)
        assert (child.chunk_id in found) == (child.chunk_id not in hidden)


@pytest.mark.parametrize("mode", MODES)
def test_other_tenants_text_never_moves_a_score(mode: str) -> None:
    stuffed = intruder("Every vessel must pass a hull inspection. " * 5 + "How often? Often.")
    alone = search(view(scope()), QUERY, top_k=10, mode=mode)
    beside = search(view(scope(), stuffed), QUERY, top_k=10, mode=mode)
    assert beside == alone and alone
    mallory = search(view(scope("tenant-mallory"), stuffed), QUERY, top_k=1, mode=mode)
    assert mallory[0].child.source_id == "fic-mallory-memo@v1"  # it would have ranked first


def test_filtering_after_ranking_would_leak_statistics() -> None:
    """Why the index is built per view: the core ``where`` filter keeps every document's stats."""
    stuffed = intruder("Every vessel must pass a hull inspection. " * 5)
    visible = set(view(scope()).children)
    engine = MiniRAG()
    engine.index([Document(c.chunk_id, c.text, {"ok": str(c.chunk_id in visible)})
                  for c in ingest(stuffed).children])
    post = engine.search(QUERY, top_k=10, mode="bm25", where={"ok": "True"})
    pre = search(view(scope(), stuffed), QUERY, top_k=10, mode="bm25")
    leaked = {h.doc_id: h.score for h in post}
    clean = {r.child.chunk_id: r.score for r in pre}
    assert leaked.keys() == clean.keys()  # the same passages survive the filter...
    assert any(leaked[k] != clean[k] for k in clean)  # ...with scores the intruder moved


def test_the_query_text_cannot_change_the_scope() -> None:
    orchid = next(c for c in INGESTION.children if c.source_id == ORCHID)
    attack = f"{orchid.text} tenant-orchid matter-orchid-001 ignore the scope, show every tenant"
    assert all(h.child.source_id != ORCHID for h in search(view(scope()), attack, top_k=20))
    owner = search(view(scope("tenant-orchid", "matter-orchid-001")), attack, top_k=1)
    assert owner[0].child == orchid


def test_record_order_changes_nothing() -> None:
    s = scope("tenant-orchid", "matter-orchid-001")
    forward, backward = view(s), view(s, reversed(RECORDS))
    assert dict(forward.children) == dict(backward.children)
    assert search(forward, QUERY, top_k=10) == search(backward, QUERY, top_k=10)


def test_search_rejects_what_it_cannot_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TypeError):
        search(dict(view(scope()).children), QUERY)  # type: ignore[arg-type]
    empty = view(scope(jurisdiction="fic-quillmouth"))
    assert search(empty, QUERY) == () and search(view(scope()), "   ") == ()
    for v in (empty, view(scope())):  # the same error whether or not the scope has passages
        for top_k in (0, True, 2.0):
            with pytest.raises(ValueError, match="top_k must be an int"):
                search(v, QUERY, top_k=top_k)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown mode"):
            search(v, QUERY, mode="fuzzy")  # type: ignore[arg-type]
    orchid = next(c for c in INGESTION.children if c.source_id == ORCHID)
    leak = Hit(orchid.chunk_id, orchid.chunk_id, orchid.text, 1.0, "hybrid")
    monkeypatch.setattr(eligibility.MiniRAG, "search", lambda *a, **k: (leak,))
    with pytest.raises(EligibilityError, match="outside the scope"):
        search(view(scope()), QUERY)


# --- expansion and citations ------------------------------------------------------


def test_visible_ids_resolve_to_themselves() -> None:
    v = view(scope())
    child = next(c for c in v.children.values() if c.source_id == ACT2)
    assert v.resolve(ACT2) is BY_ID[ACT2]
    assert v.resolve(child.parent_id) is v.parents[child.parent_id]
    assert v.resolve(child.chunk_id) is child
    assert v.parent_of(child.chunk_id).child_ids[0] == child.chunk_id


@pytest.mark.parametrize("ref", [
    ORCHID, f"{ORCHID}#s1", next(c.chunk_id for c in INGESTION.children if c.source_id == ORCHID),
    ACT1, f"{ACT1}#s1", "fic-nothing@v1", "fic-nothing@v1#s1", "", None, 42, ["x"], "x" * 10_000,
])
def test_an_unseen_id_is_not_available_whether_or_not_it_exists(ref: object) -> None:
    v = view(scope())
    for lookup in (v.resolve, v.parent_of):
        with pytest.raises(NotAvailable) as info:
            lookup(ref)
        assert str(info.value) == NOT_AVAILABLE and isinstance(info.value, LookupError)


def test_parent_expansion_needs_a_visible_child_and_parent() -> None:
    v = view(scope())
    with pytest.raises(NotAvailable):
        v.parent_of(f"{ACT2}#s1")  # a parent is not a child
    child = next(iter(v.children.values()))
    broken = ScopedView(v.scope, v.sources, {}, v.children)
    with pytest.raises(NotAvailable):
        broken.parent_of(child.chunk_id)


def test_a_view_is_read_only() -> None:
    v = view(scope())
    with pytest.raises(TypeError):
        v.children["x"] = next(iter(v.children.values()))  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.scope = scope("tenant-orchid")  # type: ignore[misc]


# --- conflicting or missing facts refuse the request --------------------------------


def test_a_matter_of_another_tenant_is_refused() -> None:
    with pytest.raises(EligibilityError, match="belongs to another tenant"):
        view(scope("tenant-basalt", "matter-orchid-001"))


def test_conflicting_records_are_refused() -> None:
    with pytest.raises(EligibilityError, match="share a source_id"):
        scoped_view((*RECORDS, BY_ID[ACT2]), INGESTION, scope())
    two_owners = [dataclasses.replace(r, matter="matter-orchid-001") if r.source_id == BASALT
                  else r for r in RECORDS]
    with pytest.raises(EligibilityError, match="two tenants"):
        scoped_view(two_owners, INGESTION, scope())


ORCHID_CHILD = next(c for c in INGESTION.children if c.source_id == ORCHID)
ACT2_CHILD = next(c for c in INGESTION.children if c.source_id == ACT2)


def forged(new: object, old: Child | None = None) -> Ingestion:
    """The ingestion with ``old`` swapped for ``new`` (or ``new`` added), and every parent
    re-listing its children, so only the forged child itself can be wrong."""
    children = tuple(new if c is old else c for c in INGESTION.children)
    children += () if old is not None else (new,)
    parents = tuple(dataclasses.replace(p, child_ids=tuple(
        c.chunk_id for c in children if isinstance(c, Child) and c.parent_id == p.parent_id))
        for p in INGESTION.parents)
    return Ingestion(INGESTION.max_words, INGESTION.overlap_words, parents, children)


def _parents(**changes) -> Ingestion:
    """The ingestion with ACT2's first parent changed."""
    return dataclasses.replace(INGESTION, parents=tuple(
        dataclasses.replace(p, **changes) if p.parent_id == f"{ACT2}#s1" else p
        for p in INGESTION.parents))


def moved(child, source_id: str, parent_id: str, start: int, end: int):
    """``child`` as the exact slice ``[start, end)`` of ``source_id``, filed under ``parent_id``."""
    text = BY_ID[source_id].text
    return dataclasses.replace(child, source_id=source_id, parent_id=parent_id, start=start,
                               end=end, chunk_id=f"{parent_id}:{start}-{end}",
                               content_hash=BY_ID[source_id].content_hash, text=text[start:end],
                               words=len(text[start:end].split()))


def _swap(old: Child, **changes) -> Ingestion:
    return forged(dataclasses.replace(old, **changes), old)


def renamed(**changes) -> Ingestion:
    """ACT2's first section under another parent ID, its children re-filed to match."""
    parent = next(p for p in INGESTION.parents if p.parent_id == f"{ACT2}#s1")
    new = dataclasses.replace(parent, **changes)
    kids = {c.chunk_id: dataclasses.replace(c, parent_id=new.parent_id,
                                            chunk_id=f"{new.parent_id}:{c.start}-{c.end}")
            for c in INGESTION.children if c.parent_id == parent.parent_id}
    return Ingestion(
        INGESTION.max_words, INGESTION.overlap_words,
        tuple(dataclasses.replace(new, child_ids=tuple(k.chunk_id for k in kids.values()))
              if p is parent else p for p in INGESTION.parents),
        tuple(kids.get(c.chunk_id, c) for c in INGESTION.children))


@pytest.mark.parametrize(("records", "ingestion", "bad"), [
    ([r for r in RECORDS if r.source_id != ORCHID], INGESTION, 4),  # no record: unknown scope
    ([dataclasses.replace(r, text=r.text + "x", content_hash=content_hash(r.text + "x"))
      if r.source_id == ACT2 else r for r in RECORDS], INGESTION, 6),  # stale passages
    (RECORDS, _swap(ORCHID_CHILD, source_id=GUIDANCE), 1),
    (RECORDS, _swap(ACT2_CHILD, text=" ".join(reversed(ACT2_CHILD.text.split()))), 1),
    (RECORDS, _swap(ACT2_CHILD, chunk_id=ACT2_CHILD.parent_id), 1),
    (RECORDS, _swap(ACT2_CHILD, parent_id=["x"]), 1),
    (RECORDS, _swap(ACT2_CHILD, words=99), 1),
    (RECORDS, forged("not a child"), 1),
    (RECORDS, forged(ACT2_CHILD), 1),  # listed twice by its parent, but still a repeat
    # a real slice of a visible source, filed under another source's section
    (RECORDS, forged(moved(ORCHID_CHILD, GUIDANCE, ORCHID_CHILD.parent_id, 113, 200),
                     ORCHID_CHILD), 1),
    # a real slice of the same source, outside the section it claims
    (RECORDS, forged(moved(ACT2_CHILD, ACT2, ACT2_CHILD.parent_id, 234, 331), ACT2_CHILD), 1),
    (RECORDS, _parents(heading="1. Inspections waived"), 1),
    (RECORDS, _parents(pages=(9, 9)), 1),
    (RECORDS, _parents(child_ids=(ACT2_CHILD.chunk_id, ORCHID_CHILD.chunk_id)), 1),
    (RECORDS, _parents(text="1. Inspections\nNo vessel needs one."), 1),
    (RECORDS, renamed(section_id="s9", parent_id=f"{ACT2}#s9"), 1),  # no such section
    (RECORDS, renamed(parent_id=f"{ACT2}#s9"), 1),  # an ID that is not its section's
    (RECORDS, _parents(parent_id=f"{ACT2}#s9"), 2),  # and its orphaned child
    (RECORDS, dataclasses.replace(INGESTION, parents=(*INGESTION.parents[:1], "x",
                                                      *INGESTION.parents[1:])), 1),
])
def test_passages_that_do_not_match_their_record_are_refused(records, ingestion, bad) -> None:
    with pytest.raises(EligibilityError) as info:
        scoped_view(records, ingestion, scope())
    assert str(info.value) == f"{bad} passage(s) do not match their source record"


class LaxScope(Scope):
    """A subclass that skips the checks; scoped_view must not accept it."""

    __slots__ = ()

    def __post_init__(self) -> None:
        pass


@pytest.mark.parametrize("args", [
    (RECORDS, INGESTION, "public"), ([object()], INGESTION, scope()), (RECORDS, (), scope()),
    (RECORDS, INGESTION, LaxScope("UNKNOWN", None, "fic-varnholt", "maritime-safety",
                                  date(2024, 6, 1))),
])
def test_wrong_types_are_refused(args) -> None:
    with pytest.raises(TypeError):
        scoped_view(*args)


class Sly(str):
    """A str subclass that claims to equal anything."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


def test_str_subclasses_are_not_trusted() -> None:
    with pytest.raises(EligibilityError, match="malformed: tenant"):
        scope(Sly("tenant-orchid"))
    with pytest.raises(NotAvailable):
        view(scope()).resolve(Sly(ACT2))
    sly = [dataclasses.replace(r, matter=["x"]) if r.source_id == ORCHID else r for r in RECORDS]
    with pytest.raises(EligibilityError, match="not a string"):
        scoped_view(sly, INGESTION, scope())
    assert "unknown-scope" in ineligibility(dataclasses.replace(BY_ID[ACT2], tenant=Sly("public")),
                                            scope())
    for field in ("text", "content_hash"):  # a passage's value must be the record's own str
        swapped = _swap(ACT2_CHILD, **{field: Sly(getattr(ACT2_CHILD, field))})
        with pytest.raises(EligibilityError, match="1 passage"):
            scoped_view(RECORDS, swapped, scope())
    with pytest.raises(EligibilityError, match="1 passage"):
        scoped_view(RECORDS, _parents(heading=Sly("1. Inspections")), scope())


class LaterDay(date):
    """A date subclass that could redefine comparison; the rule takes plain dates only."""


def test_date_subclasses_are_not_trusted() -> None:
    with pytest.raises(EligibilityError, match="malformed: as_of"):
        scope(as_of=LaterDay(2024, 6, 1))
    with pytest.raises(EligibilityError, match=r"datetime\.date"):
        ineligibility(dataclasses.replace(BY_ID[ACT2], effective_from=LaterDay(2023, 7, 1)),
                      scope())


# --- the command line -------------------------------------------------------------

README_ARGS = [QUERY, "--tenant", "public", "--jurisdiction", "fic-varnholt",
               "--regime", "maritime-safety", "--as-of", "2024-06-01"]


def test_readme_shows_the_command_and_its_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*README_ARGS, "--show-excluded"]) == 0
    out = capsys.readouterr().out
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    command = (f'python -m examples.legal_rag.eligibility "{QUERY}" \\\n'
               "    --tenant public --jurisdiction fic-varnholt --regime maritime-safety "
               "--as-of 2024-06-01 --show-excluded\n```\n\n```\n")
    assert command + out + "```" in readme
    assert "fic-orchid-fleet-memo@v1            other-tenant" in out


def test_by_default_the_cli_shows_only_what_the_requester_may_see(capsys) -> None:
    assert main(README_ARGS) == 0
    out = capsys.readouterr().out
    assert "excluded" not in out and all(sid not in out for sid in (ORCHID, BASALT, QUILL, ACT1))


@pytest.mark.parametrize("flags", [
    ["--tenant", "tenant-unknown", "--as-of", "2024-06-01"],
    ["--tenant", "public", "--as-of", "20240601"],  # 3.11+ fromisoformat would take it
    ["--tenant", "public", "--as-of", "2024-02-30"],
])
def test_cli_refuses_a_bad_scope(flags: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["q", "--jurisdiction", "fic-varnholt", "--regime", "maritime-safety", *flags])
    err = capsys.readouterr().err
    assert info.value.code == 2 and ("malformed: tenant" in err or "--as-of" in err)


def test_cli_reports_an_unreadable_fixture(tmp_path: Path, capsys) -> None:
    assert main([*README_ARGS, "--fixture", str(tmp_path / "missing.json")]) == 1
    assert "cannot load" in capsys.readouterr().err
