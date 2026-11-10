"""Bounded graph-assisted retrieval over explicit edges.

Every section an expansion returns must be one the scope may see, reached only
along an edge written in the fixtures, and each at most once. The edges these
tests add to exercise cycles and caps are test-only: they quote real words of
the fixture but are not references the fictional texts make.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import pytest

from examples.legal_rag import graph as graph_module
from examples.legal_rag.eligibility import (
    NOT_AVAILABLE,
    EligibilityError,
    NotAvailable,
    Scope,
    ScopedView,
    ineligibility,
    scoped_view,
)
from examples.legal_rag.graph import (
    MAX_CANDIDATES,
    MAX_HOPS,
    MAX_REFERENCES,
    MAX_SEEDS,
    REFERENCES,
    Expansion,
    Graph,
    Hop,
    Reached,
    Reference,
    build_graph,
    expand,
    load_graph,
    main,
)
from examples.legal_rag.ingest import ingest
from examples.legal_rag.provenance import FIXTURE, ProvenanceError, load, validate_corpus

ROOT = Path(__file__).resolve().parents[2]
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
RAW_REFS = json.loads(REFERENCES.read_text(encoding="utf-8"))
RECORDS = load()
GRAPH = load_graph(RECORDS)
ACT, ACT1, ACT2 = "fic-harbour-safety-act", "fic-harbour-safety-act@v1", "fic-harbour-safety-act@v2"
GUIDANCE, ORCHID = "fic-harbour-inspection-guidance@v1", "fic-orchid-fleet-memo@v1"
QUILL = "fic-quillmouth-port-code@v1"


def scope(day: date = date(2024, 6, 1), place: str = "fic-varnholt", tenant: str = "public",
          matter: str | None = None) -> Scope:
    return Scope(tenant, matter, place, "maritime-safety", day)


def view(s: Scope | None = None, records=RECORDS) -> ScopedView:
    records = tuple(records)
    return scoped_view(records, ingest(records), s or scope())


def seed(v: ScopedView, parent_id: str) -> str:
    """The first child passage of a section."""
    return v.parents[parent_id].child_ids[0]


def ids(result: Expansion) -> list[tuple[int, str]]:
    return [(item.hops, item.parent.parent_id) for item in result.sections]


def ref(source_section: str, quote: str, to: str, records=RECORDS) -> dict[str, object]:
    """A references-file entry quoting the first occurrence of ``quote`` in a section."""
    source_id, _, section_id = source_section.partition("#")
    record = next(r for r in records if r.source_id == source_id)
    section = next(s for s in record.sections if s.section_id == section_id)
    start = record.text.index(quote, section.start)
    return {"from": source_section, "start": start, "end": start + len(quote), "quote": quote,
            "to": to}


def refs_file(*entries: dict[str, object]) -> dict[str, object]:
    return {**RAW_REFS, "references": list(entries)}


# --- loading ----------------------------------------------------------------------


def test_the_bundled_references_are_the_fixtures_explicit_ones() -> None:
    guidance, quill = GRAPH.references
    assert (guidance.from_id, guidance.quote, guidance.family_id, guidance.section_id) == (
        f"{GUIDANCE}#s2", "the Harbour Safety Act", ACT, None)
    assert (quill.from_id, quill.quote, quill.family_id, quill.section_id) == (
        f"{QUILL}#s2", "section 1", "fic-quillmouth-port-code", "s1")
    by_id = {r.source_id: r for r in RECORDS}
    for r in GRAPH.references:
        assert by_id[r.source_id].text[r.start : r.end] == r.quote
        assert r.content_hash == by_id[r.source_id].content_hash
    assert GRAPH.supersessions == ((ACT2, ACT1),)  # taken from the records, not the file


def _set(path: str, value: object):
    def change(raw: dict) -> None:
        if path in raw:
            raw[path] = value
        else:
            raw["references"][0][path] = value
    return change


def _drop(key: str):
    return lambda raw: raw["references"][0].pop(key)


BROKEN = {
    "schema": (_set("schema", "other"), "expected schema"),
    "version": (_set("schema_version", True), "expected schema"),
    "not fictional": (_set("fictional", 1), "'fictional' must be true"),
    "notice": (_set("notice", "edges"), "notice: must say"),
    "references not a list": (_set("references", {}), "expected a list"),
    "missing key": (_drop("quote"), "missing quote"),
    "extra key": (lambda raw: raw["references"][0].update(kind="cites"), "unexpected key"),
    "from shape": (_set("from", GUIDANCE), ".from:"),
    "from unknown source": (_set("from", "fic-nowhere@v1#s1"), ".from:"),
    "from unknown section": (_set("from", f"{GUIDANCE}#s9"), ".from:"),
    "from not str": (_set("from", ["x"]), ".from:"),
    "bool offset": (_set("start", True), "must be ints"),
    "empty span": (_set("end", 327), "must be ints"),
    "span outside section": (lambda raw: raw["references"][0].update(start=113, end=135),
                             "inside the citing section"),
    "quote differs": (_set("quote", "the harbour safety act"), ".quote:"),
    "quote not str": (_set("quote", None), ".quote:"),
    "to unknown family": (_set("to", "fic-nowhere"), ".to:"),
    "to unknown section": (_set("to", f"{ACT}#s9"), ".to:"),
    "to a version": (_set("to", ACT2), ".to:"),
    "to not str": (_set("to", 7), ".to:"),
    "self reference": (_set("to", "fic-harbour-inspection-guidance#s2"), "itself"),
    "public names a private memo": (_set("to", "fic-orchid-fleet-memo"), "narrower scope"),
    "repeated": (lambda raw: raw["references"].append(copy.deepcopy(raw["references"][0])),
                 "repeats an earlier reference"),
    "too many": (lambda raw: raw.update(references=[{}] * (MAX_REFERENCES + 1)), "at most 1000"),
}


@pytest.mark.parametrize("name", sorted(BROKEN))
def test_a_broken_references_file_is_refused(name: str) -> None:
    change, expected = BROKEN[name]
    raw = copy.deepcopy(RAW_REFS)
    change(raw)
    with pytest.raises(ProvenanceError) as caught:
        build_graph(RECORDS, raw)
    assert expected in str(caught.value)


def test_the_top_level_must_be_an_object_and_records_must_be_records() -> None:
    with pytest.raises(ProvenanceError, match="expected an object"):
        build_graph(RECORDS, [])
    with pytest.raises(TypeError):
        build_graph([*RECORDS, "x"], RAW_REFS)  # type: ignore[list-item]
    with pytest.raises(ProvenanceError, match="two share a source_id"):
        build_graph([*RECORDS, RECORDS[0]], RAW_REFS)


def test_a_tenant_wide_source_cannot_name_one_matters_memo() -> None:
    records = [dataclasses.replace(r, tenant="tenant-basalt") if r.source_id == ORCHID else r
               for r in RECORDS]
    raw = refs_file(ref("fic-basalt-file-note@v1#s1", "hull inspection", "fic-orchid-fleet-memo",
                        records))
    with pytest.raises(ProvenanceError, match="narrower scope"):
        build_graph(records, raw)


def test_a_private_source_may_name_a_public_one() -> None:
    raw = refs_file(ref(f"{ORCHID}#s1", "hull inspections", ACT))
    s = scope(tenant="tenant-orchid", matter="matter-orchid-001")
    v = view(s)
    result = expand(v, build_graph(RECORDS, raw), [seed(v, f"{ORCHID}#s1")])
    assert ids(result) == [(0, f"{ORCHID}#s1"), (1, f"{ACT2}#s1"), (1, f"{ACT2}#s2"),
                           (1, f"{ACT2}#s3")]


def test_unreadable_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="not valid JSON"):
        load_graph(RECORDS, bad)
    bad.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ProvenanceError, match="duplicate JSON key"):
        load_graph(RECORDS, bad)
    with pytest.raises(OSError):
        load_graph(RECORDS, tmp_path / "missing.json")


# --- expansion through the bundled edges ------------------------------------------


def test_a_reference_reaches_the_version_in_force_with_its_citing_words() -> None:
    v = view()
    start = seed(v, f"{GUIDANCE}#s2")
    result = expand(v, GRAPH, [start])
    assert ids(result) == [(0, f"{GUIDANCE}#s2"), (1, f"{ACT2}#s1"), (1, f"{ACT2}#s2"),
                           (1, f"{ACT2}#s3")]
    assert result.sections[0] == Reached(v.parents[f"{GUIDANCE}#s2"], 0, (start,), ())
    assert result.sections[1].path == (
        Hop("reference", f"{GUIDANCE}#s2", f"{ACT2}#s1", "the Harbour Safety Act", 327, 349),)
    assert (result.not_available, result.truncated) == (0, False)


def test_supersession_never_brings_back_a_version_out_of_force() -> None:
    v = view()
    result = expand(v, GRAPH, [seed(v, f"{GUIDANCE}#s2")], max_hops=2)
    assert all(item.parent.source_id != ACT1 for item in result.sections)
    assert result.not_available == 1  # v2 -> v1, counted once for all three sections
    old = view(scope(date(2020, 6, 1)))
    result = expand(old, GRAPH, [seed(old, f"{ACT1}#s1")], max_hops=MAX_HOPS)
    assert ids(result) == [(0, f"{ACT1}#s1")]
    assert result.not_available == 1  # v1 -> v2, not yet in force


@pytest.mark.parametrize("year", range(2018, 2027))
@pytest.mark.parametrize("tenant", [("public", None), ("tenant-orchid", "matter-orchid-001"),
                                    ("tenant-basalt", None)])
def test_every_section_reached_passes_the_eligibility_policy(year: int, tenant) -> None:
    s = scope(date(year, 7, 1), tenant=tenant[0], matter=tenant[1])
    v = view(s)
    raw = refs_file(*RAW_REFS["references"], ref(f"{ORCHID}#s1", "hull inspections", ACT),
                    ref(f"{ACT2}#s1", "hull inspection", f"{ACT}#s3"))
    result = expand(v, build_graph(RECORDS, raw), sorted(v.children), max_hops=MAX_HOPS,
                    max_candidates=MAX_CANDIDATES)
    by_id = {r.source_id: r for r in RECORDS}
    reached = [item.parent.parent_id for item in result.sections]
    assert len(reached) == len(set(reached))
    for item in result.sections:
        assert item.parent.parent_id in v.parents
        assert ineligibility(by_id[item.parent.source_id], s) == ()
    assert len({item.parent.source_id for item in result.sections
                if item.parent.source_id.startswith(ACT)}) <= 1


def test_an_unknown_end_date_keeps_a_reference_closed_until_the_date_is_known() -> None:
    quill = scope(date(2022, 6, 1), place="fic-quillmouth")
    assert view(quill).sources == {}  # end "unknown" fails closed
    raw = copy.deepcopy(RAW)
    next(r for r in raw["sources"] if r["source_id"] == QUILL)["effective_to"] = None
    records = validate_corpus(raw)
    v = view(quill, records)
    result = expand(v, build_graph(records, RAW_REFS), [seed(v, f"{QUILL}#s2")])
    assert ids(result) == [(0, f"{QUILL}#s2"), (1, f"{QUILL}#s1")]
    assert result.sections[1].path[0].quote == "section 1"


# --- bounds, cycles and order ------------------------------------------------------

CYCLE = build_graph(RECORDS, refs_file(ref(f"{ACT2}#s1", "hull inspection", f"{ACT}#s3"),
                                       ref(f"{ACT2}#s3", "valid inspection", f"{ACT}#s1")))


def test_a_cycle_visits_each_section_once() -> None:
    v = view()
    result = expand(v, CYCLE, [seed(v, f"{ACT2}#s1")], max_hops=MAX_HOPS)
    assert ids(result) == [(0, f"{ACT2}#s1"), (1, f"{ACT2}#s3")]
    both = expand(v, CYCLE, [seed(v, f"{ACT2}#s3"), seed(v, f"{ACT2}#s1")], max_hops=MAX_HOPS)
    assert ids(both) == [(0, f"{ACT2}#s3"), (0, f"{ACT2}#s1")]


def test_paths_chain_hop_by_hop() -> None:
    chain = build_graph(RECORDS, refs_file(
        ref(f"{GUIDANCE}#s2", "the Harbour Safety Act", f"{ACT}#s3"),
        ref(f"{ACT2}#s3", "valid inspection", f"{ACT}#s1")))
    v = view()
    result = expand(v, chain, [seed(v, f"{GUIDANCE}#s2")], max_hops=MAX_HOPS)
    assert ids(result) == [(0, f"{GUIDANCE}#s2"), (1, f"{ACT2}#s3"), (2, f"{ACT2}#s1")]
    path = result.sections[2].path
    assert [(hop.from_id, hop.to_id, hop.quote) for hop in path] == [
        (f"{GUIDANCE}#s2", f"{ACT2}#s3", "the Harbour Safety Act"),
        (f"{ACT2}#s3", f"{ACT2}#s1", "valid inspection")]
    assert all(a.to_id == b.from_id for a, b in pairwise(path))
    assert result.not_available == 1  # v2's supersession edge, tried from both act sections
    short = expand(v, chain, [seed(v, f"{GUIDANCE}#s2")], max_hops=1)
    assert ids(short) == ids(result)[:2]


def test_hops_and_the_section_cap_bound_the_walk() -> None:
    v = view()
    start = [seed(v, f"{GUIDANCE}#s2")]
    assert ids(expand(v, GRAPH, start, max_hops=0)) == [(0, f"{GUIDANCE}#s2")]
    assert expand(v, GRAPH, []) == Expansion((), 0, False, 1, 10)
    capped = expand(v, GRAPH, start, max_candidates=2)
    assert (len(capped.sections), capped.truncated) == (3, True)
    exact = expand(v, GRAPH, start, max_candidates=3)
    assert (len(exact.sections), exact.truncated) == (4, False)
    assert (exact.max_hops, exact.max_candidates) == (1, 3)


def test_the_result_ignores_record_and_file_order() -> None:
    v = view()
    start = [seed(v, f"{GUIDANCE}#s2"), seed(v, f"{ACT2}#s1")]
    flipped = {**RAW_REFS, "references": RAW_REFS["references"][::-1]}
    records = RECORDS[::-1]
    assert expand(view(records=records), build_graph(records, flipped), start, max_hops=2) \
        == expand(v, GRAPH, start, max_hops=2)


# --- refusals ---------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"max_hops": -1}, {"max_hops": MAX_HOPS + 1},
                                    {"max_hops": True}, {"max_hops": 1.0},
                                    {"max_candidates": 0}, {"max_candidates": MAX_CANDIDATES + 1}])
def test_bad_limits_are_refused(kwargs: dict[str, object]) -> None:
    v = view()
    with pytest.raises(ValueError, match="must be an int"):
        expand(v, GRAPH, [seed(v, f"{GUIDANCE}#s2")], **kwargs)  # type: ignore[arg-type]


def test_bad_arguments_are_refused() -> None:
    v = view()
    one = seed(v, f"{GUIDANCE}#s2")
    with pytest.raises(TypeError):
        expand("view", GRAPH, [one])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        expand(dataclasses.replace(v, scope="public"), GRAPH, [one])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        expand(v, {"references": ()}, [one])  # type: ignore[arg-type]
    for seeds in (iter([one]), one, {one}):
        with pytest.raises(TypeError):
            expand(v, GRAPH, seeds)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at most"):
        expand(v, GRAPH, [one] * (MAX_SEEDS + 1))
    with pytest.raises(ValueError, match="repeat"):
        expand(v, GRAPH, [one, one])


def test_a_seed_outside_the_view_fails_with_one_message() -> None:
    v = view()
    private = view(scope(tenant="tenant-orchid", matter="matter-orchid-001"))
    for bad in (seed(private, f"{ORCHID}#s1"), f"{GUIDANCE}#s2", GUIDANCE, "nothing", 3):
        with pytest.raises(NotAvailable) as caught:
            expand(v, GRAPH, [bad, bad])  # a leak is reported before the repeat
        assert str(caught.value) == NOT_AVAILABLE


def test_a_forged_view_cannot_smuggle_a_superseded_version_through_the_graph() -> None:
    v, old = view(), view(scope(date(2020, 6, 1)))
    forged = ScopedView(
        v.scope, MappingProxyType({**v.sources, **old.sources}),
        MappingProxyType({**v.parents, **old.parents}),
        MappingProxyType({**v.children, **old.children}))
    with pytest.raises(EligibilityError, match="outside the scope"):
        expand(forged, GRAPH, [seed(v, f"{ACT2}#s1")])
    with pytest.raises(EligibilityError, match="outside the scope"):
        expand(forged, GRAPH, [seed(old, f"{ACT1}#s1")], max_hops=0)  # a seed is checked too


def test_a_forged_section_is_refused() -> None:
    v = view()
    target = f"{ACT2}#s1"
    parents = {**v.parents, target: dataclasses.replace(v.parents[target], text="Forged.")}
    forged = dataclasses.replace(v, parents=MappingProxyType(parents))
    with pytest.raises(EligibilityError, match="does not match"):
        expand(forged, GRAPH, [seed(v, f"{GUIDANCE}#s2")])


def test_a_hand_built_cross_tenant_edge_is_not_followed_or_named() -> None:
    v = view()
    leak = dataclasses.replace(GRAPH.references[0], family_id="fic-orchid-fleet-memo")
    result = expand(v, Graph((leak,), ()), [seed(v, f"{GUIDANCE}#s2")])
    assert ids(result) == [(0, f"{GUIDANCE}#s2")]
    assert result.not_available == 1
    assert "orchid" not in repr(result)


@pytest.mark.parametrize("change", [{"content_hash": "sha256:" + "0" * 64}, {"quote": "the Act"},
                                    {"source_id": ACT2}])
def test_a_stale_reference_is_refused(change: dict[str, object]) -> None:
    v = view()
    stale = dataclasses.replace(GRAPH.references[0], **change)
    with pytest.raises(EligibilityError, match="reference does not match"):
        expand(v, Graph((stale,), ()), [seed(v, f"{GUIDANCE}#s2")])


def test_a_reference_moved_to_another_section_is_refused() -> None:
    v = view()
    moved = dataclasses.replace(GRAPH.references[0], from_id=f"{GUIDANCE}#s1")
    with pytest.raises(EligibilityError, match="reference does not match"):
        expand(v, Graph((moved,), ()), [seed(v, f"{GUIDANCE}#s1")])


def test_records_are_not_mutated() -> None:
    before = copy.deepcopy(RECORDS)
    v = view()
    expand(v, GRAPH, sorted(v.children), max_hops=MAX_HOPS)
    assert before == RECORDS
    assert isinstance(GRAPH.references[0], Reference)


# --- the command line -------------------------------------------------------------

README_QUERY = "how does the guidance note relate to the Harbour Safety Act"
README_ARGS = [README_QUERY, "--tenant", "public", "--jurisdiction", "fic-varnholt",
               "--regime", "maritime-safety", "--as-of", "2024-06-01", "--top-k", "1",
               "--max-hops", "2"]


def test_readme_shows_the_command_and_its_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(README_ARGS) == 0
    out = capsys.readouterr().out
    command = (f'python -m examples.legal_rag.graph \\\n    "{README_QUERY}" \\\n'
               "    --tenant public --jurisdiction fic-varnholt --regime maritime-safety "
               "--as-of 2024-06-01 \\\n    --top-k 1 --max-hops 2\n```\n\n```\n")
    assert command + out + "```" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_cli_lines(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([*README_ARGS[:-1], "1", "--max-candidates", "1"]) == 0
    out = capsys.readouterr().out
    assert "stopped at the section cap" in out
    hop = Hop("supersedes", f"{ACT2}#s1", f"{ACT1}#s1", None, None, None)
    item = Reached(view(scope(date(2020, 6, 1))).parents[f"{ACT1}#s1"], 1, (), (hop,))
    assert graph_module._line(item).endswith(f"supersedes from {ACT2}#s1")


@pytest.mark.parametrize("flags", [["--top-k", "0"], ["--max-hops", "9"], ["--tenant", "Public"],
                                   ["--as-of", "2024-6-1"]])
def test_cli_refuses_bad_flags(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main([*README_ARGS, *flags])
    assert caught.value.code == 2


def test_cli_reports_unreadable_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for flag in ("--fixture", "--references"):
        missing = tmp_path / f"{flag[2:]}.json"
        assert main([*README_ARGS, flag, str(missing)]) == 1
        assert str(missing) in capsys.readouterr().err


def test_cli_reports_an_unavailable_seed(monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    def refuse(*args: object, **kwargs: object) -> Expansion:
        raise NotAvailable
    monkeypatch.setattr(graph_module, "expand", refuse)
    assert main(README_ARGS) == 1
    assert NOT_AVAILABLE in capsys.readouterr().err
