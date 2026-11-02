"""The legal fixture's provenance contract.

The bundled fixture must pass, and every way of breaking a record must be
caught with a message that names the rule. Each malformed case changes one
thing in an otherwise valid copy, re-hashing the text when the text changes,
so the expected message comes from the rule under test and not from a side
effect. Hostile values in any field must end in :class:`ProvenanceError`,
never another exception. These are software checks on fictional data; they
say nothing about legal accuracy.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import subprocess
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from examples.legal_rag.provenance import (
    FIXTURE,
    NOTICE,
    UNKNOWN,
    ProvenanceError,
    content_hash,
    load,
    main,
    validate_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
CARD = FIXTURE.parents[1] / "DATA_CARD.md"
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
V1, V2, V3 = (f"fic-harbour-safety-act@v{n}" for n in (1, 2, 3))
ORCHID, BASALT = "fic-orchid-fleet-memo@v1", "fic-basalt-file-note@v1"
VALID = "6 valid records: 5 families, 3 tenant/matter scopes\n"


def src(raw: dict[str, Any], source_id: str) -> dict[str, Any]:
    return next(s for s in raw["sources"] if s["source_id"] == source_id)


def problems_of(raw: object) -> tuple[str, ...]:
    with pytest.raises(ProvenanceError) as info:
        validate_corpus(raw)
    return info.value.problems


def set_text(record: dict[str, Any], text: str) -> None:
    """Change the text and keep its hash in step, so only the rule under test fires."""
    record["text"], record["content_hash"] = text, content_hash(text)


# Numbers the messages quote, read from the fixture so an edit to it cannot strand them.
V1_INDEX = [s["source_id"] for s in RAW["sources"]].index(V1)
V1_LEN = len(src(RAW, V1)["text"])
S0_START, S0_END = (src(RAW, V1)["sections"][0][k] for k in ("start", "end"))


# --- the bundled fixture ------------------------------------------------------


def test_bundled_fixture_is_valid_and_labelled_fictional() -> None:
    records = load()
    assert len(records) == 6
    assert RAW["fictional"] is True
    for r in records:
        assert r.origin == "original-fictional" and r.license == "MIT"
        assert r.title.endswith("(fictional)")
        assert r.text.startswith(NOTICE + "\n")


def test_records_are_frozen() -> None:
    record = load()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.tenant = "tenant-orchid"  # type: ignore[misc]
    assert isinstance(record.sections, tuple) and isinstance(record.labels, tuple)


def test_sections_slice_the_text_the_cases_rely_on() -> None:
    by_id = {r.source_id: r for r in load()}

    def section(source_id: str, n: int = 0) -> str:
        s = by_id[source_id].sections[n]
        return by_id[source_id].text[s.start : s.end]

    assert section(V1) == section(V2).replace("12 months", "24 months")  # outdated version
    assert "every 18 months" in section("fic-harbour-inspection-guidance@v1")  # conflict
    assert "every 6 months" in section("fic-quillmouth-port-code@v1")  # other jurisdiction
    assert by_id["fic-quillmouth-port-code@v1"].jurisdiction == "fic-quillmouth"
    for r in by_id.values():
        assert r.content_hash == content_hash(r.text)
        assert all(r.text[s.start : s.end].startswith(s.heading + "\n") for s in r.sections)


def test_fixture_exercises_every_unknown_and_every_scope() -> None:
    records = load()
    fields = ("jurisdiction", "regime", "authority_type", "effective_from", "effective_to")
    for name in fields:
        assert any(getattr(r, name) == UNKNOWN for r in records), name
    assert any(s.pages == UNKNOWN for r in records for s in r.sections)
    assert any(r.effective_to is None for r in records)  # in force, no end
    scopes = {(r.tenant, r.matter) for r in records}
    assert scopes == {
        ("public", None),
        ("tenant-orchid", "matter-orchid-001"),
        ("tenant-basalt", None),  # tenant-wide, no matter
    }


def test_supersession_hands_over_on_one_day() -> None:
    by_id = {r.source_id: r for r in load()}
    assert by_id[V2].supersedes == V1
    assert by_id[V1].effective_to == by_id[V2].effective_from == date(2023, 7, 1)


def test_data_card_rows_match_the_fixture() -> None:
    card = CARD.read_text(encoding="utf-8")
    lines = [line for line in card.splitlines() if line.startswith("| `fic-")]
    rows = {line.split(" | ")[0][2:]: line for line in lines}
    records = load()
    assert len(rows) == len(records)
    for r in records:
        row = rows[f"`{r.source_id}`"]
        for fact in (r.tenant, r.jurisdiction, r.regime, f" {r.authority_type} | "):
            assert fact in row, (fact, row)
        assert f" {', '.join(r.labels)} | " in row, row
    assert f"\n{VALID}" in card


def test_readme_shows_the_command_and_its_output() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"python -m examples.legal_rag.provenance\n```\n\n```\n{VALID}```" in readme


# --- one broken record --------------------------------------------------------

Mutation = Callable[[dict[str, Any]], None]
HUGE = 10**5000  # past Python's int-to-str digit limit, so repr() of it raises


def _swap_sections(r: dict[str, Any]) -> None:
    r["sections"][0], r["sections"][1] = r["sections"][1], r["sections"][0]


def _pages_go_back(r: dict[str, Any]) -> None:
    r["sections"][1]["pages"], r["sections"][2]["pages"] = [2, 2], [1, 1]


def _mid_line(r: dict[str, Any]) -> None:
    r["sections"][0].update(start=S0_START + 3, heading="Inspections")


def _sec(**changes: Any) -> Mutation:
    return lambda r: r["sections"][0].update(**changes)


RECORD_CASES: list[tuple[str, Mutation, str]] = [
    ("missing key", lambda r: r.pop("license"), "missing license"),
    ("unexpected key", lambda r: r.update(licence="MIT"), "unexpected key(s) 'licence'"),
    ("null unknown", lambda r: r.update(jurisdiction=None), 'write "unknown", not null'),
    ("n/a", lambda r: r.update(jurisdiction="N/A"), 'write exactly "unknown"'),
    ("empty", lambda r: r.update(jurisdiction=""), 'write exactly "unknown"'),
    ("blank", lambda r: r.update(regime="  "), 'write exactly "unknown"'),
    ("none", lambda r: r.update(regime="none"), 'write exactly "unknown"'),
    ("TBD", lambda r: r.update(jurisdiction="TBD"), 'write exactly "unknown"'),
    ("Unknown", lambda r: r.update(regime="Unknown"), 'write exactly "unknown"'),
    ("fic-unknown", lambda r: r.update(jurisdiction="fic-unknown"), 'write exactly "unknown"'),
    ("real jurisdiction", lambda r: r.update(jurisdiction="us"), "does not match fic-"),
    ("long id", lambda r: r.update(regime="r" * 81), "is longer than 80 characters"),
    ("unknown tenant", lambda r: r.update(tenant=UNKNOWN), "this field must be known"),
    ("tenant-unknown", lambda r: r.update(tenant="tenant-unknown"), "must be known"),
    ("null tenant", lambda r: r.update(tenant=None), "tenant: expected a string"),
    ("public matter", lambda r: r.update(matter="matter-x"), "public source cannot belong"),
    ("unknown matter", lambda r: r.update(matter=UNKNOWN), "this field must be known"),
    ("matter-none", lambda r: r.update(matter="matter-none"), "this field must be known"),
    ("id drift", lambda r: r.update(version=9), "source_id is not '<family_id>@v<version>'"),
    ("bool version", lambda r: r.update(version=True), "version: expected an integer"),
    ("zero version", lambda r: r.update(version=0), "from 1 to 1000000000, got 0"),
    ("huge version", lambda r: r.update(version=HUGE), "got <unprintable int>"),
    ("unlabelled title", lambda r: r.update(title="Harbour Act"), "end with '(fictional)'"),
    ("label mid-title", lambda r: r.update(title="A fictional Act"), "end with '(fictional)'"),
    ("padded title", lambda r: r.update(title=" Act (fictional)"), "trailing whitespace"),
    ("surrogate title", lambda r: r.update(title="\ud800 (fictional)"), "lone surrogate"),
    ("bad authority", lambda r: r.update(authority_type="law"), "authority_type: 'law'"),
    ("list authority", lambda r: r.update(authority_type=["statute"]), "is not one of"),
    ("slashed date", lambda r: r.update(effective_from="2019/01/01"), "is not a YYYY-MM-DD"),
    ("compact date", lambda r: r.update(effective_from="20190101"), "is not a YYYY-MM-DD"),
    ("impossible date", lambda r: r.update(effective_from="2019-02-30"), "is not a YYYY"),
    ("null start", lambda r: r.update(effective_from=None), "None is not a YYYY-MM-DD"),
    ("empty interval", lambda r: r.update(effective_to="2019-01-01"), "must be before"),
    ("reversed interval", lambda r: r.update(effective_to="2018-01-01"), "must be before"),
    ("bad supersedes", lambda r: r.update(supersedes=7), "expected a source_id or null"),
    ("bad licence", lambda r: r.update(license="proprietary"), "is not one of ['MIT']"),
    ("dict licence", lambda r: r.update(license={"spdx": "MIT"}), "is not one of ['MIT']"),
    ("bad origin", lambda r: r.update(origin="scraped"), "origin: 'scraped'"),
    ("no labels", lambda r: r.update(labels=[]), "labels: expected a non-empty list"),
    ("twin labels", lambda r: r.update(labels=["a", "a"]), "labels must be distinct"),
    ("bad label", lambda r: r.update(labels=["Old Version"]), "labels[0]: 'Old Version'"),
    ("huge label", lambda r: r.update(labels=[HUGE]), "labels[0]: expected a string"),
    ("stale hash", lambda r: r.update(text=r["text"].replace("24", "36")), "does not match the"),
    ("md5 hash", lambda r: r.update(content_hash="md5:abc"), "64 lowercase hex"),
    ("empty text", lambda r: set_text(r, ""), "expected non-empty text"),
    ("no notice", lambda r: set_text(r, r["text"].replace("FICTIONAL", "Fictional")), "open"),
    ("lone surrogate", lambda r: r.update(text=r["text"] + "\ud800"), "lone surrogate"),
    ("non-NFC text", lambda r: set_text(r, r["text"].replace("credits", "crédits")), "NFC"),
    ("no sections", lambda r: r.update(sections=[]), "sections: expected a non-empty list"),
    ("section key", lambda r: r["sections"][0].pop("pages"), "sections[0]: missing pages"),
    ("heading", _sec(heading="1. Inspection"), "not the heading"),
    ("mid-line span", _mid_line, "span must start at the beginning of a line"),
    ("outside", lambda r: r["sections"][2].update(end=10**6), f"(length {V1_LEN})"),
    ("empty span", _sec(end=S0_START), "must be non-empty"),
    ("overlap", lambda r: r["sections"][1].update(start=S0_END - 9), f"end ({S0_END})"),
    ("unordered", _swap_sections, "previous end"),
    ("twin ids", lambda r: r["sections"][1].update(section_id="s1"), "duplicate section_id"),
    ("bad id", _sec(section_id="1"), "does not match s[1-9]"),
    ("negative start", _sec(start=-1), "start: expected an integer from 0"),
    ("float start", _sec(start=1.5), "got 1.5"),
    ("huge start", _sec(start=HUGE), "got <unprintable int>"),
    ("reversed pages", _sec(pages=[2, 1]), "pages must be"),
    ("page zero", _sec(pages=[0, 1]), "pages must be"),
    ("bool pages", _sec(pages=[True, 1]), "pages must be"),
    ("text pages", _sec(pages="1-2"), "pages must be"),
    ("huge pages", _sec(pages=[1, HUGE]), "pages must be"),
    ("pages go back", _pages_go_back, "start before the previous section's"),
]


@pytest.mark.parametrize(
    ("mutate", "expected"), [c[1:] for c in RECORD_CASES], ids=[c[0] for c in RECORD_CASES]
)
def test_a_broken_record_is_rejected_by_name(mutate: Mutation, expected: str) -> None:
    raw = copy.deepcopy(RAW)
    mutate(src(raw, V1))
    found = problems_of(raw)
    assert any(expected in p for p in found), found
    # Only V1 is blamed: V2 is not also reported for pointing at the rejected V1.
    assert all(p.startswith(f"sources[{V1_INDEX}] ") for p in found), found


def test_a_record_that_is_not_an_object_is_rejected() -> None:
    raw = copy.deepcopy(RAW)
    raw["sources"][4] = ORCHID
    assert problems_of(raw) == (f"sources[4]: expected an object, got '{ORCHID}'",)


def test_a_bad_span_is_blamed_once_not_on_every_later_section() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V1)["sections"][0]["end"] = 10**8
    (problem,) = problems_of(raw)
    assert ".sections[0]: span [" in problem


def test_unexpected_keys_are_listed_at_most_five_at_a_time() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V1).update({f"extra{n:02}": n for n in range(20)})
    (problem,) = problems_of(raw)
    assert problem.endswith("'extra04' and 15 more") and "extra05" not in problem


def test_problems_in_different_records_are_all_reported() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V1)["license"] = "proprietary"
    src(raw, ORCHID)["tenant"] = UNKNOWN
    found = problems_of(raw)
    assert len(found) == 2
    assert "license" in found[0] and "tenant" in found[1]


def test_file_values_in_messages_are_escaped_and_cut_short() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V1)["jurisdiction"] = "\x1b[2J" + "x" * 10_000
    (problem,) = problems_of(raw)
    assert "\x1b" not in problem and "\\x1b[2J" in problem
    assert len(problem) < 200


HOSTILE: list[Any] = [
    None, True, 1.5, -1, HUGE, "", "  ", "\ud800", "x" * 100_000, [], {}, [[[]]], {"a": 1},
]


def test_hostile_values_in_any_field_raise_only_provenance_errors() -> None:
    record = src(RAW, V1)
    targets = [("fixture", k) for k in RAW] + [("record", k) for k in record]
    targets += [("section", k) for k in record["sections"][0]]
    for level, key in targets:
        for value in HOSTILE:
            raw = copy.deepcopy(RAW)
            nodes = {"fixture": raw, "record": src(raw, V1)}
            nodes["section"] = nodes["record"]["sections"][0]
            if value == nodes[level][key]:  # V1's own null matter and supersedes are valid
                continue
            nodes[level][key] = value
            with pytest.raises(ProvenanceError) as info:
                validate_corpus(raw)
            assert all(len(p) < 400 for p in info.value.problems), (level, key)


def test_provenance_error_survives_pickle_and_copy() -> None:
    error = ProvenanceError(["first problem", "second"])
    for clone in (pickle.loads(pickle.dumps(error)), copy.copy(error)):
        assert clone.problems == error.problems
        assert str(clone) == "first problem\nsecond"
    assert ProvenanceError("one problem").problems == ("one problem",)


# --- rules across records -----------------------------------------------------

CorpusMutation = Callable[[dict[str, Any]], None]


def _fork(raw: dict[str, Any]) -> None:
    v3 = copy.deepcopy(src(raw, V2))
    v3.update(source_id=V3, version=3, supersedes=V1)
    raw["sources"].append(v3)


def _skipped_overlap(raw: dict[str, Any]) -> None:
    """v1 ends 2023-07-01, v2's dates are unknown, and v3 starts before v1 ends."""
    src(raw, V2).update(effective_from=UNKNOWN, effective_to=UNKNOWN)
    v3 = copy.deepcopy(src(raw, V2))
    v3.update(source_id=V3, version=3, supersedes=V2, effective_from="2023-01-01")
    raw["sources"].append(v3)


def _v2(**changes: Any) -> CorpusMutation:
    return lambda raw: src(raw, V2).update(**changes)


CORPUS_CASES: list[tuple[str, CorpusMutation, str]] = [
    ("duplicate id", lambda raw: raw["sources"].append(copy.deepcopy(src(raw, V1))),
     f"{V1}: duplicate source_id"),
    ("missing target", _v2(supersedes="fic-harbour-safety-act@v7"),
     "which is not a lower version of the same family"),
    ("other family", _v2(supersedes="fic-quillmouth-port-code@v1"),
     "which is not a lower version of the same family"),
    ("self", _v2(supersedes=V2), "which is not a lower version"),
    ("fork", _fork, f"{V3}: must supersede the version before it, {V2}"),
    ("unlinked current", _v2(supersedes=None), f"{V2}: must supersede the version before it"),
    ("unlinked overlap", _v2(supersedes=None, effective_from="2019-01-01"),
     f"{V2}: starts 2019-01-01, not after {V1} (2019-01-01)"),
    ("open-ended predecessor", lambda raw: src(raw, V1).update(effective_to=None),
     "whose effective_to is null (no end)"),
    ("gap", lambda raw: src(raw, V1).update(effective_to="2023-06-01"),
     "versions must hand over on one day"),
    ("overlapping handover", lambda raw: src(raw, V1).update(effective_to="2024-01-01"),
     "versions must hand over on one day"),
    ("starts first", _v2(effective_from="2018-06-01"),
     f"starts 2018-06-01, not after {V1} (2019-01-01)"),
    ("skipped overlap", _skipped_overlap, f"{V3}: starts 2023-01-01, before {V1} ends"),
    ("ends before predecessor", _v2(effective_from=UNKNOWN, effective_to="2018-01-01"),
     f"{V2}: ends 2018-01-01, before {V1} starts (2019-01-01)"),
    ("shared matter", lambda raw: src(raw, BASALT).update(matter="matter-orchid-001"),
     "matter 'matter-orchid-001' also belongs to another tenant"),
    ("tenant drift", _v2(tenant="tenant-orchid"), f"tenant differs from {V1}"),
    ("jurisdiction drift", _v2(jurisdiction="fic-quillmouth"), "jurisdiction differs from"),
    ("regime drift", _v2(regime="harbour-fees"), "regime differs from"),
    ("authority drift", _v2(authority_type="contract"), "authority_type differs from"),
    ("not fictional", lambda raw: raw.update(fictional="true"), "'fictional' must be true"),
    ("schema version", lambda raw: raw.update(schema_version=2), "expected schema"),
    ("bool version", lambda raw: raw.update(schema_version=True), "expected schema"),
    ("float version", lambda raw: raw.update(schema_version=1.0), "expected schema"),
    ("schema name", lambda raw: raw.update(schema="other"), "expected schema"),
    ("empty notice", lambda raw: raw.update(notice=""), "fixture.notice"),
    ("unlabelled notice", lambda raw: raw.update(notice="Test data."), "say the sources are"),
    ("no sources", lambda raw: raw.update(sources=[]), "sources: expected a non-empty list"),
    ("object sources", lambda raw: raw.update(sources={}), "expected a non-empty list"),
    ("top-level key", lambda raw: raw.update(extra=1), "fixture: unexpected key(s) 'extra'"),
]


@pytest.mark.parametrize(
    ("mutate", "expected"), [c[1:] for c in CORPUS_CASES], ids=[c[0] for c in CORPUS_CASES]
)
def test_a_broken_corpus_is_rejected_by_name(mutate: CorpusMutation, expected: str) -> None:
    raw = copy.deepcopy(RAW)
    mutate(raw)
    found = problems_of(raw)
    assert any(expected in p for p in found), found


@pytest.mark.parametrize(
    ("source_id", "field"), [(V1, "effective_to"), (V2, "effective_from")]
)
def test_an_unknown_handover_date_is_allowed(source_id: str, field: str) -> None:
    raw = copy.deepcopy(RAW)
    src(raw, source_id)[field] = UNKNOWN
    assert getattr({r.source_id: r for r in validate_corpus(raw)}[source_id], field) == UNKNOWN


def test_a_third_version_extends_the_chain() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V2)["effective_to"] = "2025-01-01"
    v3 = copy.deepcopy(src(raw, V2))
    v3.update(source_id=V3, version=3, supersedes=V2, effective_from="2025-01-01")
    v3["effective_to"] = None
    raw["sources"].append(v3)
    assert [r.source_id for r in validate_corpus(raw)][-1] == V3


def test_a_broken_middle_version_is_blamed_alone() -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V2)["effective_to"] = "2025-01-01"
    v3 = copy.deepcopy(src(raw, V2))
    v3.update(source_id=V3, version=3, supersedes=V2, effective_from="2025-01-01")
    v3["effective_to"] = None
    raw["sources"].append(v3)
    src(raw, V2)["license"] = "proprietary"
    (problem,) = problems_of(raw)  # v3 is not also blamed for following the rejected v2
    assert problem.startswith(f"sources[1] '{V2}'.license")


def test_a_fixture_that_is_not_an_object_is_rejected() -> None:
    assert problems_of([]) == ("fixture: expected an object, got []",)


# --- reading files and the command line ---------------------------------------


def test_a_duplicate_json_key_cannot_hide_a_scope(tmp_path: Path) -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    path = tmp_path / "dup.json"
    doubled = '"tenant": "tenant-orchid", "tenant": "public"'
    path.write_text(text.replace('"tenant": "public"', doubled, 1), encoding="utf-8")
    with pytest.raises(ProvenanceError, match="duplicate JSON key\\(s\\) 'tenant'"):
        load(path)


SURROGATE = FIXTURE.read_text(encoding="utf-8").replace(
    '"text": "FICTIONAL', '"text": "\\ud800FICTIONAL', 1
)


@pytest.mark.parametrize(
    "content",
    ["{", '{"a": ' + "9" * 5000 + "}", "[" * 100_000, SURROGATE],
    ids=["truncated", "5000-digit int", "deep nesting", "lone surrogate"],
)
def test_a_hostile_file_fails_cleanly(
    content: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "hostile.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ProvenanceError):
        load(path)
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "problem(s)" in err and "Traceback" not in err


def test_cli_reports_a_valid_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert capsys.readouterr().out == VALID


def test_cli_lists_problems_and_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = copy.deepcopy(RAW)
    src(raw, V1)["license"] = "proprietary"
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"{path}: 1 problem(s)\n  - sources[{V1_INDEX}] '{V1}'.license")


@pytest.mark.parametrize("args", [["a.json", "b.json"], ["--help"], ["-h"]])
def test_cli_usage(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(args) == 2
    assert "usage:" in capsys.readouterr().err


def test_cli_reports_an_unreadable_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_documented_command_runs_from_the_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.legal_rag.provenance"],
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=60,
    )
    assert (result.returncode, result.stdout) == (0, VALID), result.stderr
