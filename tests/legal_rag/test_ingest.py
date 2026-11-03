"""Structure-aware ingestion of the legal fixture.

Every passage must be an exact slice of its source text, stay inside one
section, and keep an ID that does not move when the corpus is re-ingested,
reordered or extended. The bundled sections are one or two sentences, so
windows and overlap are checked on longer synthetic sections built here and
validated by the same provenance contract. These are software checks on
fictional data; they say nothing about retrieval quality or legal accuracy.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from examples.legal_rag.ingest import Child, Ingestion, ingest, main
from examples.legal_rag.provenance import (
    FIXTURE,
    NOTICE,
    ProvenanceError,
    SourceRecord,
    content_hash,
    load,
    validate_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECORDS = load()
V2 = "fic-harbour-safety-act@v2"
GUIDANCE = "fic-harbour-inspection-guidance@v1"
SUMMARY = "6 sources -> 13 sections -> 13 passages (max 120 words, overlap 30)"


def synthetic(*sections: tuple[str, str], between: str = "") -> SourceRecord:
    """A valid one-record corpus whose sections are ``(heading, body)`` pairs.

    ``between`` is a line of text outside every section, put before each
    section after the first.
    """
    raw = copy.deepcopy(RAW)
    record = next(s for s in raw["sources"] if s["source_id"] == GUIDANCE)
    text, spans = f"{NOTICE}\nSynthetic fictional record\n", []
    for i, (heading, body) in enumerate(sections, start=1):
        text += f"\n{between}\n\n" if between and i > 1 else "\n"
        start = len(text)
        text += f"{heading}\n{body}\n" if body else f"{heading}\n"
        spans.append({"section_id": f"s{i}", "heading": heading, "start": start,
                      "end": len(text), "pages": "unknown"})
    record.update(text=text, content_hash=content_hash(text), sections=spans)
    raw["sources"] = [record]
    (parsed,) = validate_corpus(raw)
    return parsed


def clauses(n: int) -> str:
    """``n`` distinct five-word sentences on one line."""
    return " ".join(f"Clause {i} sets rule {'x' * (i + 1)}." for i in range(n))


def by_parent(result: Ingestion) -> dict[str, list[Child]]:
    groups: dict[str, list[Child]] = {}
    for child in result.children:
        groups.setdefault(child.parent_id, []).append(child)
    return groups


# --- exact offsets on the bundled fixture -------------------------------------


def test_every_parent_and_child_is_an_exact_slice_of_its_source() -> None:
    by_id = {r.source_id: r for r in RECORDS}
    result = ingest(RECORDS)
    assert len(result.parents) == sum(len(r.sections) for r in RECORDS) == 13
    for parent in result.parents:
        record = by_id[parent.source_id]
        section = next(s for s in record.sections if s.section_id == parent.section_id)
        assert parent.parent_id == f"{record.source_id}#{section.section_id}"
        assert (parent.start, parent.end, parent.heading) == (
            section.start, section.end, section.heading)
        assert parent.text == record.text[section.start : section.end]
        assert parent.pages == section.pages and parent.content_hash == record.content_hash
    for child in result.children:
        record = by_id[child.source_id]
        assert child.text == record.text[child.start : child.end]
        assert child.chunk_id == f"{child.parent_id}:{child.start}-{child.end}"
        assert child.words == len(child.text.split())
        assert child.content_hash == record.content_hash
        assert not child.text[0].isspace() and not child.text[-1].isspace()


@pytest.mark.parametrize(("max_words", "overlap_words"), [(120, 30), (12, 6), (1, 0), (5, 2)])
def test_children_cover_each_section_and_never_cross_it(max_words: int, overlap_words: int) -> None:
    result = ingest(RECORDS, max_words=max_words, overlap_words=overlap_words)
    groups = by_parent(result)
    for parent in result.parents:
        kids = groups[parent.parent_id]
        assert parent.child_ids == tuple(c.chunk_id for c in kids)
        assert kids[0].start == parent.start  # starts at the heading
        assert all(parent.start <= c.start < c.end <= parent.end for c in kids)
        covered = {i for c in kids for i in range(c.start, c.end)}
        text = next(r.text for r in RECORDS if r.source_id == parent.source_id)
        gaps = [i for i in range(parent.start, parent.end) if i not in covered]
        assert all(text[i].isspace() for i in gaps), parent.parent_id


def test_text_outside_sections_is_not_ingested() -> None:
    by_id = {r.source_id: r for r in RECORDS}
    for child in ingest(RECORDS, max_words=1, overlap_words=0).children:
        sections = by_id[child.source_id].sections
        assert any(s.start <= child.start < child.end <= s.end for s in sections)
        assert child.start >= sections[0].start  # never the notice or the title line


def test_text_between_sections_is_not_ingested() -> None:
    record = synthetic(("1. One", "Rule one."), ("2. Two", "Rule two."), between="Stray note.")
    texts = [c.text for c in ingest([record]).children]
    assert "Stray note." in record.text
    assert texts == ["1. One\nRule one.", "2. Two\nRule two."]


def test_a_heading_travels_with_the_first_sentence() -> None:
    result = ingest(RECORDS, max_words=1, overlap_words=0)  # a window of one unit
    texts = {c.chunk_id: c.text for c in result.children}
    for parent in result.parents:
        first = texts[parent.child_ids[0]]
        assert first.startswith(parent.heading + "\n") and len(first) > len(parent.heading) + 1


def test_a_heading_only_section_is_one_child() -> None:
    record = synthetic(("1. Scope", ""), ("2. Rule", "A vessel must keep a log."))
    result = ingest([record])
    assert [c.text for c in result.children] == ["1. Scope", "2. Rule\nA vessel must keep a log."]


# --- windows, overlap and oversized sentences ---------------------------------


def test_overlap_repeats_trailing_sentences_and_every_window_advances() -> None:
    record = synthetic(("1. Long section", clauses(12)))
    kids = ingest([record], max_words=20, overlap_words=8).children
    assert len(kids) > 3
    assert all(c.words <= 20 for c in kids)
    for prev, nxt in pairwise(kids):
        assert prev.start < nxt.start < prev.end < nxt.end  # overlaps, and reaches further
        shared = record.text[nxt.start : prev.end]
        assert shared.startswith("Clause") and shared.endswith(".")  # whole sentences


def test_no_overlap_gives_disjoint_children_separated_only_by_whitespace() -> None:
    record = synthetic(("1. Long section", clauses(12)))
    kids = ingest([record], max_words=20, overlap_words=0).children
    for prev, nxt in pairwise(kids):
        assert prev.end < nxt.start and record.text[prev.end : nxt.start].isspace()


def test_overlap_is_dropped_when_the_next_sentence_would_not_fit() -> None:
    # Units of 7 (heading + clause), 5 and 30 words with max 12, overlap 6:
    # carrying the 5-word clause back would leave no room for the 30-word
    # sentence, so the next window would sit inside the first one.
    body = f"{clauses(2)} {' '.join(['Each'] * 29)} ends."
    kids = ingest([synthetic(("1. Mixed", body))], max_words=12, overlap_words=6).children
    assert [c.words for c in kids] == [12, 30]
    assert kids[0].end < kids[1].start


def test_an_oversized_sentence_is_kept_whole_not_truncated() -> None:
    long_sentence = " ".join(["Every"] * 40) + " vessel must comply."
    record = synthetic(("1. Long", long_sentence))
    (child,) = ingest([record], max_words=10, overlap_words=2).children
    assert child.text == f"1. Long\n{long_sentence}" and child.words == 2 + 43


def test_lines_are_boundaries_even_without_full_stops() -> None:
    body = "(a) vessels over 20 metres\n(b) vessels carrying cargo\n\n(c) ferries"
    record = synthetic(("1. Applies to", body))
    texts = [c.text for c in ingest([record], max_words=4, overlap_words=0).children]
    assert texts == ["1. Applies to\n(a) vessels over 20 metres", "(b) vessels carrying cargo",
                     "(c) ferries"]


def test_sentence_splitting_is_a_heuristic_but_offsets_stay_exact() -> None:
    record = synthetic(("1. Rule", "Dr. Lee signs it. See s. 5. (a) for more."))
    kids = ingest([record], max_words=2, overlap_words=0).children
    assert [c.text for c in kids] == ["1. Rule\nDr.", "Lee signs it.", "See s.", "5.",
                                      "(a) for more."]
    assert all(record.text[c.start : c.end] == c.text for c in kids)


def test_offsets_count_code_points_not_bytes() -> None:
    record = synthetic(("§ 1. Prüfung", "Der Rumpf wird geprüft. Das gilt für Schiffe."))
    kids = ingest([record], max_words=4, overlap_words=0).children
    assert [c.text for c in kids] == ["§ 1. Prüfung\nDer Rumpf wird geprüft.",
                                      "Das gilt für Schiffe."]
    second = record.text.index("Das gilt")
    assert kids[1].start == second != record.text.encode().index(b"Das gilt")


def test_indentation_and_trailing_spaces_stay_outside_passages() -> None:
    body = "A vessel must keep a log.  \n    (a) The log is kept aboard.  "
    record = synthetic(("1. Rule", body))
    kids = ingest([record], max_words=6, overlap_words=0).children
    assert [c.text for c in kids] == ["1. Rule\nA vessel must keep a log.",
                                      "(a) The log is kept aboard."]
    assert all(record.text[c.start : c.end] == c.text for c in kids)


# --- stable IDs and repeated ingestion ----------------------------------------


def test_repeated_and_reordered_ingestion_is_identical() -> None:
    first = ingest(RECORDS)
    assert ingest(RECORDS) == first
    assert ingest(reversed(RECORDS)) == first
    assert ingest(iter(RECORDS)) == first
    assert [p.source_id for p in first.parents] == sorted(p.source_id for p in first.parents)


def test_adding_or_removing_records_never_renames_other_passages() -> None:
    full = ingest(RECORDS)
    partial = ingest(r for r in RECORDS if r.source_id != V2)
    assert set(partial.children) < set(full.children)
    assert {c.source_id for c in set(full.children) - set(partial.children)} == {V2}


def test_an_edit_without_a_new_version_keeps_ids_but_not_the_hash() -> None:
    record = next(r for r in RECORDS if r.source_id == V2)
    text = record.text.replace("12 months", "10 months")
    edited = dataclasses.replace(record, text=text, content_hash=content_hash(text))
    old, new = ingest([record]).children[0], ingest([edited]).children[0]
    assert old.chunk_id == new.chunk_id and old.content_hash != new.content_hash


def test_results_are_frozen() -> None:
    result = ingest(RECORDS)
    for obj, name in ((result, "max_words"), (result.parents[0], "start"),
                      (result.children[0], "text")):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, name, 0)


# --- empty and malformed input ------------------------------------------------


def test_an_empty_corpus_gives_an_empty_ingestion() -> None:
    assert ingest([]) == Ingestion(120, 30, (), ())


def test_a_repeated_record_is_rejected() -> None:
    with pytest.raises(ProvenanceError, match="duplicate parent ID"):
        ingest([*RECORDS, RECORDS[0]])


RECORD = next(r for r in RECORDS if r.source_id == V2)
S1 = RECORD.sections[0]


def _with(**fields: Any) -> SourceRecord:
    return dataclasses.replace(RECORD, **fields)


def _section(**fields: Any) -> SourceRecord:
    return _with(sections=(dataclasses.replace(S1, **fields),))


TAMPERED = [
    pytest.param({"source_id": V2}, "expected a SourceRecord, got dict", id="not-a-record"),
    pytest.param(_with(text=RECORD.text + "x"), "does not match its content_hash", id="edited"),
    pytest.param(_with(text="\ud800"), "does not match its content_hash", id="surrogate"),
    pytest.param(_with(sections=()), "non-empty tuple", id="no-sections"),
    pytest.param(_with(sections=[S1]), "non-empty tuple", id="sections-list"),
    pytest.param(_section(start=S1.start + 1), "sections[0] needs", id="mid-line"),
    pytest.param(_section(start=10_000), "sections[0] needs", id="outside-text"),
    pytest.param(_with(sections=(S1, S1)), "sections[1] needs", id="overlapping"),
    pytest.param(_section(heading=""), "sections[0] needs", id="empty-heading"),
    pytest.param(_section(start=True), "sections[0] needs", id="bool-offset"),
    pytest.param(_with(sections=("s1",)), "sections[0] needs", id="not-a-section"),
    pytest.param(_section(section_id="s1:0-3#s2"), "sections[0] needs", id="forged-section-id"),
    pytest.param(_section(section_id="s" + "1" * 100), "sections[0] needs", id="long-section-id"),
    pytest.param(_section(pages=object()), "sections[0] needs", id="pages-object"),
    pytest.param(_section(pages=(2, 1)), "sections[0] needs", id="pages-reversed"),
    pytest.param(_with(source_id="fic-harbour-safety-act@v2#s9"), "source_id is not",
                 id="forged-source-id"),
    pytest.param(_with(source_id="x\n#s1:0-1"), "source_id is not", id="newline-source-id"),
    pytest.param(_with(source_id="fic-other@v2"), "source_id is not", id="id-not-family"),
    pytest.param(_with(source_id="x#s1@v2", family_id="x#s1"), "source_id is not",
                 id="forged-family-id"),
    pytest.param(_with(version="2"), "source_id is not", id="string-version"),
    pytest.param(_with(source_id="a" * 2000 + "@v2", family_id="a" * 2000), "source_id is not",
                 id="long-source-id"),
]


@pytest.mark.parametrize(("bad", "expected"), TAMPERED)
def test_a_tampered_record_fails_closed_with_a_named_problem(bad: object, expected: str) -> None:
    with pytest.raises(ProvenanceError) as info:
        ingest([RECORDS[0], bad])  # type: ignore[list-item]
    (problem,) = info.value.problems
    assert problem.startswith("records[1]") and expected in problem, problem
    assert len(problem) < 200  # field values are not echoed


@pytest.mark.parametrize(
    ("max_words", "overlap_words"),
    [(0, 0), (-1, 0), (10, 10), (10, 6), (1, 1), (10, -1), (True, 0), (10, False),
     (10.0, 2), ("10", 2)],
)
def test_bad_limits_are_rejected(max_words: Any, overlap_words: Any) -> None:
    with pytest.raises(ValueError, match=r"max_words|overlap_words") as info:
        ingest(RECORDS, max_words=max_words, overlap_words=overlap_words)
    assert not isinstance(info.value, ProvenanceError)


# --- the command line ---------------------------------------------------------


def test_cli_lists_every_passage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == SUMMARY and len(lines) == 14
    assert lines[1].split()[:3] == ["fic-basalt-file-note@v1#s1:87-251", "31", "words"]


def test_the_readme_split_command_splits_two_sections(capsys: pytest.CaptureFixture[str]) -> None:
    flags = "--max-words 12 --overlap-words 6"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{flags}` splits two of them in two" in readme
    assert main(flags.split()) == 0
    passages = [line.split()[0] for line in capsys.readouterr().out.splitlines()[1:]]
    assert len(passages) == 15
    split = Counter(p.split(":")[0] for p in passages)
    assert sorted(parent for parent, n in split.items() if n == 2) == [
        "fic-basalt-file-note@v1#s1", "fic-harbour-inspection-guidance@v1#s2"]


def test_cli_rejects_bad_limits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--max-words", "5", "--overlap-words", "5"])
    assert info.value.code == 2 and "overlap_words" in capsys.readouterr().err


def test_cli_reports_a_broken_fixture(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = copy.deepcopy(RAW)
    raw["sources"][0]["sections"][0]["start"] += 1
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "1 problem(s)" in err and "beginning of a line" in err


def test_cli_reports_an_unreadable_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.json")]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_documented_command_runs_from_the_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.legal_rag.ingest"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines()[0] == SUMMARY


def test_readme_shows_the_command_and_its_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"python -m examples.legal_rag.ingest\n```\n\n```\n{out}```" in readme
