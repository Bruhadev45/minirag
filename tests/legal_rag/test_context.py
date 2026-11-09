"""Parent retrieval and context packing.

Every packed block must be an exact quote of an eligible source version, the
whole context must fit the declared budget as counted, and every ranked passage
must be either packed or listed as omitted with a reason. Token counts here
come from a character estimate or a test counter, never a model tokenizer.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import date
from pathlib import Path
from types import MappingProxyType

import pytest

from examples.legal_rag.baseline import compare
from examples.legal_rag.context import (
    SEPARATOR,
    CharEstimate,
    Evidence,
    Omitted,
    Packed,
    main,
    pack,
)
from examples.legal_rag.eligibility import (
    NOT_AVAILABLE,
    EligibilityError,
    NotAvailable,
    Scope,
    ScopedView,
    scoped_view,
    search,
)
from examples.legal_rag.ingest import ingest
from examples.legal_rag.provenance import (
    FIXTURE,
    MAX_INT,
    NOTICE,
    content_hash,
    load,
    validate_corpus,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = json.loads(FIXTURE.read_text(encoding="utf-8"))
RECORDS = load()
ACT1, ACT2 = "fic-harbour-safety-act@v1", "fic-harbour-safety-act@v2"
GUIDANCE, ORCHID = "fic-harbour-inspection-guidance@v1", "fic-orchid-fleet-memo@v1"
QUERY = "how often must a vessel pass a hull inspection"
PUBLIC = Scope("public", None, "fic-varnholt", "maritime-safety", date(2024, 6, 1))


class Chars:
    """A test counter that calls each character a token and claims to be exact."""

    label = "test: one token per character"
    exact = True

    def count(self, text: str) -> int:
        return len(text)


def view(s: Scope = PUBLIC, records=RECORDS, **limits: int) -> ScopedView:
    records = tuple(records)
    return scoped_view(records, ingest(records, **limits), s)


def ranked(v: ScopedView, query: str = QUERY, top_k: int = 5) -> list[str]:
    return [hit.child.chunk_id for hit in search(v, query, top_k=top_k)]


def public_record(family: str, *sections: tuple[str, str]) -> tuple:
    """The fixture plus one valid public record of ``family`` with ``(heading, body)`` sections."""
    raw = copy.deepcopy(RAW)
    record = copy.deepcopy(next(s for s in raw["sources"] if s["source_id"] == GUIDANCE))
    text, spans = f"{NOTICE}\n{family} (fictional)\n", []
    for i, (heading, body) in enumerate(sections, start=1):
        text += "\n"
        start = len(text)
        text += f"{heading}\n{body}\n"
        spans.append({"section_id": f"s{i}", "heading": heading, "start": start,
                      "end": len(text), "pages": [i, i + 1]})
    record.update(source_id=f"{family}@v1", family_id=family, title=f"{family} (fictional)",
                  text=text, content_hash=content_hash(text), sections=spans)
    raw["sources"].append(record)
    return validate_corpus(raw)


def clauses(n: int) -> str:
    """``n`` distinct five-word sentences on one line."""
    return " ".join(f"Clause {i} sets rule {'x' * (i + 1)}." for i in range(n))


LONG = public_record("fic-long-code", ("1. Rules", clauses(8)), ("2. More", clauses(3)))
LONG_ID = "fic-long-code@v1"


def long_view() -> ScopedView:
    """Overlapping children: ten-word windows of five-word sentences, five words repeated."""
    return view(records=LONG, max_words=10, overlap_words=5)


def kids(v: ScopedView, parent_id: str) -> list[str]:
    return list(v.parents[parent_id].child_ids)


def check(v: ScopedView, ids: list[str], packed: Packed, counter=None) -> None:
    """The invariants every packing must keep, whatever the budget."""
    counter = counter or CharEstimate()
    covered = [cid for b in packed.evidence for cid in b.chunk_ids]
    covered += [cid for o in packed.omitted for cid in o.chunk_ids]
    assert sorted(covered) == sorted(ids)  # every ranked passage packed or listed, once
    assert packed.text == SEPARATOR.join(b.render() for b in packed.evidence)
    assert packed.tokens == counter.count(packed.text) <= packed.budget
    assert [b.ref for b in packed.evidence] == list(range(1, len(packed.evidence) + 1))
    for b in packed.evidence:
        record = v.sources[b.source_id]
        assert b.text == record.text[b.start : b.end] == b.text.strip()
        assert (b.family_id, b.version, b.content_hash) == (
            record.family_id, record.version, record.content_hash)
        assert b.tokens == counter.count(b.render())
        assert list(b.ranks) == sorted(b.ranks) == [ids.index(cid) + 1 for cid in b.chunk_ids]
        children = [v.children[cid] for cid in b.chunk_ids]
        assert all(c.parent_id == b.parent_id for c in children)
        if b.granularity == "excerpt":  # exactly the union of its ranked children
            spans = sorted((c.start, c.end) for c in children)
            assert (b.start, b.end) == (spans[0][0], max(end for _, end in spans))
            reach = spans[0][1]
            for start, end in spans[1:]:  # bridged only where they share text
                assert start < reach
                reach = max(reach, end)
        else:
            parent = v.parents[b.parent_id]
            assert record.text[parent.start : parent.end].strip() == b.text


# --- parents, versions and the budget -------------------------------------------


def test_ranked_passages_become_their_whole_sections() -> None:
    v = view()
    ids = ranked(v)
    packed = pack(v, ids, budget=1000, max_per_family=3)
    check(v, ids, packed)
    assert [b.parent_id for b in packed.evidence] == [v.children[c].parent_id for c in ids]
    assert {b.granularity for b in packed.evidence} == {"section"} and packed.omitted == ()
    assert packed.evidence[0].render() == (
        f"[1] {ACT2} s1, p. 1, chars 94-232\n1. Inspections\nA vessel longer than 20 metres "
        "that berths in a Varnholt harbour must pass a hull inspection at least once every "
        "12 months.")


def test_children_of_one_section_collapse_into_it() -> None:
    v = long_view()
    ids = kids(v, f"{LONG_ID}#s1")
    assert len(ids) > 2
    packed = pack(v, ids[::-1], budget=1000)
    (block,) = packed.evidence
    assert block.granularity == "section" and block.chunk_ids == tuple(ids[::-1])
    assert block.ranks == tuple(range(1, len(ids) + 1))
    check(v, ids[::-1], packed)


def test_overlapping_excerpts_merge_and_stay_exact() -> None:
    v = long_view()
    k, c, text = kids(v, f"{LONG_ID}#s1"), v.children, v.sources[LONG_ID].text
    assert c[k[2]].start < c[k[1]].end < c[k[3]].end  # these windows overlap
    assert text[c[k[0]].end : c[k[1]].start] == " "  # these only sit side by side
    ids = [k[2], k[1], k[3], k[6], k[0]]
    section = pack(v, ids, budget=1000).evidence[0]
    assert section.granularity == "section"
    packed = pack(v, ids, budget=section.tokens - 1)
    check(v, ids, packed)
    merged = packed.evidence[0]
    assert merged.granularity == "excerpt" and merged.chunk_ids == (k[2], k[1], k[3])
    assert merged.ranks == (1, 2, 3)
    assert (merged.start, merged.end) == (c[k[1]].start, c[k[3]].end)
    assert merged.header() == f"[1] {LONG_ID} excerpt of s1, pp. 1-2, chars 109-206"
    assert all(k[0] not in b.chunk_ids or b.chunk_ids == (k[0],) for b in packed.evidence)


def test_side_by_side_excerpts_stay_separate_so_the_best_one_can_fit() -> None:
    v = long_view()
    first, second = kids(v, f"{LONG_ID}#s2")
    section = pack(v, [second, first], budget=1000).evidence[0]
    packed = pack(v, [second, first], budget=section.tokens - 1)
    check(v, [second, first], packed)
    assert packed.evidence[0].chunk_ids == (second,)
    assert {b.granularity for b in packed.evidence} == {"excerpt"}


def test_a_section_that_is_its_one_passage_is_not_retried_as_an_excerpt() -> None:
    v = view()
    (only,) = kids(v, f"{ACT2}#s1")
    section = pack(v, [only], budget=1000).evidence[0]
    assert (section.start, section.end) == (v.children[only].start, v.children[only].end)
    assert pack(v, [only], budget=section.tokens - 1).omitted == (
        Omitted(f"{ACT2}#s1", (only,), (1,), "oversized", section.tokens),)


def test_identical_words_in_another_source_are_not_deduplicated() -> None:
    act = next(r for r in RECORDS if r.source_id == ACT2)
    heading, body = act.text[94:232].split("\n")
    records = public_record("fic-harbour-copy", (heading, body))
    v = view(records=records)
    copy_child = kids(v, "fic-harbour-copy@v1#s1")[0]
    ids = [kids(v, f"{ACT2}#s1")[0], copy_child]
    packed = pack(v, ids, budget=1000)
    a, b = packed.evidence
    assert a.text == b.text and (a.source_id, b.source_id) == (ACT2, "fic-harbour-copy@v1")
    assert a.content_hash != b.content_hash
    check(v, ids, packed)


def test_each_scope_packs_the_version_in_force() -> None:
    for as_of, source, months in ((date(2020, 1, 1), ACT1, 24), (date(2024, 6, 1), ACT2, 12)):
        v = view(dataclasses.replace(PUBLIC, as_of=as_of))
        (block,) = pack(v, ranked(v, top_k=1), budget=1000).evidence
        assert (block.source_id, block.version) == (source, int(source[-1]))
        assert f"every {months} months" in block.text


def test_the_budget_covers_headers_and_separators_exactly() -> None:
    v = view()
    one = [kids(v, f"{ACT2}#s1")[0]]
    size = len(pack(v, one, budget=MAX_INT, counter=Chars()).evidence[0].render())
    assert pack(v, one, budget=size, counter=Chars()).tokens == size
    tight = pack(v, one, budget=size - 1, counter=Chars())
    assert tight.evidence == () and tight.omitted == (
        Omitted(f"{ACT2}#s1", tuple(one), (1,), "oversized", size),)
    two = [*one, kids(v, f"{ACT2}#s2")[0]]
    full = pack(v, two, budget=MAX_INT, counter=Chars())
    assert full.tokens == len(full.text) == sum(b.tokens for b in full.evidence) + len(SEPARATOR)
    short = pack(v, two, budget=full.tokens - 1, counter=Chars())
    assert [o.reason for o in short.omitted] == ["over-budget"]
    assert short.exact and short.counter == Chars.label


def test_a_later_smaller_block_still_fits_after_one_that_does_not() -> None:
    v = view()
    ids = [kids(v, f"{ACT2}#s{n}")[0] for n in (1, 2, 3)]
    sizes = [pack(v, [cid], budget=MAX_INT).evidence[0].tokens for cid in ids]
    packed = pack(v, ids, budget=sizes[0] + sizes[2] + 1, max_per_family=3)
    assert [b.parent_id for b in packed.evidence] == [f"{ACT2}#s1", f"{ACT2}#s3"]
    assert [(o.parent_id, o.reason) for o in packed.omitted] == [(f"{ACT2}#s2", "over-budget")]


def test_nothing_is_ever_cut_whatever_the_budget() -> None:
    for v, ids in ((view(), ranked(view())), (long_view(), [*kids(long_view(), f"{LONG_ID}#s2"),
                                                            *kids(long_view(), f"{LONG_ID}#s1")])):
        for budget in range(1, 420, 3):
            for cap in (1, 2):
                check(v, ids, pack(v, ids, budget=budget, max_per_family=cap))


def test_a_tiny_budget_packs_nothing_and_says_why() -> None:
    v = view()
    ids = ranked(v)
    packed = pack(v, ids, budget=5)
    assert packed.evidence == () and packed.text == "" and packed.tokens == 0
    assert {o.reason for o in packed.omitted} == {"oversized"}
    assert all(o.tokens > 5 for o in packed.omitted)


def test_the_family_cap_keeps_room_for_other_sources() -> None:
    v = view()
    ids = [kids(v, f"{ACT2}#s{n}")[0] for n in (1, 2, 3)] + [kids(v, f"{GUIDANCE}#s1")[0]]
    packed = pack(v, ids, budget=1000)
    assert [b.parent_id for b in packed.evidence] == [f"{ACT2}#s1", f"{ACT2}#s2",
                                                      f"{GUIDANCE}#s1"]
    assert packed.omitted == (Omitted(f"{ACT2}#s3", (ids[2],), (3,), "family-cap", None),)
    one = pack(v, ids, budget=1000, max_per_family=1)
    assert [o.reason for o in one.omitted] == ["family-cap", "family-cap"]


def test_a_section_left_out_for_budget_does_not_use_the_family_cap() -> None:
    v = view()
    big, small = kids(v, f"{ACT2}#s1")[0], kids(v, f"{ACT2}#s3")[0]
    size = pack(v, [small], budget=MAX_INT).tokens
    packed = pack(v, [big, small], budget=size, max_per_family=1)
    assert [b.parent_id for b in packed.evidence] == [f"{ACT2}#s3"]
    assert packed.omitted[0].reason == "oversized"


def test_any_ranking_of_the_view_can_be_packed() -> None:
    v = view()
    hybrid = [row.chunk_id for row in compare(v, QUERY).hybrid]
    check(v, hybrid, pack(v, hybrid, budget=150))
    assert pack(v, [], budget=1) == Packed(1, CharEstimate().label, False, 0, (), (), "")


def test_packing_is_deterministic_and_frozen() -> None:
    v = view()
    ids = ranked(v)
    packed = pack(v, ids, budget=120)
    assert packed == pack(v, list(ids), budget=120)
    with pytest.raises(dataclasses.FrozenInstanceError):
        packed.evidence[0].text = "forged"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        packed.tokens = 0  # type: ignore[misc]


@pytest.mark.parametrize(("pages", "label"), [((3, 3), "p. 3"), ((2, 4), "pp. 2-4"),
                                              ("unknown", "pages unknown")])
def test_headers_name_the_version_section_pages_and_span(pages, label) -> None:
    block = Evidence(1, "section", ACT2, "fic-harbour-safety-act", 2, f"{ACT2}#s1", "s1", pages,
                     94, 232, "sha256:" + "0" * 64, ("x",), (1,), 1, "text")
    assert block.header() == f"[1] {ACT2} s1, {label}, chars 94-232"


def test_token_counts_stay_labelled_as_estimates() -> None:
    estimate = CharEstimate()
    assert (estimate.count(""), estimate.count("abcd"), estimate.count("abcde")) == (0, 1, 2)
    assert estimate.exact is False and "not a model tokenizer" in estimate.label
    assert "chars / 3" in CharEstimate(3).label
    packed = pack(view(), ranked(view()), budget=200)
    assert (packed.counter, packed.exact) == (estimate.label, False)
    for bad in (0, -1, 2.5, True, "4"):
        with pytest.raises(ValueError):
            CharEstimate(bad)  # type: ignore[arg-type]


# --- what the view cannot show fails closed --------------------------------------


def hand_built(v: ScopedView, **tables: dict) -> ScopedView:
    fields = {"sources": dict(v.sources), "parents": dict(v.parents),
              "children": dict(v.children)}
    fields.update(tables)
    return ScopedView(v.scope, **{k: MappingProxyType(t) for k, t in fields.items()})


def test_passages_outside_the_scope_are_not_available() -> None:
    v = view()
    everything = ingest(RECORDS)
    orchid = next(c.chunk_id for c in everything.children if c.source_id == ORCHID)
    superseded = next(c.chunk_id for c in everything.children if c.source_id == ACT1)
    ok = ranked(v)[0]
    for leak in (orchid, superseded, "no-such-passage", f"{ACT2}#s1", ACT2, None, 3, b"x"):
        with pytest.raises(NotAvailable) as info:
            pack(v, [ok, leak], budget=1000)
        assert str(info.value) == NOT_AVAILABLE
    with pytest.raises(NotAvailable):  # a leak is reported before a repeat
        pack(v, [ok, ok, orchid], budget=1000)


def test_a_missing_parent_or_source_is_not_available() -> None:
    v = view()
    cid = ranked(v)[0]
    parent_id = v.children[cid].parent_id
    no_parent = {k: p for k, p in v.parents.items() if k != parent_id}
    no_source = {k: r for k, r in v.sources.items() if k != ACT2}
    for forged in (hand_built(v, parents=no_parent), hand_built(v, sources=no_source)):
        with pytest.raises(NotAvailable):
            pack(forged, [cid], budget=1000)


def test_a_passage_that_does_not_match_its_record_fails_closed() -> None:
    v = view()
    cid = ranked(v)[0]
    child, parent = v.children[cid], v.parents[v.children[cid].parent_id]
    other = v.parents[f"{GUIDANCE}#s1"]
    forgeries = [
        {"children": {cid: dataclasses.replace(child, text=child.text.replace("12", "99"))}},
        {"children": {cid: dataclasses.replace(child, start=child.start - 1,
                                               text=" " + child.text)}},
        {"children": {cid: dataclasses.replace(child, content_hash=content_hash("x"))}},
        {"parents": {parent.parent_id: dataclasses.replace(parent, end=parent.start + 5,
                                                           text=parent.text[:5])}},
        {"parents": {parent.parent_id: other}},
        {"parents": {parent.parent_id: child}},
    ]
    for tables in forgeries:
        merged = {name: {**getattr(v, name), **tables.get(name, {})}
                  for name in ("sources", "parents", "children")}
        with pytest.raises(EligibilityError):
            pack(hand_built(v, **merged), [cid], budget=1000)


@pytest.mark.parametrize(("kwargs", "error"), [
    ({"budget": 0}, ValueError), ({"budget": -1}, ValueError), ({"budget": True}, ValueError),
    ({"budget": 1.5}, ValueError), ({"budget": MAX_INT + 1}, ValueError),
    ({"max_per_family": 0}, ValueError), ({"max_per_family": None}, ValueError),
    ({"counter": object()}, TypeError),
    ({"counter": dataclasses.make_dataclass("C", [("label", str), ("exact", object)])(
        "", False)}, TypeError),
])
def test_bad_arguments_are_refused(kwargs: dict, error: type[Exception]) -> None:
    v = view()
    with pytest.raises(error):
        pack(v, ranked(v), **{"budget": 100, **kwargs})


class Broken(Chars):
    def __init__(self, result: object) -> None:
        self.result = result

    def count(self, text: str) -> int:
        return self.result  # type: ignore[return-value]


@pytest.mark.parametrize("result", [-1, 1.0, True, None])
def test_a_counter_that_returns_nonsense_is_refused(result: object) -> None:
    v = view()
    with pytest.raises(ValueError, match="int >= 0"):
        pack(v, ranked(v), budget=100, counter=Broken(result))


def test_bad_rankings_are_refused_without_echoing_ids() -> None:
    v = view()
    ids = ranked(v)
    with pytest.raises(TypeError):
        pack(object(), ids, budget=100)  # type: ignore[arg-type]
    for shape in (ids[0], set(ids), iter(ids), None):
        with pytest.raises(TypeError):
            pack(v, shape, budget=100)  # type: ignore[arg-type]
    with pytest.raises(ValueError) as info:
        pack(v, [ids[0], ids[1], ids[0]], budget=100)
    assert str(info.value) == "ranked[2] repeats an earlier chunk ID"
    with pytest.raises(ValueError, match="more IDs than the view"):
        pack(v, ids * 3, budget=100)


# --- the command line -------------------------------------------------------------

README_QUERY = "how does the guidance relate to the hull inspection rule in the Harbour Safety Act"
README_ARGS = [README_QUERY, "--tenant", "public", "--jurisdiction", "fic-varnholt",
               "--regime", "maritime-safety", "--as-of", "2024-06-01", "--budget", "40",
               "--max-per-family", "1", "--max-words", "12", "--overlap-words", "6"]


def test_readme_shows_the_command_and_its_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(README_ARGS) == 0
    out = capsys.readouterr().out
    command = (f'python -m examples.legal_rag.context \\\n    "{README_QUERY}" \\\n'
               "    --tenant public --jurisdiction fic-varnholt --regime maritime-safety "
               "--as-of 2024-06-01 \\\n    --budget 40 --max-per-family 1 --max-words 12 "
               "--overlap-words 6\n```\n\n```\n")
    assert command + out + "```" in (ROOT / "README.md").read_text(encoding="utf-8")
    for reason in ("excerpt", "over-budget", "family-cap", "oversized"):
        assert reason in out
    assert ORCHID not in out


def test_the_cli_refuses_bad_flags_and_reports_a_missing_fixture(tmp_path, capsys) -> None:
    for flags in (["--budget", "0"], ["--tenant", "tenant-unknown"], ["--top-k", "0"],
                  ["--max-per-family", "0"]):
        with pytest.raises(SystemExit) as info:
            main([*README_ARGS, *flags])
        assert info.value.code == 2
    assert main([*README_ARGS, "--fixture", str(tmp_path / "missing.json")]) == 1
    assert "cannot load" in capsys.readouterr().err


def test_the_cli_prints_none_when_nothing_is_omitted(capsys) -> None:
    assert main([QUERY, *README_ARGS[1:9], "--budget", "1000", "--max-per-family", "3"]) == 0
    assert "omitted 0: none" in capsys.readouterr().out
