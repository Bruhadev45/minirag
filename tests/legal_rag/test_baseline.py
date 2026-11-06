"""The hybrid baseline: three rankings of one scope's passages, and nothing else's.

The learned side runs on the fake ``sentence_transformers`` module from
``tests/conftest.py``: these check plumbing, isolation and determinism on
fictional data, not retrieval quality or legal accuracy.
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from examples.legal_rag.baseline import DEPTH, Baseline, compare, main
from examples.legal_rag.eligibility import EligibilityError, Scope, ScopedView, scoped_view, search
from examples.legal_rag.ingest import ingest
from examples.legal_rag.provenance import FIXTURE, NOTICE, content_hash, load, validate_corpus
from extras.embed import HashedTfidfText, MixedSpaceError, SentenceTransformerEmbedder, Space
from minirag.fusion import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[2]
RECORDS = load()
ORCHID = "fic-orchid-fleet-memo@v1"
QUERY = "how often must a vessel pass a hull inspection"
COMMIT = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODES = ("bm25", "dense", "hybrid")
TIE_QUERY = "small vessels guidance status"
SCOPES = {
    "public": Scope("public", None, "fic-varnholt", "maritime-safety", date(2024, 6, 1)),
    "public-2023": Scope("public", None, "fic-varnholt", "maritime-safety", date(2023, 1, 1)),
    "orchid": Scope("tenant-orchid", "matter-orchid-001", "fic-varnholt", "maritime-safety",
                    date(2024, 6, 1)),
    "quill": Scope("public", None, "fic-quillmouth", "maritime-safety", date(2024, 6, 1)),
}
QUERIES = (QUERY, TIE_QUERY, "who keeps the records", "berthing penalty", "the of and")


def intruder(body: str) -> tuple:
    """The fixture plus a valid private record of another tenant whose text is ``body``."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
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


def view(name: str, records=RECORDS) -> ScopedView:
    records = tuple(records)
    return scoped_view(records, ingest(records), SCOPES[name])


def ids(rows) -> list[str]:
    return [row.chunk_id for row in rows]


def learned(fake_st) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(MODEL, revision=COMMIT)


# --- eligibility --------------------------------------------------------------


@pytest.mark.parametrize("name", SCOPES)
@pytest.mark.parametrize("query", QUERIES)
def test_every_candidate_is_a_visible_passage(name: str, query: str) -> None:
    v = view(name)
    baseline = compare(v, query, top_k=5)
    for mode in MODES:
        assert set(ids(getattr(baseline, mode))) <= set(v.children)


@pytest.mark.parametrize("name", ["public", "orchid"])
@pytest.mark.parametrize("query", [QUERY, TIE_QUERY])
def test_hashed_rankings_match_the_core_engine_mode_for_mode(name: str, query: str) -> None:
    v, baseline = view(name), compare(view(name), query, top_k=5)
    for mode in MODES:
        core = search(v, query, top_k=5, mode=mode)
        assert ids(getattr(baseline, mode)) == [r.child.chunk_id for r in core]
        # Scores are rounded to 12 decimals here, and not in the core.
        assert [r.score for r in getattr(baseline, mode)] == pytest.approx(
            [r.score for r in core], rel=0, abs=1e-12)


def test_another_tenants_stuffed_record_moves_no_score_and_no_candidate(fake_st) -> None:
    stuffed = intruder("Every vessel must pass a hull inspection. " * 5 + "How often? Often.")
    for embedder in (None, learned(fake_st)):
        before = compare(view("public"), QUERY, embedder, top_k=10)
        beside = compare(view("public", stuffed), QUERY, embedder, top_k=10)
        assert beside == before
    mallory = scoped_view(stuffed, ingest(stuffed), Scope(
        "tenant-mallory", None, "fic-varnholt", "maritime-safety", date(2024, 6, 1)))
    assert compare(mallory, QUERY, top_k=1).bm25[0].chunk_id.startswith("fic-mallory-memo@v1")


def test_a_hashed_embedders_own_weights_are_never_used() -> None:
    v = view("public")
    own = HashedTfidfText.fit([v.children[cid].text for cid in sorted(v.children)])
    wide = HashedTfidfText.fit([c.text for c in ingest(RECORDS).children])
    with pytest.raises(ValueError, match="does not describe"):  # forged pairing
        HashedTfidfText(own.space, wide.core)
    with pytest.raises(ValueError, match="read-only"):
        own.core.idf[:] = wide.core.idf
    own.core.idf.flags.writeable = True  # a determined caller can still get there
    own.core.idf[:] = wide.core.idf
    assert compare(v, QUERY, own) == compare(v, QUERY)  # refitted on the scope regardless


def test_bm25_scores_equal_up_to_float_noise_tie_on_chunk_id(monkeypatch) -> None:
    v = view("public")
    first, second = sorted(v.children)[:2]
    monkeypatch.setattr("minirag.bm25.search", lambda index, tokens, top_k: (
        (1, 2.0), (0, 2.0 - 4e-16)))
    assert [(c.chunk_id, c.score) for c in compare(v, QUERY).bm25] == [(first, 2.0), (second, 2.0)]


def test_a_hashed_embedder_fitted_beyond_the_scope_is_refused() -> None:
    v = view("public")
    everything = HashedTfidfText.fit([c.text for c in ingest(RECORDS).children])
    with pytest.raises(EligibilityError, match="fitted on this scope's passages"):
        compare(v, QUERY, everything)
    own = HashedTfidfText.fit([v.children[cid].text for cid in sorted(v.children)])
    assert compare(v, QUERY, own) == compare(v, QUERY)
    narrower = HashedTfidfText.fit([v.children[cid].text for cid in sorted(v.children)], dim=512)
    assert compare(v, QUERY, narrower).space.dim == 512  # same texts, another width: allowed
    huge = Space("hashed-tfidf", own.space.model, own.space.revision, 10**10)

    class Claims:  # declares a width the refit would have to allocate
        space = huge

    with pytest.raises(ValueError, match="dim must be an int from 1 to"):
        compare(v, QUERY, Claims())  # type: ignore[arg-type]


def test_one_learned_model_serves_each_scope_from_that_scopes_text_alone(fake_st) -> None:
    embedder = learned(fake_st)
    for name in ("public", "orchid"):
        v = view(name)
        fake_st.built[0].calls.clear()
        compare(v, QUERY, embedder)
        (kind, texts, _), (qkind, query, _) = fake_st.built[0].calls
        assert (kind, qkind, query) == ("document", "query", [QUERY])
        assert texts == [v.children[cid].text for cid in sorted(v.children)]  # raw, unchanged


# --- what the three rankings say ----------------------------------------------


def test_the_learned_side_can_disagree_with_bm25_and_hybrid_fuses_both(fake_st) -> None:
    baseline = compare(view("public"), "how often is a boat surveyed", learned(fake_st), top_k=5)
    assert baseline.space == Space("learned", MODEL, COMMIT, 16) and baseline.space.pinned
    assert baseline.bm25 == ()  # no shared term after stopwords ("how", "is", "a") go
    assert baseline.dense and all(row.bm25_rank is None for row in baseline.dense)
    assert ids(baseline.hybrid) == ids(baseline.dense)


@pytest.mark.parametrize("query", QUERIES[:4])
def test_hybrid_is_the_cores_rrf_over_the_two_lists_with_their_ranks(query: str) -> None:
    v = view("public")
    deep = compare(v, query, top_k=DEPTH)
    baseline = compare(v, query, top_k=3)
    fused = reciprocal_rank_fusion({"bm25": ids(deep.bm25), "dense": ids(deep.dense)}, top_k=3)
    assert [(row.chunk_id, row.score) for row in baseline.hybrid] == list(fused)
    for mode, rank in (("bm25", "bm25_rank"), ("dense", "dense_rank")):
        assert [getattr(row, rank) for row in getattr(deep, mode)] == list(
            range(1, len(getattr(deep, mode)) + 1))
        position = {cid: i for i, cid in enumerate(ids(getattr(deep, mode)), start=1)}
        assert all(getattr(row, rank) == position.get(row.chunk_id) for row in baseline.hybrid)


def test_an_exact_rrf_tie_breaks_on_best_rank_then_bm25_first() -> None:
    baseline = compare(view("public"), TIE_QUERY, top_k=2)
    first, second = baseline.hybrid
    assert first.score == second.score == 1 / 61 + 1 / 62
    assert (first.bm25_rank, first.dense_rank, second.bm25_rank, second.dense_rank) == (1, 2, 2, 1)
    assert first.chunk_id == baseline.bm25[0].chunk_id


def test_dense_ties_break_on_chunk_id(fake_st, monkeypatch) -> None:
    monkeypatch.setattr(fake_st, "_encode", lambda self, kind, texts, kw: [[1.0] * 16] * len(texts))
    v = view("public")
    baseline = compare(v, QUERY, learned(fake_st), top_k=5)
    assert ids(baseline.dense) == sorted(v.children)


def test_results_repeat_exactly_and_weights_shift_the_fusion() -> None:
    v = view("public")
    assert compare(v, TIE_QUERY) == compare(v, TIE_QUERY)
    dense_first = compare(v, TIE_QUERY, weights={"bm25": 1.0, "dense": 2.0}, top_k=1)
    assert ids(dense_first.hybrid) == ids(compare(v, TIE_QUERY).dense)[:1]


@pytest.mark.parametrize("query", ["", "   \n", "the of and"])
def test_blank_and_stopword_queries_find_nothing_lexically(query: str) -> None:
    baseline = compare(view("public"), query)
    assert (baseline.bm25, baseline.dense, baseline.hybrid) == ((), (), ())
    assert baseline.space.kind == "hashed-tfidf"


def test_a_stopword_query_still_reaches_a_learned_model(fake_st) -> None:
    baseline = compare(view("public"), "the of and", learned(fake_st))
    assert baseline.bm25 == () and isinstance(baseline, Baseline)
    assert fake_st.built[0].calls[-1][:2] == ("query", ["the of and"])


def test_a_scope_with_no_passages_returns_empty_lists_without_embedding(fake_st) -> None:
    empty = view("quill")
    assert not empty.children
    baseline = compare(empty, QUERY, learned(fake_st))
    assert (baseline.bm25, baseline.dense, baseline.hybrid) == ((), (), ())
    assert not fake_st.built[0].calls


# --- arguments ----------------------------------------------------------------


@pytest.mark.parametrize(("kwargs", "error"), [
    ({"top_k": 0}, ValueError), ({"top_k": True}, ValueError), ({"top_k": 21}, ValueError),
    ({"top_k": 3, "depth": 2}, ValueError), ({"depth": 5.0}, ValueError),
    ({"weights": {"bm25": float("nan")}}, ValueError), ({"weights": {"dense": -1.0}}, ValueError),
    ({"weights": {"dense": 0}}, ValueError),
    ({"weights": {"rerank": 1.0}}, ValueError), ({"weights": {"bm25": True}}, ValueError),
    ({"weights": {"bm25": "1"}}, ValueError), ({"rrf_k": 0}, ValueError),
    ({"rrf_k": 60.0}, ValueError),
])
def test_compare_checks_its_arguments(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        compare(view("public"), QUERY, **kwargs)


def test_compare_needs_a_view_a_string_and_a_named_space() -> None:
    v = view("public")
    with pytest.raises(TypeError, match="ScopedView"):
        compare(object(), QUERY)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="query"):
        compare(v, b"query")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Space"):
        compare(v, QUERY, object())  # type: ignore[arg-type]


def test_mixed_spaces_cannot_meet_inside_compare(fake_st, monkeypatch) -> None:
    embedder = learned(fake_st)
    monkeypatch.setattr(SentenceTransformerEmbedder, "space", property(
        lambda self: Space("learned", MODEL, "main" if self._model.calls else COMMIT, 16)))
    with pytest.raises(MixedSpaceError):  # the space changed between build and query
        compare(view("public"), QUERY, embedder)


# --- the command line -----------------------------------------------------------

ARGS = [TIE_QUERY, "--tenant", "public", "--jurisdiction", "fic-varnholt",
        "--regime", "maritime-safety", "--as-of", "2024-06-01"]


def test_readme_shows_the_command_and_its_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(ARGS) == 0
    out = capsys.readouterr().out
    command = (f'python -m examples.legal_rag.baseline "{TIE_QUERY}" \\\n'
               "    --tenant public --jurisdiction fic-varnholt --regime maritime-safety "
               "--as-of 2024-06-01\n```\n\n```\n")
    assert command + out + "```" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "hashed TF-IDF (lexical, not a learned embedding)" in out and ORCHID not in out


def test_the_cli_runs_a_learned_model_only_when_asked(fake_st, capsys) -> None:
    assert main([*ARGS, "--model", MODEL, "--revision", "main", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"dense side: learned embedding: {MODEL} @ main, 16 dims\n")
    assert "warning: the revision is not a commit hash" in out
    assert fake_st.built[0].kwargs["local_files_only"] is True
    assert main([*ARGS, "--model", MODEL, "--revision", COMMIT, "--allow-download"]) == 0
    assert "warning" not in capsys.readouterr().out
    assert fake_st.built[1].kwargs["local_files_only"] is False


def test_the_cli_prints_no_match_for_an_empty_ranking(capsys) -> None:
    assert main(["the of and", *ARGS[1:]]) == 0
    assert capsys.readouterr().out.count("(no match)") == 3


@pytest.mark.parametrize("flags", [["--model", MODEL], ["--revision", COMMIT], ["--top-k", "0"],
                                   ["--top-k", "21"], ["--tenant", "tenant-unknown"]])
def test_the_cli_refuses_bad_flags(flags: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as info:
        main([*ARGS, *flags])
    assert info.value.code == 2


def test_the_cli_reports_a_missing_package_or_model(monkeypatch, fake_st, tmp_path, capsys) -> None:
    def missing(self, *args, **kwargs):
        raise OSError("not in the local cache")

    monkeypatch.setattr(fake_st, "__init__", missing)
    assert main([*ARGS, "--model", MODEL, "--revision", COMMIT]) == 1
    assert "not in the local cache" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert main([*ARGS, "--model", MODEL, "--revision", COMMIT]) == 1
    assert ".[embeddings]" in capsys.readouterr().err
    assert main([*ARGS, "--fixture", str(tmp_path / "missing.json")]) == 1
    assert "cannot load" in capsys.readouterr().err


def test_the_cli_reports_a_model_that_fails_while_loading_or_ranking(monkeypatch, fake_st,
                                                                    capsys) -> None:
    def corrupt(self, *args, **kwargs):
        raise RuntimeError("bad safetensors header")

    monkeypatch.setattr(fake_st, "get_embedding_dimension", corrupt)
    assert main([*ARGS, "--model", MODEL, "--revision", COMMIT]) == 1
    assert "cannot load model" in capsys.readouterr().err
    monkeypatch.setattr(fake_st, "get_embedding_dimension", lambda self: 16)
    monkeypatch.setattr(fake_st, "_encode", lambda self, kind, texts, kw: [[np.nan] * 16] * len(
        texts))
    assert main([*ARGS, "--model", MODEL, "--revision", COMMIT]) == 1
    assert "cannot rank: ValueError" in capsys.readouterr().err
