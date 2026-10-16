"""The evaluation harness: the metric math, the table, and the pinned numbers.

``extras/eval.py`` turns "hybrid settles it" into a table. The metric
functions are checked on hand-worked rankings so a wrong number can't hide
behind a plausible one, and the measured fixture result is pinned so any
change to tokenising, chunking or fusion that moves it fails here first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from extras.eval import (
    KS,
    MODES,
    ModeResult,
    Query,
    evaluate,
    format_table,
    load_fixture,
    main,
    recall_at_k,
    reciprocal_rank,
    run,
)
from minirag import Document, MiniRAG

REPO = Path(__file__).resolve().parent.parent


def test_recall_at_k_counts_relevant_docs_inside_the_cutoff() -> None:
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, frozenset({"c"}), 2) == 0.0
    assert recall_at_k(ranked, frozenset({"c"}), 3) == 1.0
    assert recall_at_k(ranked, frozenset({"a", "d"}), 3) == 0.5
    assert recall_at_k(ranked, frozenset({"a", "z"}), 10) == 0.5  # k past the list is fine


@pytest.mark.parametrize(("relevant", "k"), [(frozenset({"a"}), 0), (frozenset(), 1)])
def test_recall_at_k_rejects_meaningless_input(relevant: frozenset[str], k: int) -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], relevant, k)


@pytest.mark.parametrize(
    ("ranked", "expected"),
    [(["x", "a"], 0.5), (["a", "x"], 1.0), (["x", "y", "b"], 1 / 3), (["x"], 0.0), ([], 0.0)],
)
def test_reciprocal_rank_uses_the_first_relevant_hit(ranked: list[str], expected: float) -> None:
    assert reciprocal_rank(ranked, frozenset({"a", "b"})) == pytest.approx(expected)


@pytest.fixture(scope="module")
def tiny_engine() -> MiniRAG:
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)
    engine.index(
        [
            Document("europa", "Europa is a moon of Jupiter with an ocean under its ice."),
            Document("io", "Io is the most volcanically active body in the solar system."),
            Document("mars", "Mars has polar caps of water ice and a thin atmosphere."),
        ]
    )
    return engine


def test_evaluate_averages_hits_and_misses(tiny_engine: MiniRAG) -> None:
    # "ocean" finds europa first; "volcanic" never matches "volcanically"
    # (no stemmer), so BM25 returns nothing and that query scores zero.
    queries = [Query("ocean", frozenset({"europa"})), Query("volcanic", frozenset({"io"}))]
    result = evaluate(tiny_engine, queries, "bm25", depth=3, ks=(1, 3))
    assert result == ModeResult(mode="bm25", recall={1: 0.5, 3: 0.5}, mrr=0.5)


def test_evaluate_result_is_read_only(tiny_engine: MiniRAG) -> None:
    queries = [Query("ocean", frozenset({"europa"}))]
    result = evaluate(tiny_engine, queries, "dense", depth=3, ks=(1,))
    with pytest.raises(TypeError):
        result.recall[1] = 0.0  # type: ignore[index]


def test_evaluate_rejects_no_queries_and_k_past_depth(tiny_engine: MiniRAG) -> None:
    with pytest.raises(ValueError, match="at least one query"):
        evaluate(tiny_engine, [], "bm25", depth=3)
    with pytest.raises(ValueError, match="exceeds the ranking depth"):
        evaluate(tiny_engine, [Query("ocean", frozenset({"europa"}))], "bm25", depth=3, ks=(5,))


def test_load_fixture_reads_every_document_and_labelled_query() -> None:
    documents, queries = load_fixture()
    assert len(documents) == 20
    assert len(queries) == 5
    doc_ids = {d.doc_id for d in documents}
    assert all(q.relevant <= doc_ids for q in queries)


def test_measured_numbers_on_the_fixture_corpus() -> None:
    # The output of `python -m extras.eval`: every mode at the ceiling, because
    # the five fixture queries are too easy to separate them. Any mode losing
    # an answer drops a cell below 1.0 and fails here.
    results = run()
    assert tuple(r.mode for r in results) == MODES
    for result in results:
        assert dict(result.recall) == {1: 1.0, 3: 1.0, 5: 1.0}, result.mode
        assert result.mrr == 1.0, result.mode


def test_format_table_renders_one_markdown_row_per_mode() -> None:
    results = [
        ModeResult("bm25", {1: 0.6, 3: 1.0}, 0.8),
        ModeResult("hybrid", {1: 1.0, 3: 1.0}, 1.0),
    ]
    assert format_table(results).splitlines() == [
        "| mode | recall@1 | recall@3 | MRR |",
        "| --- | ---: | ---: | ---: |",
        "| `bm25` | 0.6000 | 1.0000 | 0.8000 |",
        "| `hybrid` | 1.0000 | 1.0000 | 1.0000 |",
    ]


def test_main_prints_the_summary_line_and_a_row_per_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()
    out = capsys.readouterr().out
    assert out.startswith("5 queries over 20 documents (fixture corpus)\n")
    assert all(f"| `{mode}` |" in out for mode in MODES)
    assert f"recall@{KS[-1]}" in out


def test_runs_as_a_module_from_the_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "extras.eval"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "| mode | recall@1 | recall@3 | recall@5 | MRR |" in proc.stdout
