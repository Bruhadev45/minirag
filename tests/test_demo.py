"""The demo makes two claims in prose; these tests pin both of them.

``python -m minirag.demo`` tells the reader that every mode answers the
first query and that every mode misses ``io`` on the second. If a future
change (a stemmer, a real embedder) makes either claim false, the demo's
docstring is now lying to readers -- so these fail rather than the prose
quietly rotting.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from minirag import Document, MiniRAG
from minirag.demo import DOCUMENTS, MODES, QUERIES, main


@pytest.fixture(scope="module")
def demo_engine() -> MiniRAG:
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)
    engine.index([Document(doc_id=doc_id, text=text) for doc_id, text in DOCUMENTS])
    return engine


def test_demo_documents_have_unique_ids() -> None:
    doc_ids = [doc_id for doc_id, _ in DOCUMENTS]
    assert len(doc_ids) == len(set(doc_ids)) == 6


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_answers_the_first_query_with_europa(demo_engine: MiniRAG, mode: str) -> None:
    query = QUERIES[0][0]
    hits = demo_engine.search(query, top_k=3, mode=mode)  # type: ignore[arg-type]
    assert hits[0].doc_id == "europa", f"{mode}: {[h.doc_id for h in hits]}"


@pytest.mark.parametrize("mode", MODES)
def test_no_mode_finds_io_for_the_second_query(demo_engine: MiniRAG, mode: str) -> None:
    query = QUERIES[1][0]
    hits = demo_engine.search(query, top_k=3, mode=mode)  # type: ignore[arg-type]
    assert hits, f"{mode} returned nothing at all"
    assert "io" not in [h.doc_id for h in hits], (
        f"{mode} now finds io -- 'volcanic' matches 'volcanically', so update the demo prose"
    )


def test_main_prints_a_line_per_mode_for_every_query(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"indexed {len(DOCUMENTS)} documents -> 6 chunks"
    for query, note in QUERIES:
        assert f"query: {query!r}" in lines
        assert f"  expect: {note}" in lines
    for mode in MODES:
        ranked = [ln for ln in lines if ln.strip().startswith(mode)]
        assert len(ranked) == len(QUERIES), f"{mode}: {ranked}"
        assert all("1. " in ln for ln in ranked), ranked


def test_the_module_runs_as_a_script() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "minirag.demo"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "europa" in result.stdout
    assert result.stderr == ""
