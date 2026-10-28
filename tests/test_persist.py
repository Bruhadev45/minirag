"""Persistence: identical search after a round trip, and loud failure on anything else.

``extras/persist.py`` promises that a loaded engine answers every query
exactly as the saved one did, and that a file it cannot vouch for -- another
version, a pair from two saves, a tampered field -- raises
:class:`PersistError` instead of serving different results quietly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import extras.persist as persist
from extras.persist import FORMAT_VERSION, PersistError, load, save
from minirag import Document, MiniRAG, tokenize
from minirag.vectors import HashedTfidfEmbedder

QUERIES = (
    "which moon has an ocean under the ice",
    "red storm",
    "rings of ice and rock",
    "the who",  # all stopwords: every mode returns ()
    "zzyzx quasar",  # out of vocabulary
)


@pytest.fixture(scope="module")
def fixture_engine(documents: tuple[Document, ...]) -> MiniRAG:
    rag = MiniRAG(max_tokens=60, overlap_tokens=20)
    rag.index(documents)
    return rag


@pytest.fixture
def saved(fixture_engine: MiniRAG, tmp_path: Path) -> Path:
    return save(fixture_engine, tmp_path / "index.npz")[0]


def _sidecar(npz_path: Path) -> dict[str, Any]:
    return json.loads(npz_path.with_suffix(".json").read_text(encoding="utf-8"))


def _rewrite_sidecar(npz_path: Path, header: dict[str, Any]) -> None:
    npz_path.with_suffix(".json").write_text(json.dumps(header), encoding="utf-8")


def _rewrite_arrays(npz_path: Path, *, compress: bool = False, **arrays: Any) -> None:
    """Replace the ``.npz`` and re-sign the sidecar, as a deliberate tamperer would."""
    buffer = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buffer, **arrays)
    npz_path.write_bytes(buffer.getvalue())
    header = _sidecar(npz_path)
    header["npz_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
    _rewrite_sidecar(npz_path, header)


def _assert_same_answers(a: MiniRAG, b: MiniRAG, queries: tuple[str, ...], **where: str) -> None:
    found = 0
    for query in queries:
        for mode in ("bm25", "dense", "hybrid"):
            for top_k in (1, 5, 50):
                args = {"top_k": top_k, "mode": mode, "where": where or None}
                hits = a.search(query, **args)
                assert hits == b.search(query, **args), (query, mode, top_k)
                found += len(hits)
    assert found, "every search came back empty, so nothing was compared"


def test_loaded_engine_answers_every_query_identically(
    fixture_engine: MiniRAG, saved: Path, query_cases: tuple[tuple[str, str], ...]
) -> None:
    loaded = load(saved)
    queries = QUERIES + tuple(query for query, _ in query_cases)
    assert loaded.chunk_count == fixture_engine.chunk_count == 40
    _assert_same_answers(fixture_engine, loaded, queries)
    _assert_same_answers(fixture_engine, loaded, queries, title="Europa")


def test_loaded_state_matches_the_saved_state_exactly(fixture_engine: MiniRAG, saved: Path) -> None:
    original, restored = fixture_engine._state, load(saved)._state
    assert original is not None and restored is not None
    assert restored.chunks == original.chunks
    assert restored.metadata == original.metadata
    assert restored.lexical == original.lexical  # rebuilt from text, equal to the original
    assert np.array_equal(restored.dense.matrix, original.dense.matrix)
    assert np.array_equal(restored.embedder.idf, original.embedder.idf)
    assert not restored.dense.matrix.flags.writeable
    assert not restored.embedder.idf.flags.writeable


def test_settings_metadata_and_chunkless_documents_survive(tmp_path: Path) -> None:
    settings = {"max_tokens": 12, "overlap_tokens": 4, "dim": 64, "rrf_k": 7}
    weights = {"bm25": 2.5, "dense": 0.5}
    rag = MiniRAG(**settings, weights=weights)
    rag.index(
        [
            Document("café", "Crème brûlée is a custard. It has burnt sugar.", {"lang": "fr"}),
            Document("blank", "   ", {"note": "no chunks at all"}),
            Document("tea", "Tea is brewed from leaves. Sugar is optional in tea."),
            Document("odd", "Lone surrogate \udcff survives.", {"k": "\udcff"}),
            Document("stop", "It is what it is."),  # all stopwords: an all-zero vector
        ]
    )
    loaded = load(save(rag, tmp_path / "small.npz")[0])
    assert {**loaded._window, "dim": loaded._dim, "rrf_k": loaded._rrf_k} == settings
    assert loaded._weights == weights
    assert loaded._state is not None and rag._state is not None
    assert loaded._state.metadata == rag._state.metadata
    assert list(loaded._state.metadata) == ["café", "blank", "tea", "odd", "stop"]
    assert not loaded._state.dense.matrix[-1].any()
    _assert_same_answers(rag, loaded, ("sugar custard", "tea leaves", "surrogate"))


def test_an_empty_index_round_trips(tmp_path: Path) -> None:
    rag = MiniRAG()
    rag.index([])
    loaded = load(save(rag, tmp_path / "empty.npz")[0])
    assert loaded.chunk_count == 0
    assert loaded.search("anything") == ()


def test_save_writes_the_pair_and_leaves_no_temporary_files(
    fixture_engine: MiniRAG, tmp_path: Path
) -> None:
    npz_path, json_path = save(fixture_engine, str(tmp_path / "index.npz"))
    assert (npz_path, json_path) == (tmp_path / "index.npz", tmp_path / "index.json")
    header = _sidecar(npz_path)
    assert header["format"] == "minirag-index" and header["version"] == FORMAT_VERSION
    assert header["npz_sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(npz_path) as archive:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["index.json", "index.npz"]
    save(fixture_engine, npz_path)  # replacing an existing pair is fine
    assert load(npz_path).chunk_count == fixture_engine.chunk_count


def _fail_replace_after(monkeypatch: pytest.MonkeyPatch, successes: int) -> None:
    real_replace, calls = os.replace, []

    def replace(src: str, dst: str) -> None:
        calls.append(dst)
        if len(calls) > successes:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(persist.os, "replace", replace)


def test_a_failed_first_write_keeps_the_previous_pair_and_cleans_up(
    fixture_engine: MiniRAG, saved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = (saved.read_bytes(), saved.with_suffix(".json").read_bytes())
    _fail_replace_after(monkeypatch, successes=0)
    with pytest.raises(OSError, match="disk full"):
        save(fixture_engine, saved)
    assert (saved.read_bytes(), saved.with_suffix(".json").read_bytes()) == before
    assert sorted(p.name for p in saved.parent.iterdir()) == ["index.json", "index.npz"]
    monkeypatch.undo()
    assert load(saved).chunk_count == fixture_engine.chunk_count


def test_a_crash_between_the_two_writes_is_caught_not_served(
    saved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = MiniRAG(dim=64)
    other.index([Document("x", "A different corpus entirely.")])
    _fail_replace_after(monkeypatch, successes=1)  # new arrays land, the sidecar does not
    with pytest.raises(OSError, match="disk full"):
        save(other, saved)
    monkeypatch.undo()
    assert sorted(p.name for p in saved.parent.iterdir()) == ["index.json", "index.npz"]
    with pytest.raises(PersistError, match="does not match"):
        load(saved)


def test_missing_files_raise_oserror(saved: Path) -> None:
    saved.unlink()
    with pytest.raises(FileNotFoundError):
        load(saved)
    saved.with_suffix(".json").unlink()
    with pytest.raises(FileNotFoundError):
        load(saved)


def test_save_refuses_what_it_cannot_write_faithfully(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="index"):
        save(MiniRAG(), tmp_path / "index.npz")
    rag = MiniRAG()
    rag.index([Document("a", "Alpha text here.", {"year": 2024})])  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="str to str"):
        save(rag, tmp_path / "index.npz")
    assert list(tmp_path.iterdir()) == []  # refused before anything was written
    with pytest.raises(ValueError, match=r"\.npz"):
        save(rag, tmp_path / "index.json")
    assert rag._state is not None
    original = rag._state
    rag._state = dataclasses.replace(original, embedder=SimpleNamespace(dim=1))
    with pytest.raises(TypeError, match="HashedTfidfEmbedder"):
        save(rag, tmp_path / "index.npz")

    class Subclassed(HashedTfidfEmbedder):  # could override encode(); load would not know
        pass

    embedder = Subclassed(dim=original.embedder.dim, idf=original.embedder.idf)
    rag._state = dataclasses.replace(original, embedder=embedder)
    with pytest.raises(TypeError, match="HashedTfidfEmbedder"):
        save(rag, tmp_path / "index.npz")
    with pytest.raises(ValueError, match=r"\.npz"):
        load(tmp_path / "index")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", FORMAT_VERSION + 1, "format version 2"),
        ("version", True, "format version True"),
        ("minirag", "0.0.1", "saved by minirag '0.0.1'"),
        ("format", "something-else", "not a minirag index"),
        ("fingerprint", "0" * 64, "tokeniser or term hashing differs"),
    ],
)
def test_a_different_version_fails_with_a_clear_message(
    saved: Path, field: str, value: Any, message: str
) -> None:
    header = _sidecar(saved)
    header[field] = value
    _rewrite_sidecar(saved, header)
    with pytest.raises(PersistError, match=message):
        load(saved)


def test_a_changed_tokeniser_fails_the_fingerprint(
    saved: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def keeps_stopwords(text: str, *, remove_stopwords: bool = True) -> tuple[str, ...]:
        return tokenize(text, remove_stopwords=False)

    monkeypatch.setattr(persist, "tokenize", keeps_stopwords)  # as if tokenize.py were edited
    with pytest.raises(PersistError, match="tokeniser or term hashing differs"):
        load(saved)


def test_file_controlled_values_are_truncated_in_errors(saved: Path) -> None:
    header = _sidecar(saved)
    header["minirag"] = "x" * 100_000
    _rewrite_sidecar(saved, header)
    with pytest.raises(PersistError) as caught:
        load(saved)
    assert len(str(caught.value)) < 300


def test_arrays_from_another_save_are_detected(saved: Path, tmp_path: Path) -> None:
    other = MiniRAG(dim=64)
    other.index([Document("x", "A different corpus entirely.")])
    other_npz = save(other, tmp_path / "other.npz")[0]
    saved.write_bytes(other_npz.read_bytes())
    with pytest.raises(PersistError, match="does not match"):
        load(saved)


@pytest.mark.parametrize(
    "text",
    ["{not json", '{"format": "minirag-index", "format": "x"}', "[]", "\udcff", "[" * 100_000],
)
def test_an_unreadable_sidecar_fails_cleanly(saved: Path, text: str) -> None:
    saved.with_suffix(".json").write_bytes(text.encode("utf-8", "surrogateescape"))
    with pytest.raises(PersistError):
        load(saved)


def test_a_duplicated_document_cannot_shadow_the_real_one(saved: Path) -> None:
    # json.loads would keep the last of two equal keys and load without complaint.
    json_path = saved.with_suffix(".json")
    text = json_path.read_text(encoding="utf-8")
    assert text.count('"documents": {') == 1
    json_path.write_text(
        text.replace('"documents": {', '"documents": {"mercury": {"title": "Shadow"}, ', 1),
        encoding="utf-8",
    )
    with pytest.raises(PersistError, match="duplicate key"):
        load(saved)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("config", "dim"), 0, "must both be >= 1"),
        (("config", "rrf_k"), "60", "config.rrf_k"),
        (("config", "max_tokens"), 1.5, "config.max_tokens"),
        (("config", "weights", "bm25"), float("nan"), "finite"),
        (("config", "weights", "dense"), True, "finite"),
        (("config", "weights", "dense"), 10**400, "finite"),
        (("config", "weights"), [], "config.weights"),
        (("documents", "europa"), ["not", "a", "dict"], "documents"),
        (("documents", "europa", "title"), 3, "documents"),
        (("chunks", 0), ["europa", "europa#0", "text"], "expected"),
        (("chunks", 0, 0), "nowhere", "unknown document"),
        (("chunks", 1, 1), "mercury#0", "duplicate chunk_id"),
        (("chunks", 0, 3), 5, "empty or negative"),
        (("chunks", 0, 4), "2", "span"),
        (("chunks",), {}, "chunks"),
    ],
)
def test_a_tampered_sidecar_field_is_rejected(
    saved: Path, path: tuple[Any, ...], value: Any, message: str
) -> None:
    header = _sidecar(saved)
    target = header
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _rewrite_sidecar(saved, header)
    with pytest.raises(PersistError, match=message):
        load(saved)


Arrays = dict[str, Any]
TAMPERED: dict[str, tuple[Callable[[np.ndarray, np.ndarray], Arrays], str]] = {
    "a row short": (lambda m, i: {"matrix": m[:-1], "idf": i}, "expected float64"),
    "float32": (lambda m, i: {"matrix": m.astype(np.float32), "idf": i}, "expected float64"),
    "big-endian": (lambda m, i: {"matrix": m.astype(">f8"), "idf": i}, "expected float64"),
    "NaN": (lambda m, i: {"matrix": np.where(m > 0, np.nan, 0.0), "idf": i}, "NaN"),
    "infinite idf": (lambda m, i: {"matrix": m, "idf": np.full_like(i, np.inf)}, "infinity"),
    "not unit-norm": (lambda m, i: {"matrix": m * 50.0, "idf": i}, "unit-norm"),
    "missing idf": (lambda m, i: {"matrix": m}, "expected arrays"),
    "extra array": (lambda m, i: {"matrix": m, "idf": i, "extra": i}, "expected arrays"),
    "pickled": (lambda m, i: {"matrix": np.array([{}] * 3, dtype=object), "idf": i}, "unreadable"),
}


@pytest.mark.parametrize("case", sorted(TAMPERED))
def test_tampered_arrays_are_rejected_even_when_re_signed(
    fixture_engine: MiniRAG, saved: Path, case: str
) -> None:
    state = fixture_engine._state
    assert state is not None
    build, message = TAMPERED[case]
    _rewrite_arrays(saved, **build(state.dense.matrix, state.embedder.idf))
    with pytest.raises(PersistError, match=message):
        load(saved)


def test_compressed_arrays_are_refused(fixture_engine: MiniRAG, saved: Path) -> None:
    state = fixture_engine._state
    assert state is not None
    _rewrite_arrays(saved, compress=True, matrix=state.dense.matrix, idf=state.embedder.idf)
    with pytest.raises(PersistError, match="compressed"):
        load(saved)


def test_fortran_ordered_arrays_load_as_c_order(fixture_engine: MiniRAG, saved: Path) -> None:
    state = fixture_engine._state
    assert state is not None
    _rewrite_arrays(saved, matrix=np.asfortranarray(state.dense.matrix), idf=state.embedder.idf)
    loaded = load(saved)
    assert loaded._state is not None and loaded._state.dense.matrix.flags.c_contiguous
    _assert_same_answers(fixture_engine, loaded, QUERIES)


def test_a_file_that_is_not_an_archive_is_rejected(saved: Path) -> None:
    saved.write_bytes(b"not a zip file")
    header = _sidecar(saved)
    header["npz_sha256"] = hashlib.sha256(b"not a zip file").hexdigest()
    _rewrite_sidecar(saved, header)
    with pytest.raises(PersistError, match=r"not an \.npz"):
        load(saved)


def test_load_never_unpickles(monkeypatch: pytest.MonkeyPatch, saved: Path) -> None:
    calls: list[dict[str, Any]] = []
    real_load = np.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(persist.np, "load", spy)
    load(saved)
    assert calls == [{"allow_pickle": False}]
