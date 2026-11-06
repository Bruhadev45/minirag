"""extras/embed.py: named embedding spaces, raw-text adapters and a space-checked index.

The learned adapter runs against the fake ``sentence_transformers`` module in
``conftest.py``. These tests check what the adapter passes in and how it
checks what comes back; they say nothing about any real model's quality.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from extras import embed
from extras.embed import (
    HASHED_MODEL,
    HashedTfidfText,
    MixedSpaceError,
    SentenceTransformerEmbedder,
    Space,
    TextEmbedder,
    VectorIndex,
)
from minirag.tokenize import tokenize
from minirag.vectors import HashedTfidfEmbedder

COMMIT = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = ("The Harbour Board inspects hulls.", "Berthing fees are due monthly.",
         "Records are kept for six years.")
IDS = ("c", "a", "b")


class Fixed:
    """A stub embedder returning whatever it is told to, in a learned space."""

    def __init__(self, documents: object, query: object = None, dim: int = 3) -> None:
        self.space = Space("learned", "stub", COMMIT, dim)
        self._documents, self._query = documents, query

    def embed_documents(self, texts):
        return self._documents

    def embed_query(self, text):
        return self._query


# --- spaces -------------------------------------------------------------------


@pytest.mark.parametrize("fields", [
    ("semantic", "m", "r", 3), ("learned", "", "r", 3), ("learned", " m", "r", 3),
    ("learned", "m", "r\n", 3), ("learned", "m", None, 3), ("learned", "m", "r", 0),
    ("learned", "m", "r", True), ("learned", "m", "r", 3.0),
])
def test_a_space_refuses_vague_identity(fields: tuple) -> None:
    with pytest.raises(ValueError):
        Space(*fields)


def test_labels_never_call_hashed_tfidf_learned() -> None:
    hashed, learned = Space("hashed-tfidf", HASHED_MODEL, "idf-x", 8), Space("learned", MODEL,
                                                                              COMMIT, 8)
    assert "not a learned embedding" in hashed.label and learned.label == "learned embedding"
    assert hashed.describe().startswith("hashed TF-IDF") and f"@ {COMMIT}, 8 dims" in (
        learned.describe())


@pytest.mark.parametrize(("kind", "revision", "pinned"), [
    ("learned", COMMIT, True), ("learned", "main", False), ("learned", COMMIT.upper(), False),
    ("learned", COMMIT[:12], False), ("hashed-tfidf", "idf-0123", True),
])
def test_only_a_commit_hash_pins_learned_weights(kind: str, revision: str, pinned: bool) -> None:
    assert Space(kind, "m", revision, 4).pinned is pinned


# --- hashed TF-IDF over raw text ----------------------------------------------


def test_hashed_tfidf_matches_the_core_embedder_on_tokens() -> None:
    embedder = HashedTfidfText.fit(TEXTS, dim=64)
    tokens = [tokenize(t) for t in TEXTS]
    core = HashedTfidfEmbedder.fit(tokens, dim=64)
    assert np.array_equal(embedder.embed_documents(TEXTS), core.encode(tokens))
    assert np.array_equal(embedder.embed_query(TEXTS[0]), core.encode([tokenize(TEXTS[0])])[0])
    assert isinstance(embedder, TextEmbedder) and type(embedder.space) is Space
    assert embedder.space.kind == "hashed-tfidf" and not embedder.core.idf.flags.writeable


def test_a_hand_built_hashed_embedder_must_match_its_weights() -> None:
    narrow, wide = HashedTfidfText.fit(TEXTS, dim=64), HashedTfidfText.fit((*TEXTS, "x y"), dim=64)
    for space, core in [(narrow.space, wide.core), (narrow.space, "weights"),
                        (("hashed-tfidf", HASHED_MODEL, narrow.space.revision, 64), narrow.core),
                        (narrow.space, HashedTfidfEmbedder(dim=64, idf=list(narrow.core.idf))),
                        (narrow.space, HashedTfidfEmbedder(dim=32, idf=narrow.core.idf))]:
        with pytest.raises(ValueError, match="does not describe these IDF weights"):
            HashedTfidfText(space, core)  # type: ignore[arg-type]


def test_last_bit_noise_in_the_weights_keeps_the_fit_name() -> None:
    fit = HashedTfidfText.fit(TEXTS, dim=64)
    noisy = HashedTfidfEmbedder(dim=64, idf=np.nextafter(fit.core.idf, np.inf))
    assert HashedTfidfText(fit.space, noisy).space == fit.space


def test_a_hashed_space_depends_on_the_fitted_corpus_not_its_order() -> None:
    space = HashedTfidfText.fit(TEXTS).space
    assert HashedTfidfText.fit(TEXTS[::-1]).space == space
    assert HashedTfidfText.fit((*TEXTS, "another tenant's text")).space != space
    assert HashedTfidfText.fit(TEXTS, dim=64).space != space


@pytest.mark.parametrize("texts", ["one string", [b"bytes"], [None]])
def test_hashed_tfidf_refuses_anything_but_strings(texts: object) -> None:
    with pytest.raises(TypeError):
        HashedTfidfText.fit(texts)  # type: ignore[arg-type]


@pytest.mark.parametrize("dim", [0, True, 2.5])
def test_hashed_tfidf_refuses_a_bad_width(dim: object) -> None:
    with pytest.raises(ValueError):
        HashedTfidfText.fit(TEXTS, dim=dim)  # type: ignore[arg-type]


# --- the index ------------------------------------------------------------------


def test_the_index_keeps_unit_read_only_rows_in_the_given_order() -> None:
    rows = [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    index = VectorIndex.build(Fixed(rows), IDS, TEXTS)
    assert index.ids == IDS and not index.matrix.flags.writeable
    assert np.allclose(index.matrix, [[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_huge_and_tiny_rows_keep_their_direction() -> None:
    rows = [[1e308, 1e308, 0.0], [1e-200, 0.0, 1e-200], [5e-324, 0.0, 0.0]]
    index = VectorIndex.build(Fixed(rows), IDS, TEXTS)
    half = 2 ** -0.5
    assert np.allclose(index.matrix, [[half, half, 0.0], [half, 0.0, half], [1.0, 0.0, 0.0]])


def test_scores_equal_up_to_float_noise_tie_and_break_on_id() -> None:
    space = Space("learned", "stub", COMMIT, 3)
    index = VectorIndex(space, ("b", "a"), np.array([[1.0, 0.0, 0.0], [1.0 - 4e-16, 0.0, 0.0]]))
    hits = index.search(Fixed(None, [1.0, 0.0, 0.0]), "q", top_k=2)
    assert hits == (("a", 1.0), ("b", 1.0))
    tiny = VectorIndex(space, ("a",), np.array([[1e-17, 1.0, 0.0]]))
    assert tiny.search(Fixed(None, [1.0, 0.0, 0.0]), "q", top_k=1) == ()  # noise is no match


def test_search_breaks_ties_on_id_and_drops_non_matches() -> None:
    same = [[1.0, 0.0, 0.0]] * 3
    index = VectorIndex.build(Fixed(same, [1.0, 0.0, 0.0]), IDS, TEXTS)
    assert [cid for cid, _ in index.search(Fixed(same, [2.0, 0.0, 0.0]), "q", top_k=3)] == [
        "a", "b", "c"]
    mixed = Fixed([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [1.0, 0.0, 0.0])
    hits = VectorIndex.build(mixed, IDS, TEXTS).search(mixed, "q", top_k=3)
    assert hits == (("c", 1.0),)  # cosine 0 and -1 are no match, as in the core


def test_search_accepts_a_one_row_query_and_caps_at_top_k() -> None:
    embedder = Fixed([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.5, 0.5, 0.0]], [[1.0, 0.0, 0.0]])
    hits = VectorIndex.build(embedder, IDS, TEXTS).search(embedder, "q", top_k=2)
    assert [cid for cid, _ in hits] == ["c", "a"]


def test_an_empty_index_never_calls_the_embedder() -> None:
    class Explodes(Fixed):
        def embed_documents(self, texts):
            raise AssertionError("called")

        embed_query = embed_documents

    embedder = Explodes(None)
    index = VectorIndex.build(embedder, (), ())
    assert index.matrix.shape == (0, 3) and index.search(embedder, "q", top_k=1) == ()


@pytest.mark.parametrize(("ids", "texts", "error"), [
    (("a", "a", "b"), TEXTS, ValueError), (("a", "b"), TEXTS, ValueError),
    ("abc", TEXTS, TypeError), (IDS, [*TEXTS[:2], 3], TypeError),
])
def test_the_index_refuses_ids_and_texts_that_do_not_pair_up(ids, texts, error) -> None:
    with pytest.raises(error):
        VectorIndex.build(Fixed([[1.0, 0.0, 0.0]] * 3), ids, texts)


@pytest.mark.parametrize("output", [
    [[1.0, 0.0, 0.0]] * 2, [[1.0, 0.0]] * 3, [[1.0, 0.0, np.nan]] * 3,
    [[np.inf, 0.0, 0.0]] * 3, [[-np.inf, 1.0, 0.0]] * 3, [["x", "y", "z"]] * 3, object(),
])
def test_the_index_refuses_malformed_embedder_output(output: object) -> None:
    with pytest.raises(ValueError):
        VectorIndex.build(Fixed(output), IDS, TEXTS)


def test_the_index_refuses_an_embedder_without_a_space() -> None:
    class NoSpace:
        space = ("learned", "stub", COMMIT, 3)

    with pytest.raises(TypeError):
        VectorIndex.build(NoSpace(), IDS, TEXTS)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", [[0.0, 1.0], [np.nan, 0.0, 0.0], [[1.0, 0.0, 0.0]] * 2, 1.0])
def test_search_refuses_a_malformed_query_vector(query: object) -> None:
    index = VectorIndex.build(Fixed([[1.0, 0.0, 0.0]] * 3), IDS, TEXTS)
    with pytest.raises(ValueError):
        index.search(Fixed(None, query), "q", top_k=1)


@pytest.mark.parametrize(("query", "top_k", "error"), [
    (b"q", 1, TypeError), ("q", 0, ValueError), ("q", True, ValueError), ("q", 1.0, ValueError),
])
def test_search_checks_its_arguments(query: object, top_k: object, error: type) -> None:
    embedder = Fixed([[1.0, 0.0, 0.0]] * 3, [1.0, 0.0, 0.0])
    with pytest.raises(error):
        VectorIndex.build(embedder, IDS, TEXTS).search(embedder, query, top_k=top_k)  # type: ignore[arg-type]


def test_search_refuses_every_other_space() -> None:
    hashed = HashedTfidfText.fit(TEXTS, dim=16)
    index = VectorIndex.build(hashed, IDS, TEXTS)
    others = [
        HashedTfidfText.fit((*TEXTS, "one more passage"), dim=16),  # same width, other corpus
        HashedTfidfText.fit(TEXTS, dim=32),
        Fixed(None, [1.0] * 16, dim=16),  # learned, same width
    ]
    for other in others:
        with pytest.raises(MixedSpaceError, match="index is hashed TF-IDF"):
            index.search(other, TEXTS[0], top_k=1)
    with pytest.raises(MixedSpaceError, match="not a Space"):
        index.search(object(), TEXTS[0], top_k=1)  # type: ignore[arg-type]

    class Agreeable(Space):
        def __eq__(self, other: object) -> bool:
            return True

        __hash__ = Space.__hash__

    liar = Fixed(None, [1.0] * 16, dim=16)
    liar.space = Agreeable("learned", "stub", COMMIT, 16)
    with pytest.raises(MixedSpaceError):
        index.search(liar, TEXTS[0], top_k=1)
    assert index.search(hashed, TEXTS[0], top_k=1)[0][0] == "c"


# --- the sentence-transformers adapter (fake model) ---------------------------


def test_the_adapter_loads_local_files_only_and_never_trusts_remote_code(fake_st) -> None:
    embedder = SentenceTransformerEmbedder(MODEL, revision=COMMIT, device="cpu")
    (model,) = fake_st.built
    assert model.name == MODEL and model.kwargs == {
        "revision": COMMIT, "device": "cpu", "local_files_only": True, "trust_remote_code": False}
    assert embedder.space == Space("learned", MODEL, COMMIT, 16) and embedder.space.pinned
    assert isinstance(embedder, TextEmbedder) and type(embedder.space) is Space
    SentenceTransformerEmbedder(MODEL, revision="main", allow_download=True)
    assert fake_st.built[1].kwargs["local_files_only"] is False


def test_the_adapter_hands_the_model_raw_text(fake_st) -> None:
    embedder = SentenceTransformerEmbedder(MODEL, revision=COMMIT, batch_size=4)
    raw = ["  The Board MUST inspect;\nhulls (s. 5).", "Ünïcode & the stopwords stay"]
    rows = embedder.embed_documents(raw)
    vector = embedder.embed_query("How often?  ")
    kinds = [(kind, texts) for kind, texts, _ in fake_st.built[0].calls]
    assert kinds == [("document", raw), ("query", ["How often?  "])]
    assert all(kw == {"batch_size": 4, "convert_to_numpy": True, "normalize_embeddings": True,
                      "show_progress_bar": False} for _, _, kw in fake_st.built[0].calls)
    assert rows.shape == (2, 16) and vector.shape == (16,)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0) and np.isclose(np.linalg.norm(vector), 1)


def test_the_adapter_checks_what_the_model_returns(fake_st, monkeypatch) -> None:
    embedder = SentenceTransformerEmbedder(MODEL, revision=COMMIT)
    monkeypatch.setattr(fake_st, "_encode", lambda self, kind, texts, kw: [[np.nan] * 16])
    with pytest.raises(ValueError, match="non-finite"):
        embedder.embed_documents(["one"])
    monkeypatch.setattr(fake_st, "_encode", lambda self, kind, texts, kw: [[1.0] * 8])
    with pytest.raises(ValueError, match="shape"):
        embedder.embed_query("one")


def test_a_local_directory_cannot_claim_a_commit_pin(fake_st, tmp_path) -> None:
    with pytest.raises(ValueError, match="local model directory ignores revision"):
        SentenceTransformerEmbedder(str(tmp_path), revision=COMMIT)
    assert not fake_st.built
    local = SentenceTransformerEmbedder(str(tmp_path), revision="copied-2026-11-06")
    assert not local.space.pinned


def test_the_adapter_skips_the_model_for_no_texts_and_refuses_non_strings(fake_st) -> None:
    embedder = SentenceTransformerEmbedder(MODEL, revision=COMMIT)
    assert embedder.embed_documents([]).shape == (0, 16) and not fake_st.built[0].calls
    for bad in ("one string", [1]):
        with pytest.raises(TypeError):
            embedder.embed_documents(bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        embedder.embed_query(None)  # type: ignore[arg-type]


def test_the_adapter_falls_back_to_the_older_width_method(fake_st, monkeypatch) -> None:
    monkeypatch.delattr(fake_st, "get_embedding_dimension")
    monkeypatch.setattr(fake_st, "get_sentence_embedding_dimension", lambda self: 24,
                        raising=False)
    assert SentenceTransformerEmbedder(MODEL, revision=COMMIT).space.dim == 24


@pytest.mark.parametrize("dim", [None, 0, 3.0, True])
def test_the_adapter_refuses_a_model_without_a_fixed_width(fake_st, monkeypatch, dim) -> None:
    monkeypatch.setattr(fake_st, "get_embedding_dimension", lambda self: dim)
    with pytest.raises(ValueError, match="fixed embedding width"):
        SentenceTransformerEmbedder(MODEL, revision=COMMIT)


def test_the_adapter_refuses_a_model_with_no_width_method(fake_st, monkeypatch) -> None:
    monkeypatch.delattr(fake_st, "get_embedding_dimension")
    with pytest.raises(ValueError, match="fixed embedding width"):
        SentenceTransformerEmbedder(MODEL, revision=COMMIT)


@pytest.mark.parametrize(("model", "kwargs"), [
    (MODEL, {"revision": ""}), ("", {"revision": COMMIT}), (MODEL, {"revision": None}),
    (MODEL, {"revision": COMMIT, "batch_size": 0}),
    (MODEL, {"revision": COMMIT, "batch_size": True}),
])
def test_the_adapter_checks_its_arguments_before_loading(fake_st, model, kwargs) -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(model, **kwargs)
    assert not fake_st.built


def test_without_the_package_the_error_names_the_extra(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match=r"\.\[embeddings\]"):
        SentenceTransformerEmbedder(MODEL, revision=COMMIT)


@pytest.mark.parametrize("error", [ModuleNotFoundError("No module named 'torch'", name="torch"),
                                   ImportError("libtorch_cpu.dylib not found")])
def test_a_broken_install_is_not_reported_as_a_missing_extra(monkeypatch, error) -> None:
    def broken(name: str) -> None:
        raise error

    monkeypatch.setattr(embed.importlib, "import_module", broken)
    with pytest.raises(ImportError) as info:
        SentenceTransformerEmbedder(MODEL, revision=COMMIT)
    assert info.value is error


def test_two_revisions_of_one_model_are_different_spaces(fake_st) -> None:
    old = SentenceTransformerEmbedder(MODEL, revision="0" * 40)
    new = SentenceTransformerEmbedder(MODEL, revision=COMMIT)
    index = VectorIndex.build(old, IDS, TEXTS)
    with pytest.raises(MixedSpaceError):
        index.search(new, TEXTS[0], top_k=1)
    assert index.search(old, TEXTS[0], top_k=1)[0][0] == "c"
