"""Save an indexed engine to disk and load it back, without pickle.

:meth:`MiniRAG.index` chunks, tokenises and embeds the whole corpus every
time. :func:`save` writes the result to two files; :func:`load` rebuilds an
engine whose ``search`` returns exactly what the saved one returned.

- ``index.npz`` holds the two numeric arrays: the dense ``matrix`` (one
  unit-norm row per chunk) and the embedder's ``idf`` weights. They are
  stored as raw float64, so they come back bit for bit.
- ``index.json``, the sidecar, holds everything else: a versioned header,
  the engine's settings, each document's metadata and every chunk's text.

**The BM25 index is rebuilt, not stored.** Its postings are a pure function
of the chunk text and the tokeniser, which costs one tokenise pass at load
and keeps the lexical side in step with how queries are tokenised. The
dense matrix *is* stored, because it is the part a better embedder makes
expensive to recompute.

**Mismatches fail loudly.** A changed tokeniser or term hash would make the
saved vectors and new queries disagree without any error, so the header
records the format version, the minirag version and a fingerprint: what the
tokeniser and the hash do to one fixed probe sentence. Any difference raises
:class:`PersistError`. The fingerprint catches an edit made without a version
bump, but only one the probe exercises.

**Saving is atomic per file, not per pair.** Each file is written to a
temporary file and renamed into place, so a reader never sees half of one,
and the sidecar records the ``.npz``'s SHA-256, so a pair from two saves is
caught at load instead of served. A crash between the two renames is caught
too, but it leaves neither the old index nor the new one loadable: save to a
new path when the old one must survive. Files are created owner-only
(``0600`` on POSIX), and replacing a file gives it that mode.

**Trust.** Nothing here unpickles: ``np.load`` runs with
``allow_pickle=False`` and compressed archives are refused, so memory use
stays linear in the file's size. Every field is type-, shape- and
range-checked before use. That turns corruption into a clear error. It does
not authenticate: anyone who can write the sidecar can recompute the hash,
and a crafted pair decides which chunk text ranks first and so reaches your
prompt. Load only indexes you or your own pipeline wrote.

This module reads the engine's private state, because the core has no line
left for a public export API; ``tests/test_persist.py`` pins the round trip
so a core refactor that breaks it fails there first.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import numbers
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from minirag import Chunk, MiniRAG, __version__, bm25, tokenize
from minirag.engine import _State
from minirag.vectors import DEFAULT_DIM, DenseIndex, HashedTfidfEmbedder

FORMAT = "minirag-index"
FORMAT_VERSION = 1
_ARRAYS = frozenset({"matrix", "idf"})
_WINDOW = ("max_tokens", "overlap_tokens")
#: Exercises case, digits, apostrophes, stopwords, symbols and non-ASCII letters.
_PROBE = "The Europa ocean's 2 MOONS don't shine; C++ naïve café, 42!"


class PersistError(ValueError):
    """A saved index that cannot be loaded: wrong version, mismatched or malformed."""


def _short(value: object, limit: int = 60) -> str:
    """``repr`` of a file-controlled value, cut so a huge field cannot flood a log."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def fingerprint() -> str:
    """SHA-256 of the probe's tokens and the columns they hash to, in this minirag."""
    tokens = tokenize(_PROBE)
    row = HashedTfidfEmbedder(dim=DEFAULT_DIM, idf=np.ones(DEFAULT_DIM)).encode([tokens])[0]
    probe = json.dumps([list(tokens), np.flatnonzero(row).tolist()])  # no floats: portable
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()


def save(engine: MiniRAG, path: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Write ``engine``'s index to ``path`` (a ``.npz``) and its ``.json`` sidecar.

    Existing files are replaced. Returns ``(npz_path, sidecar_path)``.

    Raises:
        RuntimeError: If ``engine`` has not been indexed.
        ValueError: If ``path`` does not end in ``.npz``.
        TypeError: If any metadata key or value is not a ``str``.
    """
    npz_path = _npz_path(path)
    state = engine._state
    if state is None:
        raise RuntimeError("call index() before save()")
    if type(state.embedder) is not HashedTfidfEmbedder:  # a subclass may encode differently
        raise TypeError("only the built-in HashedTfidfEmbedder can be saved")
    for doc_id, fields in state.metadata.items():
        for key, value in fields.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    f"metadata for {_short(doc_id)} must map str to str to be saved; "
                    f"found a {type(key).__name__} key with a {type(value).__name__} value"
                )

    buffer = io.BytesIO()
    np.savez(buffer, matrix=state.dense.matrix, idf=state.embedder.idf)  # stored, not deflated
    blob = buffer.getvalue()
    sidecar = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "minirag": __version__,
        "fingerprint": fingerprint(),
        "npz_sha256": hashlib.sha256(blob).hexdigest(),
        "config": {
            **engine._window, "dim": engine._dim, "rrf_k": engine._rrf_k,
            "weights": dict(engine._weights),
        },
        "documents": {doc_id: dict(fields) for doc_id, fields in state.metadata.items()},
        "chunks": [[c.doc_id, c.chunk_id, c.text, *c.sentence_span] for c in state.chunks],
    }
    # ensure_ascii (the default) escapes lone surrogates, so the encode below cannot fail.
    text = json.dumps(sidecar, indent=1, allow_nan=False)
    json_path = npz_path.with_suffix(".json")
    _write_atomic(npz_path, blob)  # the hash catches a crash between these in either order
    _write_atomic(json_path, text.encode("ascii"))
    return npz_path, json_path


def load(path: str | os.PathLike[str]) -> MiniRAG:
    """Rebuild the engine saved at ``path`` (the ``.npz``; the sidecar sits beside it).

    Raises:
        PersistError: If the format, its version, the minirag version or the
            fingerprint differ, the two files do not belong together, or any
            field is malformed.
        ValueError: If ``path`` does not end in ``.npz``.
        OSError: If either file cannot be read.
    """
    npz_path = _npz_path(path)
    json_path = npz_path.with_suffix(".json")
    try:
        header = json.loads(json_path.read_bytes(), object_pairs_hook=_unique_keys)
    except PersistError:
        raise
    except (ValueError, RecursionError) as exc:  # bad JSON or UTF-8, or absurd nesting
        raise PersistError(f"{json_path.name} is not valid JSON: {exc}") from exc
    _check_header(_typed(header, dict, "sidecar"), json_path.name)

    blob = npz_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != header.get("npz_sha256"):
        raise PersistError(
            f"{npz_path.name} does not match {json_path.name}: they come from different "
            "saves, or a save was interrupted; save the engine again"
        )
    config = _typed(header.get("config"), dict, "config")
    window = {k: _typed(config.get(k), int, f"config.{k}") for k in _WINDOW}
    dim, rrf_k = (_typed(config.get(k), int, f"config.{k}") for k in ("dim", "rrf_k"))
    if dim < 1 or rrf_k < 1:
        raise PersistError("config: dim and rrf_k must both be >= 1")
    weights = {
        name: _finite(value, f"config.weights[{_short(name)}]")
        for name, value in _typed(config.get("weights"), dict, "config.weights").items()
    }
    metadata = _documents(header.get("documents"))
    chunks = _chunks(header.get("chunks"), metadata)
    matrix, idf = _arrays(blob, n_chunks=len(chunks), dim=dim)

    # Window settings are checked by the chunker on the next index(), as for a fresh engine.
    engine = MiniRAG(**window, dim=dim, rrf_k=rrf_k, weights=weights)
    engine._state = _State(
        chunks=chunks,
        metadata=metadata,
        lexical=bm25.build_index(tuple(tokenize(chunk.text) for chunk in chunks)),
        embedder=HashedTfidfEmbedder(dim=dim, idf=idf),
        dense=DenseIndex(matrix=matrix),
    )
    return engine


def _npz_path(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path)
    if resolved.suffix != ".npz":
        raise ValueError(f"path must end in .npz, got {resolved.name!r}")
    return resolved


def _write_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same directory, then rename over ``path``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json`` keeps the last of two equal keys; a saved index never has any."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise PersistError("sidecar has a duplicate key")
    return result


def _typed(value: Any, kind: type, what: str) -> Any:
    if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
        raise PersistError(f"{what}: expected {kind.__name__}, got {type(value).__name__}")
    return value


def _finite(value: Any, what: str) -> float:
    if not isinstance(value, bool) and isinstance(value, numbers.Real):
        try:
            if math.isfinite(value):
                return value
        except OverflowError:  # an int too large for a float, e.g. 10**400
            pass
    raise PersistError(f"{what}: expected a finite number")


def _check_header(header: dict[str, Any], name: str) -> None:
    if header.get("format") != FORMAT:
        raise PersistError(f"{name} is not a minirag index sidecar")
    version = header.get("version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise PersistError(
            f"{name} is format version {_short(version)}; this minirag reads version "
            f"{FORMAT_VERSION}. Re-index and save again."
        )
    if header.get("minirag") != __version__:
        raise PersistError(
            f"{name} was saved by minirag {_short(header.get('minirag'))}, this is "
            f"{__version__}; the tokeniser or embedder may differ, so re-index and save again"
        )
    if header.get("fingerprint") != fingerprint():
        raise PersistError(
            f"{name} was saved by code whose tokeniser or term hashing differs from this "
            "minirag's; re-index and save again"
        )


def _documents(raw: Any) -> dict[str, Mapping[str, str]]:
    metadata: dict[str, Mapping[str, str]] = {}
    for doc_id, fields in _typed(raw, dict, "documents").items():
        where = f"documents[{_short(doc_id)}]"
        for key, value in _typed(fields, dict, where).items():
            _typed(value, str, f"{where}[{_short(key)}]")
        metadata[doc_id] = dict(fields)
    return metadata


def _chunks(raw: Any, metadata: Mapping[str, Mapping[str, str]]) -> tuple[Chunk, ...]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for n, entry in enumerate(_typed(raw, list, "chunks")):
        if not isinstance(entry, list) or len(entry) != 5:
            raise PersistError(f"chunks[{n}]: expected [doc_id, chunk_id, text, start, end]")
        doc_id, chunk_id, text = (_typed(v, str, f"chunks[{n}]") for v in entry[:3])
        start, end = (_typed(v, int, f"chunks[{n}] span") for v in entry[3:])
        if doc_id not in metadata:
            raise PersistError(f"chunks[{n}] belongs to unknown document {_short(doc_id)}")
        if chunk_id in seen:
            raise PersistError(f"chunks[{n}]: duplicate chunk_id {_short(chunk_id)}")
        if not 0 <= start < end:
            raise PersistError(f"chunks[{n}]: sentence span is empty or negative")
        seen.add(chunk_id)
        chunks.append(Chunk(doc_id, chunk_id, text, (start, end)))
    return tuple(chunks)


def _arrays(blob: bytes, *, n_chunks: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Read and check both arrays; returned read-only so a search cannot alter them."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise PersistError(f"the arrays file is not an .npz archive: {exc}") from exc
    names = sorted(info.filename for info in entries)
    if names != sorted(f"{name}.npy" for name in _ARRAYS):  # checked before anything is read
        raise PersistError(f"expected arrays {sorted(_ARRAYS)}, found {_short(names)}")
    if any(info.compress_type != zipfile.ZIP_STORED for info in entries):
        raise PersistError("compressed arrays are refused: save() never writes them")
    try:
        with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in _ARRAYS}
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise PersistError(f"unreadable arrays: {exc}") from exc
    for name, shape in (("matrix", (n_chunks, dim)), ("idf", (dim,))):
        array = arrays[name]
        if array.dtype != np.float64 or array.shape != shape:
            raise PersistError(f"{name}: expected float64 {shape}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise PersistError(f"{name} contains NaN or infinity")
        arrays[name] = np.ascontiguousarray(array)  # C order, as save() wrote it
        arrays[name].setflags(write=False)
    norms = np.linalg.norm(arrays["matrix"], axis=1)
    if not np.all((np.abs(norms - 1.0) < 1e-6) | (norms == 0.0)):  # empty chunks embed to 0
        raise PersistError("matrix rows must be unit-norm or all zero, as the embedder makes them")
    return arrays["matrix"], arrays["idf"]
