"""The budget is a feature, so it is enforced like one.

minirag promises a retrieval engine you can read in an afternoon. That
promise is only true while the implementation stays small, and "stays
small" is not something a README can enforce. This test can.

It counts every physical line of ``src/`` -- code, docstrings, comments and
blanks alike -- because those are the lines a reader actually scrolls
through. If you need more room, delete something first.
"""

from __future__ import annotations

from pathlib import Path

LINE_BUDGET = 800
SRC = Path(__file__).resolve().parents[1] / "src" / "minirag"


def _counts() -> dict[str, int]:
    return {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in sorted(SRC.glob("*.py"))
    }


def test_implementation_stays_under_the_line_budget() -> None:
    counts = _counts()
    total = sum(counts.values())
    breakdown = "\n".join(f"  {name:<14} {n:>4}" for name, n in counts.items())
    assert total < LINE_BUDGET, (
        f"src/minirag is {total} lines, budget is {LINE_BUDGET}:\n{breakdown}"
    )


def test_every_module_is_individually_readable() -> None:
    oversized = {name: n for name, n in _counts().items() if n > 250}
    assert not oversized, f"split these up: {oversized}"


def test_the_package_has_no_dependencies_beyond_numpy() -> None:
    forbidden = ("langchain", "llama_index", "sklearn", "scikit", "torch", "transformers")
    for path in SRC.glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        for name in forbidden:
            assert f"import {name}" not in source, f"{path.name} imports {name}"
