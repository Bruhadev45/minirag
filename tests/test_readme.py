"""The README's first-result examples show pasted output; these keep it true.

A new user copies a command from the README and compares what they see with
the block printed under it. If a change moves a score or reorders a hit, the
README is now wrong, so each example is run here and its output compared
line for line with the block that follows it. They run against this checkout;
the Git install command is only checked as text, since running it needs the
network and a fresh venv.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
HEADING = re.compile(r"#{1,3} ")
INSTALL = '.venv/bin/pip install "git+https://github.com/Bruhadev45/minirag.git"'


def section_blocks(heading: str) -> list[tuple[str, str]]:
    """``(language, body)`` for each fenced block between ``heading`` and the next heading.

    Scans line by line so a ``## `` comment inside a code block does not end
    the section early.
    """
    text = README.read_text(encoding="utf-8")
    marker = f"\n{heading}\n"
    if marker not in text:
        pytest.fail(f"README.md has no heading {heading!r}")
    blocks: list[tuple[str, str]] = []
    lang: str | None = None
    body: list[str] = []
    for line in text.split(marker, 1)[1].splitlines():
        if lang is None and HEADING.match(line):
            break
        if lang is None and line.startswith("```"):
            lang, body = line[3:].strip(), []
        elif lang is not None and line == "```":
            blocks.append((lang, "\n".join(body)))
            lang = None
        elif lang is not None:
            body.append(line)
    return blocks


def example_and_output(heading: str, lang: str) -> tuple[str, str]:
    """The first ``lang`` block under ``heading`` and the output block right after it."""
    blocks = section_blocks(heading)
    langs = [block_lang for block_lang, _ in blocks]
    assert lang in langs, f"{heading!r} has no {lang} block: {langs}"
    i = langs.index(lang)
    assert langs[i + 1 : i + 2] == [""], f"{heading!r}: no output block after the {lang} block"
    example, expected = blocks[i][1], blocks[i + 1][1]
    assert example.strip(), f"{heading!r}: the {lang} block is empty"
    assert expected.strip(), f"{heading!r}: the output block is empty"
    return example, expected


def run_python(args: list[str]) -> list[str]:
    """Stdout lines of ``python *args`` from the repo root, as a reader would run it."""
    result = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        check=False,
        cwd=ROOT,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    # stderr is not compared: a platform's BLAS warning is not a README error.
    assert result.returncode == 0, result.stderr
    return result.stdout.rstrip("\n").splitlines()


def test_first_result_installs_from_git_and_shows_the_demo_output() -> None:
    commands, expected = example_and_output("## First result in a minute", "bash")
    assert INSTALL in commands.splitlines()
    assert ".venv/bin/python -m minirag.demo" in commands.splitlines()
    assert run_python(["-m", "minirag.demo"]) == expected.splitlines()


@pytest.mark.parametrize(
    "heading", ["### Use it from Python", "## Develop: clone, test, and run the extras"]
)
def test_python_example_prints_the_output_shown_under_it(heading: str) -> None:
    code, expected = example_and_output(heading, "python")
    assert run_python(["-c", code]) == expected.splitlines()
