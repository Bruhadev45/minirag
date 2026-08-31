"""Turn raw text into a normalised token stream.

Every retrieval system starts here, and so does almost every retrieval bug.
If the indexer and the query use different tokenisers, the searcher looks
for words the index does not contain and quietly returns nothing. So
minirag has exactly ONE tokeniser and both sides call it.

Three normalisation steps, each with a cost. **Lowercasing** collapses
"Mars" and "mars" into one term -- cheap, almost always right for English,
and it loses the signal that a word was a proper noun. **Punctuation
stripping** turns ``planet's`` into ``planet`` -- cheap, occasionally wrong
(``C++`` becomes ``c``), and the usual reason a symbol-heavy corpus
retrieves badly. **Stopword removal** drops high-frequency function words;
BM25 already discounts them via IDF, so this is mostly an index-size
optimisation, but it does change behaviour on short queries like "the who",
which becomes empty. That is why removal is optional and why
:func:`tokenize` never returns ``None`` -- an empty tuple is a legitimate
answer callers must handle.

A production system would add stemming and subword handling. Both are
deliberately absent: they cost a dependency or a hundred lines of
Porter-stemmer rules, and neither teaches you anything new.
"""

from __future__ import annotations

import re

#: Words dropped when ``remove_stopwords=True``. Kept small and inline on
#: purpose -- a 600-word list would be more "correct" and would make this
#: module unreadable, the wrong trade for a repo you read top to bottom.
# A readable block beats a 114-element list literal, so SIM905 is waived.
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    just me more most my no nor not now of off on once only or other our out over own same
    she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where which while who whom
    why will with you your
    """.split()  # noqa: SIM905
)

#: A token is a run of letters/digits, optionally with one internal
#: apostrophe so ``don't`` survives as a unit until it is truncated below.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str, *, remove_stopwords: bool = True) -> tuple[str, ...]:
    """Normalise ``text`` into a tuple of lowercase tokens, in order.

    ``""`` is valid and yields ``()``. Set ``remove_stopwords=False`` when
    you care about phrase-ish queries ("to be or not to be"). Duplicates are
    kept -- term frequency is the basis of BM25, so this must not dedupe.

    Raises:
        TypeError: If ``text`` is not a ``str``. Failing loudly is much
            kinder than indexing ``"b'bytes'"`` by accident.
    """
    if not isinstance(text, str):
        raise TypeError(f"tokenize() expects str, got {type(text).__name__}")

    tokens = (t.split("'", 1)[0] for t in _TOKEN_RE.findall(text.lower()))
    if remove_stopwords:
        return tuple(t for t in tokens if t and t not in STOPWORDS)
    return tuple(t for t in tokens if t)

