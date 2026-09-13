"""A runnable tour of the engine: ``python -m minirag.demo``.

Six one-line documents and two questions, small enough to guess the answers
before running it. Every mode gets the first right and all three get the
second wrong, in the way this repo is honest about: "volcanic" never
matches "volcanically" without a stemmer, and a hashed embedder cannot
bridge that gap. See ``docs/walkthrough/05-putting-it-together.md``.
"""

from __future__ import annotations

from minirag.engine import Document, MiniRAG, SearchMode

DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("europa", "Europa is a moon of Jupiter. Beneath its icy crust lies a salty ocean."),
    ("enceladus", "Enceladus is a small moon of Saturn that vents water into space."),
    ("titan", "Titan is a moon with a thick nitrogen atmosphere and lakes of methane."),
    ("mars", "Mars has polar caps of water ice and a thin carbon dioxide atmosphere."),
    ("io", "Io is the most volcanically active body in the solar system."),
    ("pluto", "Pluto has mountains of water ice and a heart-shaped nitrogen glacier."),
)
QUERIES: tuple[tuple[str, str], ...] = (
    ("which moon hides an ocean beneath the ice", "europa, and all three modes agree"),
    ("volcanic eruptions on a moon", "io missing everywhere -- no stemming, no synonyms"),
)
MODES: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")


def main() -> None:
    """Index the sample corpus and print each mode's top three per query."""
    engine = MiniRAG(max_tokens=60, overlap_tokens=20)
    engine.index([Document(doc_id=doc_id, text=text) for doc_id, text in DOCUMENTS])
    print(f"indexed {len(DOCUMENTS)} documents -> {engine.chunk_count} chunks")
    for query, note in QUERIES:
        print(f"\nquery: {query!r}\n  expect: {note}")
        for mode in MODES:
            hits = engine.search(query, top_k=3, mode=mode)
            ranked = "   ".join(f"{i}. {h.doc_id} {h.score:.4f}" for i, h in enumerate(hits, 1))
            print(f"  {mode:>6}   {ranked or '(no hits)'}")
    print("\nScores compare only within a mode; the ranking is the point.")


if __name__ == "__main__":
    main()
