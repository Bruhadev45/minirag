"""minirag -- a complete RAG retrieval engine in under 800 lines.

Read the modules in this order; each one stands alone::

    tokenize -> chunker -> bm25 -> vectors -> fusion -> engine

See ``README.md`` for a quickstart and ``docs/walkthrough/`` for the same
system built up in four stages.
"""

from minirag.chunker import Chunk, chunk_corpus, chunk_document
from minirag.engine import Document, Hit, MiniRAG, SearchMode
from minirag.fusion import reciprocal_rank_fusion
from minirag.tokenize import tokenize

__version__ = "0.1.0"
__all__ = [
    "Chunk",
    "Document",
    "Hit",
    "MiniRAG",
    "SearchMode",
    "chunk_corpus",
    "chunk_document",
    "reciprocal_rank_fusion",
    "tokenize",
]
