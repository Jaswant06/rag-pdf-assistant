"""Split page text into overlapping, sentence-aware chunks for retrieval.

Each chunk stays under a character budget and overlaps its neighbour by a
sentence, so a fact on a boundary is not lost. Chunks keep their page number so
answers can cite sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Page

DEFAULT_MAX_CHARS = 700
DEFAULT_OVERLAP_SENTENCES = 1


@dataclass
class Chunk:
    """A retrievable piece of the document."""

    text: str
    page: int
    index: int  # position of the chunk within the whole document


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminators and line breaks."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_pages(
    pages: list[Page],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Turn pages into overlapping, page-tagged chunks."""
    chunks: list[Chunk] = []
    idx = 0

    for page in pages:
        sentences = _split_sentences(page.text)
        i = 0
        while i < len(sentences):
            # Grow a window of sentences until the character budget is hit.
            buf: list[str] = []
            length = 0
            j = i
            while j < len(sentences) and (length + len(sentences[j]) + 1 <= max_chars or not buf):
                buf.append(sentences[j])
                length += len(sentences[j]) + 1
                j += 1

            text = " ".join(buf).strip()
            if text:
                chunks.append(Chunk(text=text, page=page.number, index=idx))
                idx += 1

            if j >= len(sentences):
                break
            # Step forward, leaving `overlap_sentences` of overlap with this chunk.
            i = max(i + 1, j - overlap_sentences)

    return chunks
