"""End-to-end RAG pipeline: ingest a PDF, then answer questions about it."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunker import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_SENTENCES, chunk_pages
from .generator import Generator
from .loader import load_pdf
from .store import Retrieved, VectorStore


@dataclass
class Answer:
    """A generated answer plus the chunks it was grounded in."""

    text: str
    sources: list[Retrieved] = field(default_factory=list)


class RagPipeline:
    """Wires the loader, chunker, vector store, and generator together.

    The generator is injectable so the retrieval half can be tested without
    calling a language model.
    """

    def __init__(self, generator: Generator | None = None):
        self.store = VectorStore()
        self.generator = generator or Generator()
        self.num_chunks = 0

    def ingest(
        self,
        pdf_path: str | Path,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    ) -> int:
        """Load and index a PDF. Returns the number of chunks created."""
        pages = load_pdf(pdf_path)
        chunks = chunk_pages(pages, max_chars=max_chars, overlap_sentences=overlap_sentences)
        self.store.add(chunks)
        self.num_chunks = len(chunks)
        return self.num_chunks

    @property
    def ready(self) -> bool:
        return self.num_chunks > 0

    def answer(self, question: str, k: int = 4) -> Answer:
        """Retrieve the most relevant chunks and generate a grounded answer."""
        retrieved = self.store.search(question, k=k)
        text = self.generator.answer(question, retrieved)
        return Answer(text=text, sources=retrieved)
