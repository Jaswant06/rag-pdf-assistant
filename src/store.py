"""In-memory vector store: embed chunks once, retrieve the closest to a query.

For a single document, holding the embeddings in memory is simpler and faster
than running a full vector database.
"""

from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .chunker import Chunk

EMBED_MODEL = "all-MiniLM-L6-v2"


@dataclass
class Retrieved:
    """A chunk returned by a search, with its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Holds chunk embeddings and answers nearest-neighbour queries."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self._model = SentenceTransformer(model_name)
        self._chunks: list[Chunk] = []
        self._embeddings = None

    def add(self, chunks: list[Chunk]) -> None:
        """Embed and store the chunks, replacing anything already held."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._embeddings = None
            return
        texts = [c.text for c in self._chunks]
        self._embeddings = self._model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        """Return the k chunks most similar to the query, best first."""
        if not self._chunks:
            return []
        query_vec = self._model.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )
        scores = self._model.similarity(query_vec, self._embeddings)[0]
        k = min(k, len(self._chunks))
        top = scores.topk(k)
        return [
            Retrieved(chunk=self._chunks[int(i)], score=float(s))
            for s, i in zip(top.values, top.indices)
        ]
