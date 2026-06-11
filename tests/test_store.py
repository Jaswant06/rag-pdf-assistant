"""Tests for the vector store retrieval."""

from src.chunker import Chunk
from src.store import VectorStore


def _store_with(texts):
    store = VectorStore()
    store.add([Chunk(text=t, page=i + 1, index=i) for i, t in enumerate(texts)])
    return store


def test_retrieves_most_relevant_chunk_first():
    store = _store_with([
        "The capital of France is Paris.",
        "Photosynthesis happens in plant leaves.",
        "The recipe needs two cups of flour.",
    ])
    results = store.search("Which city is the French capital?", k=1)
    assert results
    assert "Paris" in results[0].chunk.text


def test_returns_at_most_k_results():
    store = _store_with(["one apple", "two bananas", "three cherries"])
    results = store.search("fruit", k=2)
    assert len(results) == 2
    # scores come back sorted, best first
    assert results[0].score >= results[1].score


def test_empty_store_returns_nothing():
    assert VectorStore().search("anything") == []
