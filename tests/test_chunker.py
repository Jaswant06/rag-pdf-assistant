"""Tests for the chunker."""

from src.chunker import chunk_pages
from src.loader import Page


def test_short_page_is_one_chunk():
    pages = [Page(number=1, text="This is a short page. It has two sentences.")]
    chunks = chunk_pages(pages, max_chars=700)
    assert len(chunks) == 1
    assert chunks[0].page == 1


def test_long_page_splits_into_multiple_chunks():
    sentence = "This is sentence number {}. ".format
    text = "".join(sentence(i) for i in range(40))
    chunks = chunk_pages([Page(number=2, text=text)], max_chars=200)
    assert len(chunks) > 1
    assert all(c.page == 2 for c in chunks)
    assert all(len(c.text) <= 260 for c in chunks)  # budget plus one trailing sentence


def test_chunks_carry_increasing_index_and_page():
    pages = [
        Page(number=1, text="Alpha one. Alpha two. Alpha three."),
        Page(number=5, text="Beta one. Beta two. Beta three."),
    ]
    chunks = chunk_pages(pages, max_chars=20)
    indices = [c.index for c in chunks]
    assert indices == sorted(indices)
    assert {c.page for c in chunks} == {1, 5}
