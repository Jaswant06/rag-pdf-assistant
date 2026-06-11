"""End-to-end pipeline test with a fake chat model (no network)."""

from src.generator import Generator
from src.pipeline import RagPipeline


class FakeChat:
    """Records the prompt and returns a canned answer, so tests stay offline."""

    def __init__(self):
        self.last_messages = None

    def __call__(self, messages):
        self.last_messages = messages
        return "A grounded answer."


def _write_pages_as_chunks(pipeline, texts):
    from src.chunker import Chunk
    pipeline.store.add([Chunk(text=t, page=i + 1, index=i) for i, t in enumerate(texts)])
    pipeline.num_chunks = len(texts)


def test_answer_returns_text_and_sources():
    fake = FakeChat()
    pipeline = RagPipeline(generator=Generator(chat=fake))
    _write_pages_as_chunks(pipeline, [
        "Mount Everest is the tallest mountain on Earth.",
        "The Pacific is the largest ocean.",
    ])

    result = pipeline.answer("What is the tallest mountain?", k=1)

    assert result.text == "A grounded answer."
    assert len(result.sources) == 1
    assert "Everest" in result.sources[0].chunk.text
    # the retrieved context was actually placed in the prompt
    assert "Everest" in fake.last_messages[-1]["content"]


def test_answer_without_ingest_reports_no_context():
    pipeline = RagPipeline(generator=Generator(chat=FakeChat()))
    result = pipeline.answer("anything?")
    assert result.sources == []
    assert "could not find" in result.text.lower()
