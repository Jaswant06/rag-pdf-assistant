"""Read text out of a PDF, one entry per page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    """A single PDF page with its 1-based number and extracted text."""

    number: int
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    """Extract text from a PDF, one entry per page.

    Pages with no extractable text (for example scanned images) are skipped.
    """
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(number=i, text=text))
    return pages
