"""FastAPI REST interface for the RAG PDF Assistant.

Endpoints:
  GET  /health        liveness check
  POST /ingest        upload a PDF to index (multipart form field "file")
  POST /ask           ask a question about the indexed PDF (JSON body)

Run: uvicorn api:app --reload
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src import RagPipeline

app = FastAPI(title="RAG PDF Assistant API", version="1.0.0")

# Built once at startup so the embedding model and generator are not reloaded.
pipeline = RagPipeline()


class IngestResponse(BaseModel):
    chunks: int
    message: str


class AskRequest(BaseModel):
    question: str
    k: int = 4


class Source(BaseModel):
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": pipeline.ready, "chunks": pipeline.num_chunks}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        count = pipeline.ingest(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if count == 0:
        raise HTTPException(
            status_code=422,
            detail="No readable text found (the PDF may be a scanned image).",
        )
    return IngestResponse(chunks=count, message="PDF indexed.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not pipeline.ready:
        raise HTTPException(status_code=409, detail="Ingest a PDF before asking.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = pipeline.answer(request.question, k=request.k)
    sources = [
        Source(page=r.chunk.page, score=round(r.score, 4), text=r.chunk.text)
        for r in result.sources
    ]
    return AskResponse(answer=result.text, sources=sources)
