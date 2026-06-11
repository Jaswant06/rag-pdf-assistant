# DocuMind: Chat With Your PDF (Retrieval Augmented Generation, NLP)

DocuMind lets you upload a PDF, ask questions about it, and get answers grounded
in the document with the page number each answer came from. Built with
**Retrieval Augmented Generation (RAG)**: relevant passages are retrieved with
sentence embeddings, then a language model writes an answer using only those
passages, so it does not make things up.

> Upload a PDF -> ask a question -> get a cited answer, or an honest "not found".

**Live demo:** https://huggingface.co/spaces/JaswantDev/rag-pdf-assistant

## Why RAG (and not just a chatbot)

A plain language model only knows its training data. Ask it about your specific
document and it guesses. RAG fixes this: it finds the passages in your PDF that
match the question and gives them to the model as context, so the answer is
grounded in the real text. When the answer is not in the document, the assistant
says so instead of inventing one.

## How it works

1. **Load:** extract text from the PDF, one entry per page (`src/loader.py`).
2. **Chunk:** split each page into overlapping, sentence-aware chunks that
   remember their page number (`src/chunker.py`).
3. **Embed and store:** turn every chunk into a vector with `all-MiniLM-L6-v2`
   and keep them in memory (`src/store.py`).
4. **Retrieve:** embed the question and find the closest chunks by cosine
   similarity.
5. **Generate:** send those chunks plus the question to a language model
   (`Qwen/Qwen2.5-7B-Instruct` via the Hugging Face Inference API) with an
   instruction to answer only from the context and cite pages (`src/generator.py`).

Steps 2 to 4 are the same embedding-and-similarity idea as a semantic search
tool. RAG adds the PDF input and the grounded generation step on top.

## Tech stack

Python, Sentence-Transformers, Hugging Face Inference API, pypdf, FastAPI, Gradio,
cosine similarity.

## Run it locally

```bash
pip install -r requirements.txt

# The generation step uses the Hugging Face Inference API, so it needs a token:
hf auth login            # or: export HF_TOKEN=your_token

python app.py            # the web UI at http://127.0.0.1:7860
# or the REST API:
uvicorn api:app --reload # http://127.0.0.1:8000/docs
```

Run the tests (the LLM is faked, so they need no API token):

```bash
pip install -r requirements-dev.txt
pytest
```

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | status, whether a PDF is indexed |
| POST | `/ingest` | multipart `file` (a PDF) | number of chunks indexed |
| POST | `/ask` | JSON `{ "question": "...", "k": 4 }` | answer plus the source chunks |

## Configuration

- `RAG_MODEL`: the generation model id (default `Qwen/Qwen2.5-7B-Instruct`).
- `HF_TOKEN`: Hugging Face token used for inference. On a Hugging Face Space, add
  it as a secret.

## Project structure

```text
rag-pdf-assistant/
├── app.py                  # Gradio web UI
├── api.py                  # FastAPI REST API
├── src/
│   ├── __init__.py         # public API
│   ├── loader.py           # PDF to page text
│   ├── chunker.py          # page text to overlapping chunks
│   ├── store.py            # embed chunks, retrieve by similarity
│   ├── generator.py        # context plus question to grounded answer
│   └── pipeline.py         # ties ingest and answer together
├── tests/                  # unit tests (offline, fake model)
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── README.md
```

## Limitations

- **Retrieval is similarity-based.** If the answer is worded very differently
  from the question, or is spread thin across many pages, the right chunk may not
  surface. Garbage retrieval means a weak answer.
- **One document at a time.** The index is shared and rebuilt on each upload, so
  it is not isolated per user. Fine for a demo; a real product needs per-session
  or persistent storage.
- **No OCR.** Scanned, image-only PDFs have no extractable text and return
  nothing. They would need an OCR step first.
- **In-memory store.** Embeddings are held in memory and rebuilt each upload.
  That is the right choice for a single document; a large corpus or persistence
  would call for a real vector database (FAISS, Chroma, pgvector).
- **Generation needs a hosted model.** Answers come from the Hugging Face
  Inference API, so a token and network are required. Quality scales with the
  chosen model.

## Possible next steps

- Persist embeddings in a real vector store and support multiple documents.
- Add OCR so scanned PDFs work.
- Per-session document isolation for multi-user use.
- Show the exact source snippets in the UI, not just page numbers.

---

Keywords: rag, retrieval augmented generation, chat with pdf, pdf question
answering, semantic search, sentence embeddings, NLP, LLM, Hugging Face, FastAPI,
Gradio, Python.
