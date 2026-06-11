"""Generate a grounded answer from retrieved context using an LLM.

Generation goes through the Hugging Face Inference API, so no model weights are
downloaded. The model id comes from RAG_MODEL (default Qwen2.5-7B-Instruct). The
chat client is injected, which keeps the tests offline.
"""

from __future__ import annotations

import os
from typing import Protocol

from huggingface_hub import InferenceClient

from .store import Retrieved

DEFAULT_MODEL = os.environ.get("RAG_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = (
    "You answer questions using only the provided context from a document. "
    "Be concise and factual. If the answer is not in the context, say you could "
    "not find it in the document. When you use a passage, cite its page in "
    "square brackets, for example [p. 3]."
)

NO_CONTEXT_MESSAGE = "I could not find anything relevant to that in the document."


def build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    """Assemble the context block and question into a single user message."""
    context = "\n\n".join(f"[p. {r.chunk.page}] {r.chunk.text}" for r in retrieved)
    return f"Context:\n{context}\n\nQuestion: {question}"


class ChatModel(Protocol):
    """Anything that can turn a list of chat messages into a string reply."""

    def __call__(self, messages: list[dict]) -> str: ...


class HuggingFaceChat:
    """Default chat backend: Hugging Face Inference API."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 400):
        self.model = model
        self.max_tokens = max_tokens
        self._client = InferenceClient(model=model)

    def __call__(self, messages: list[dict]) -> str:
        response = self._client.chat_completion(
            messages, max_tokens=self.max_tokens, temperature=0.2
        )
        return response.choices[0].message.content.strip()


class Generator:
    """Builds the prompt and asks the chat model for a grounded answer."""

    def __init__(self, chat: ChatModel | None = None):
        self.chat = chat or HuggingFaceChat()

    def answer(self, question: str, retrieved: list[Retrieved]) -> str:
        if not retrieved:
            return NO_CONTEXT_MESSAGE
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, retrieved)},
        ]
        return self.chat(messages)
