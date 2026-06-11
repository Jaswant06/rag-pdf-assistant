"""Gradio web app for the RAG PDF Assistant.

Upload a PDF, ask questions about it, and get answers grounded in the document
with the page numbers they came from.
"""

import gradio as gr

from src import RagPipeline

# Loaded once: the embedding model and the generator are shared across requests.
pipeline = RagPipeline()

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
)

CSS = """
#wrap { max-width: 920px; margin: 0 auto; padding-bottom: 28px; }
#hero { margin: 8px 0 20px; }
#card {
  background: var(--background-fill-primary);
  border-radius: 18px; padding: 20px 20px 12px;
  box-shadow: 0 12px 34px rgba(2, 6, 23, 0.07);
  border: 1px solid rgba(2, 6, 23, 0.05);
}
#ask-btn {
  background: linear-gradient(135deg, #0d9488 0%, #4f46e5 100%) !important;
  border: none !important; color: #fff !important; font-weight: 600 !important;
  box-shadow: 0 8px 22px rgba(79, 70, 229, 0.28) !important;
  transition: transform .15s ease, box-shadow .15s ease !important;
}
#ask-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 14px 32px rgba(79, 70, 229, 0.40) !important;
}
footer { display: none !important; }
"""

HERO = """
<div style="background: linear-gradient(135deg,#0f766e 0%,#0e7490 45%,#4338ca 100%);
            border-radius: 22px; padding: 48px 28px; text-align: center; color: #ffffff;
            box-shadow: 0 18px 50px rgba(67,56,202,0.28);">
  <div style="display:inline-block; padding:5px 14px; border-radius:999px; font-size:.8rem;
              letter-spacing:.08em; text-transform:uppercase; background:rgba(255,255,255,.16);
              margin-bottom:16px;">Retrieval Augmented Generation</div>
  <div style="font-size: 2.8rem; font-weight: 800; letter-spacing: -0.02em; line-height:1.08;">
    Chat with your PDF
  </div>
  <div style="font-size: 1.1rem; opacity: .92; margin: 16px auto 0; max-width: 600px; line-height:1.55;">
    Upload a document and ask questions about it. Answers are grounded in the
    text, with the page each one came from.
  </div>
</div>
"""


def ingest(file) -> str:
    if file is None:
        return "Upload a PDF to begin."
    try:
        count = pipeline.ingest(file.name)
    except Exception as exc:  # noqa: BLE001  surface the reason to the user
        return f"Could not read that PDF: {exc}"
    if count == 0:
        return "No readable text found. This looks like a scanned (image-only) PDF."
    return f"Indexed **{count} chunks**. Ask a question below."


def ask(question: str) -> str:
    if not pipeline.ready:
        return "Please upload a PDF first."
    if not question.strip():
        return "Type a question about the document."
    try:
        result = pipeline.answer(question)
    except Exception as exc:  # noqa: BLE001  the answer model was unreachable
        return f"The answer model could not be reached right now ({type(exc).__name__}). Please try again."
    pages = ", ".join(f"p. {r.chunk.page}" for r in result.sources)
    footer = f"\n\n<sub>Retrieved from: {pages}</sub>" if pages else ""
    return result.text + footer


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=THEME, css=CSS, title="RAG PDF Assistant") as demo:
        with gr.Column(elem_id="wrap"):
            gr.HTML(HERO, elem_id="hero")
            with gr.Column(elem_id="card"):
                pdf_in = gr.File(label="Your PDF", file_types=[".pdf"])
                status = gr.Markdown("Upload a PDF to begin.")
                question_in = gr.Textbox(
                    label="Your question", lines=2,
                    placeholder="What does the document say about ...?",
                )
                ask_btn = gr.Button("Ask", variant="primary", size="lg", elem_id="ask-btn")
                answer_out = gr.Markdown()

            pdf_in.change(ingest, inputs=pdf_in, outputs=status)
            ask_btn.click(ask, inputs=question_in, outputs=answer_out)
            question_in.submit(ask, inputs=question_in, outputs=answer_out)
    return demo


if __name__ == "__main__":
    build_ui().launch()
