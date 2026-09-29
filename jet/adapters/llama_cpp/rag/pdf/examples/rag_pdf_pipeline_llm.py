"""
rag_pdf_pipeline.py
High-precision RAG pipeline using Docling for extraction and custom element-aware chunking.
Uses llm_utils_observed for LLM calls and includes CLI query support.
Telemetry decorators are now applied at the pipeline level for better modularity.
"""

import argparse
import logging

from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet_telemetry import chain, initialize_telemetry, llm, tool

initialize_telemetry(service_name="rag-pdf-pipeline-llm", endpoint=PHOENIX_BASE_URL)
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.adapters.llama_cpp.rag.pdf.docling_chunker import chunk_docling_document
from jet.adapters.llama_cpp.rag.pdf.pdf_extractor import PdfExtractor
from jet.adapters.llama_cpp.rag.pdf.vector_store_utils_llm import (
    build_vectorstore,
    execute_rag_query,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PDF = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/data/Resume Latest - Jethro Estrada.pdf"
DEFAULT_QUERY = "Summarize this job seeker's resume"


@tool(
    name="extract-pdf-content", description="Extracts structured text from a PDF file."
)
def load_and_extract_pdf(pdf_path: str):
    """Load PDF and extract DoclingDocument and Markdown content."""
    extractor = PdfExtractor()
    doc = extractor.extract_from_path(pdf_path)
    markdown_content = extractor.export_to_markdown(doc)
    return doc, markdown_content


@llm(model_name="qwen3.5-uncensored:2b")
def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """Wrapper for LLM generation to ensure telemetry capture."""
    llm_result = chat(
        prompt_or_messages=user_prompt,
        system_message=system_prompt,
        project_name="rag-pdf-answer",
        temperature=0.1,
    )
    return llm_result.content


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-precision RAG pipeline with Docling and observed LLM."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=DEFAULT_PDF,
        help="Path to PDF file (defaults to Resume Latest - Jethro Estrada.pdf)",
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Search query (defaults to 'Summarize this job seeker's resume')",
    )
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        default=450,
        help="Maximum tokens per chunk (default: 450)",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping tokens between chunks (default: 50)",
    )
    parser.add_argument(
        "--save-index", action="store_true", help="Save FAISS index locally"
    )
    return parser.parse_args()


@chain(name="rag-pdf-pipeline-llm")
def main(
    pdf_path: str,
    query: str,
    max_tokens: int = 450,
    overlap: int = 50,
    save_index: bool = False,
):
    """Main pipeline execution. Decorated with @chain to serve as the single root span."""
    logger.info(
        f"Starting RAG pipeline for: {pdf_path} (max_tokens={max_tokens}, overlap={overlap})"
    )

    # Extraction
    doc, markdown_content = load_and_extract_pdf(pdf_path)

    # Chunking
    docs = chunk_docling_document(doc, max_tokens=max_tokens, chunk_overlap=overlap)

    if not docs:
        logger.error("Pipeline failed: No chunks were generated from the document.")
        return

    # Indexing
    vs = build_vectorstore(
        docs, save_path="faiss_docling_index" if save_index else None
    )

    # Retrieval & Reranking
    result = execute_rag_query(vs, query, docs)

    # Generation
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'. Include citation IDs."
    )
    user_prompt = f"Question: {query}\nContext:\n{result['context']}"

    logger.info("Generating answer using observed LLM...")
    answer = generate_answer(system_prompt, user_prompt)

    print("\n=== ANSWER ===\n", answer)
    print("\n=== SOURCES ===")
    for c in result["citations"]:
        print(c)


if __name__ == "__main__":
    args = get_args()
    main(args.pdf, args.query, args.max_tokens, args.overlap, args.save_index)
