"""
rag_pdf_pipeline.py
High-precision RAG pipeline using Docling for extraction and custom element-aware chunking.
Uses llm_utils_observed for LLM calls and includes CLI query support.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

from jet.adapters.llama_cpp.chunk_strategies._common import detect_text_overlap
from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from jet.adapters.llama_cpp.config import EMBED_MODEL, PHOENIX_BASE_URL
from jet_telemetry import chain, initialize_telemetry

# Initialize telemetry for the pipeline
initialize_telemetry(service_name="rag-pdf-pipeline", endpoint=PHOENIX_BASE_URL)

from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.adapters.llama_cpp.rag.pdf.pdf_extractor import PdfExtractor
from jet.adapters.llama_cpp.rerank_utils import rerank
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RETRIEVE_K = 50
RERANK_TOP_N = 10

DEFAULT_PDF = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/data/Resume Latest - Jethro Estrada.pdf"
DEFAULT_QUERY = "Summarize this job seeker's resume"


def load_and_extract_pdf(pdf_path: str) -> Tuple[Any, str]:
    """Load PDF and extract DoclingDocument and Markdown content."""
    extractor = PdfExtractor()
    doc = extractor.extract_from_path(pdf_path)
    markdown_content = extractor.export_to_markdown(doc)
    return doc, markdown_content


@chain(name="docling-element-chunker")
def chunk_docling_document(
    doc,
    model: str = "qwen3.5-uncensored:2b",
    max_tokens: int = 450,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Chunk a DoclingDocument by iterating through its structural elements.

    This approach preserves atomic elements like tables and code blocks
    while grouping smaller text items into coherent paragraphs with token-aware overlap.

    Args:
        doc: The DoclingDocument to chunk.
        model: Model key for tokenizer resolution.
        max_tokens: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
    """
    chunks = []
    current_text_parts = []
    current_token_count = 0
    previous_chunk_text = ""

    size_fn = _get_size_fn(model)

    for item, level in doc.iterate_items():
        elem_type = type(item).__name__
        content = ""

        if hasattr(item, "text"):
            content = item.text
        elif hasattr(item, "orig"):
            content = item.orig

        # Handle atomic elements (Tables, Code, Formulas)
        if elem_type in ["TableItem", "CodeItem", "FormulaItem"]:
            # If we have accumulated text, flush it first
            if current_text_parts:
                chunk_text = "\n".join(current_text_parts)
                chunks.append(Document(page_content=chunk_text))
                previous_chunk_text = chunk_text
                current_text_parts = []
                current_token_count = 0

            # Add atomic element as its own chunk
            if elem_type == "TableItem":
                content = item.export_to_html(doc)
            chunks.append(Document(page_content=content, metadata={"type": elem_type}))
            previous_chunk_text = content
            continue

        # Handle regular text items
        if content:
            item_tokens = len(size_fn(content))

            # If adding this item exceeds the limit, flush the current buffer
            if current_token_count + item_tokens > max_tokens and current_text_parts:
                chunk_text = "\n".join(current_text_parts)

                # Calculate overlap
                overlap_text = ""
                if chunk_overlap > 0 and previous_chunk_text:
                    overlap_candidate, _ = detect_text_overlap(
                        previous_chunk_text, chunk_text, size_fn
                    )
                    if overlap_candidate:
                        overlap_text = overlap_candidate

                # Start new chunk with overlap
                if overlap_text:
                    current_text_parts = [overlap_text, content]
                    current_token_count = len(size_fn(overlap_text)) + item_tokens
                else:
                    current_text_parts = [content]
                    current_token_count = item_tokens

                chunks.append(Document(page_content=chunk_text))
                previous_chunk_text = chunk_text
            else:
                current_text_parts.append(content)
                current_token_count += item_tokens

    # Flush any remaining text
    if current_text_parts:
        chunk_text = "\n".join(current_text_parts)
        chunks.append(Document(page_content=chunk_text))

    logger.info(
        f"Generated {len(chunks)} chunks from DoclingDocument (max_tokens={max_tokens}, overlap={chunk_overlap})."
    )
    return chunks


def build_vectorstore(docs: List[Document], save_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents."""
    if not docs:
        raise ValueError("Cannot build vectorstore: No documents provided.")

    texts = [d.page_content for d in docs]

    # Use batch_size=1 to avoid 'input too large' errors on local servers
    embeddings = embed(texts, model=EMBED_MODEL, show_progress=True, batch_size=1)

    import numpy as np

    embeddings_np = np.array(embeddings)

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings_np)), embedding=None
    )

    if save_path:
        vectorstore.save_local(save_path)
    return vectorstore


@chain(name="rag-query-execution")
def execute_rag_query(
    vectorstore: FAISS, query: str, original_docs: List[Document]
) -> Dict[str, Any]:
    """Retrieve, rerank, and prepare context for LLM."""
    candidates = vectorstore.similarity_search(query, k=RETRIEVE_K)
    if not candidates:
        return {"answer": "No relevant context found.", "citations": []}

    doc_texts = [d.page_content for d in candidates]
    ranked_results = rerank(
        query=query, documents=doc_texts, top_n=RERANK_TOP_N, method="auto"
    )

    context_blocks = []
    citations = []
    for rank, res in enumerate(ranked_results):
        cid = f"chunk-{res['index']}"
        excerpt = res["text"][:400].replace("\n", " ").strip()
        context_blocks.append(f"--- {cid} ---\n{excerpt}")
        citations.append(f"{cid} (score={res['score']:.4f})")

    context = "\n".join(context_blocks)
    return {"context": context, "citations": citations}


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


def main(
    pdf_path: str,
    query: str,
    max_tokens: int = 450,
    overlap: int = 50,
    save_index: bool = False,
):
    """Main pipeline execution."""
    logger.info(
        f"Starting RAG pipeline for: {pdf_path} (max_tokens={max_tokens}, overlap={overlap})"
    )

    # 1. Extract
    doc, markdown_content = load_and_extract_pdf(pdf_path)

    # 2. Chunk using custom Docling-aware logic
    docs = chunk_docling_document(doc, max_tokens=max_tokens, chunk_overlap=overlap)

    if not docs:
        logger.error("Pipeline failed: No chunks were generated from the document.")
        return

    # 3. Index
    vs = build_vectorstore(
        docs, save_path="faiss_docling_index" if save_index else None
    )

    # 4. Query
    result = execute_rag_query(vs, query, docs)

    # 5. Generate Answer using observed LLM
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'. Include citation IDs."
    )

    user_prompt = f"Question: {query}\n\nContext:\n{result['context']}"

    logger.info("Generating answer using observed LLM...")
    llm_result = chat(
        prompt_or_messages=user_prompt,
        system_message=system_prompt,
        project_name="rag-pdf-answer",
        temperature=0.1,
    )

    print("\n=== ANSWER ===\n", llm_result.content)
    print("\n=== SOURCES ===")
    for c in result["citations"]:
        print(c)


if __name__ == "__main__":
    args = get_args()
    main(args.pdf, args.query, args.max_tokens, args.overlap, args.save_index)
