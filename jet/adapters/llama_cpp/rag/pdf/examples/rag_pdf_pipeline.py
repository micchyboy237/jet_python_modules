"""
rag_pdf_pipeline.py
High-precision RAG pipeline using Docling for extraction and custom element-aware chunking.
Updated to reuse features from jet.adapters.langchain.* for Embeddings and Chat Models.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

from jet.adapters.langchain.factory import get_chat_openai, get_openai_embeddings
from jet.adapters.llama_cpp.chunk_strategies._common import detect_text_overlap
from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from jet.adapters.llama_cpp.config import EMBED_MODEL, PHOENIX_BASE_URL
from jet.adapters.llama_cpp.rerank_utils import rerank
from jet_telemetry import chain, initialize_telemetry
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

initialize_telemetry(service_name="rag-pdf-pipeline", endpoint=PHOENIX_BASE_URL)

from jet.adapters.llama_cpp.rag.pdf.pdf_extractor import PdfExtractor

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

        if elem_type in ["TableItem", "CodeItem", "FormulaItem"]:
            if current_text_parts:
                chunk_text = "\n".join(current_text_parts)
                chunks.append(Document(page_content=chunk_text))
                previous_chunk_text = chunk_text
                current_text_parts = []
                current_token_count = 0

            if elem_type == "TableItem":
                content = item.export_to_html(doc)
                chunks.append(
                    Document(page_content=content, metadata={"type": elem_type})
                )
                previous_chunk_text = content
            continue

        if content:
            item_tokens = len(size_fn(content))
            if current_token_count + item_tokens > max_tokens and current_text_parts:
                chunk_text = "\n".join(current_text_parts)
                overlap_text = ""
                if chunk_overlap > 0 and previous_chunk_text:
                    overlap_candidate, _ = detect_text_overlap(
                        previous_chunk_text, chunk_text, size_fn
                    )
                    if overlap_candidate:
                        overlap_text = overlap_candidate

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

    if current_text_parts:
        chunk_text = "\n".join(current_text_parts)
        chunks.append(Document(page_content=chunk_text))

    logger.info(
        f"Generated {len(chunks)} chunks from DoclingDocument (max_tokens={max_tokens}, overlap={chunk_overlap})."
    )
    return chunks


def build_vectorstore(docs: List[Document], save_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents using LangChain embeddings."""
    if not docs:
        raise ValueError("Cannot build vectorstore: No documents provided.")

    # Reuse existing feature: Get LangChain-compatible embeddings
    embeddings = get_openai_embeddings(embed_model=EMBED_MODEL)

    # Use LangChain's built-in from_documents method
    vectorstore = FAISS.from_documents(docs, embeddings)

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

    # Reuse existing feature: Reranking
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

    # 2. Chunk
    docs = chunk_docling_document(doc, max_tokens=max_tokens, chunk_overlap=overlap)
    if not docs:
        logger.error("Pipeline failed: No chunks were generated from the document.")
        return

    # 3. Index (Reusing jet.adapters.langchain.factory for embeddings)
    vs = build_vectorstore(
        docs, save_path="faiss_docling_index" if save_index else None
    )

    # 4. Retrieve & Rerank
    result = execute_rag_query(vs, query, docs)

    # 5. Generate Answer (Reusing jet.adapters.langchain.factory for Chat Model)
    system_prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know'. Include citation IDs."
    )
    user_prompt = f"Question: {query}\nContext:\n{result['context']}"

    logger.info("Generating answer using ChatLlamaCpp...")

    # Initialize LangChain-compatible chat model
    llm = get_chat_openai(temperature=0.1, agent_name="rag-pdf-answer", verbose=True)

    # Invoke using LangChain interface
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    response = llm.invoke(messages)

    print("\n=== ANSWER ===\n", response.content)
    print("\n=== SOURCES ===")
    for c in result["citations"]:
        print(c)


if __name__ == "__main__":
    args = get_args()
    main(args.pdf, args.query, args.max_tokens, args.overlap, args.save_index)
