"""
rag_pdf_pipeline.py
High-precision RAG pipeline using Docling for extraction and custom element-aware chunking.
Uses llm_utils_observed for LLM calls and includes CLI query support.
Telemetry decorators are now applied at the pipeline level for better modularity.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

from jet.adapters.llama_cpp.chunk_strategies._common import detect_text_overlap
from jet.adapters.llama_cpp.chunking_utils import _get_size_fn
from jet.adapters.llama_cpp.config import EMBED_MODEL, LLM_MODEL, PHOENIX_BASE_URL
from jet_telemetry import (
    chain,
    embedding,
    initialize_telemetry,
    llm,
    reranker,
    retriever,
    tool,
)

initialize_telemetry(service_name="rag-pdf-pipeline", endpoint=PHOENIX_BASE_URL)

from jet.adapters.langchain.factory import get_openai_embeddings
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.llm_reranker_utils import LLMReranker
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.adapters.llama_cpp.rag.pdf.pdf_extractor import PdfExtractor
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVE_K = 50
RERANK_TOP_N = 10
DEFAULT_PDF = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/data/Resume Latest - Jethro Estrada.pdf"
DEFAULT_QUERY = "Summarize this job seeker's resume"

# Initialize once outside the function if possible, or keep inside for simplicity
_reranker_instance = LLMReranker(model=LLM_MODEL)


@tool(
    name="extract-pdf-content", description="Extracts structured text from a PDF file."
)
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


@embedding(model_name=EMBED_MODEL)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Wrapper for embedding to ensure telemetry capture."""
    # Using the existing embed utility which handles batching and prefixes
    embeddings = embed(texts, model=EMBED_MODEL, show_progress=False, batch_size=64)

    # Ensure we return a plain Python list of lists to avoid NumPy truthiness issues
    # and ensure compatibility with LangChain/FAISS
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def build_vectorstore(docs: List[Document], save_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents using the LangChain factory."""
    if not docs:
        raise ValueError("Cannot build vectorstore: No documents provided.")

    # Get the standardized embedding object from the factory
    embeddings = get_openai_embeddings(embed_model=EMBED_MODEL)

    # FAISS.from_documents handles the embedding of texts internally
    vectorstore = FAISS.from_documents(docs, embeddings)

    if save_path:
        vectorstore.save_local(save_path)
    return vectorstore


@retriever(name="vector-search", model_name="cosine-similarity")
def retrieve_candidates(vectorstore: FAISS, query: str, k: int) -> List[Document]:
    """Wrapper for retrieval to ensure telemetry capture."""
    # This now works because vectorstore has a valid embedding_function
    return vectorstore.similarity_search(query, k=k)


@reranker(name="llm-grammar-reranker", model_name="qwen3.5-uncensored:2b")
def rerank_documents(query: str, documents: List[str], top_n: int) -> List[Dict]:
    """Wrapper for LLM-based reranking with GBNF grammar."""
    if not documents:
        return []

    # LLMReranker returns a list of dicts with 'index', 'score', and optionally 'reason'
    # We need to map this to the expected format: {'rank', 'index', 'score', 'raw_score', 'text'}
    ranked_results = _reranker_instance.rerank(
        query=query,
        documents=documents,
        top_k=top_n,
        min_score=7.0,  # Adjust based on your LLM's scoring tendency
        include_reasoning=False,  # Set to True if you want reasons in logs
        max_doc_length=200,  # Keep this low to save tokens
    )

    # Map LLMReranker output to the expected RerankResult format
    final_results = []
    for rank_pos, item in enumerate(ranked_results, start=1):
        final_results.append(
            {
                "rank": rank_pos,
                "index": item["index"],
                "score": item["score"] / 10.0,  # Normalize 0-10 score to 0-1 if needed
                "raw_score": item["score"],
                "text": documents[item["index"]],
            }
        )

    return final_results


@chain(name="rag-query-execution")
def execute_rag_query(
    vectorstore: FAISS, query: str, original_docs: List[Document]
) -> Dict[str, Any]:
    """Retrieve, rerank, and prepare context for LLM."""
    # 1. Retrieve
    candidates = retrieve_candidates(vectorstore, query, k=RETRIEVE_K)

    if not candidates:
        return {"answer": "No relevant context found.", "citations": []}

    doc_texts = [d.page_content for d in candidates]

    # 2. Rerank
    ranked_results = rerank_documents(
        query=query, documents=doc_texts, top_n=RERANK_TOP_N
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


@chain(name="rag-pdf-pipeline")
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

    # 1. Extract (Child span via @tool)
    doc, markdown_content = load_and_extract_pdf(pdf_path)

    # 2. Chunk (Child span via @chain)
    docs = chunk_docling_document(doc, max_tokens=max_tokens, chunk_overlap=overlap)
    if not docs:
        logger.error("Pipeline failed: No chunks were generated from the document.")
        return

    # 3. Index (Child span containing @embedding)
    vs = build_vectorstore(
        docs, save_path="faiss_docling_index" if save_index else None
    )

    # 4. Retrieve & Rerank (Child span via @chain, containing @retriever and @reranker)
    result = execute_rag_query(vs, query, docs)

    # 5. Generate Answer (Child span via @llm)
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
