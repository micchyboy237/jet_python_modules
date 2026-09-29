"""
rag_pdf_pipeline.py
High-precision RAG pipeline using Docling for extraction and SmartChunker for processing.
Uses llm_utils_observed for LLM calls and includes CLI query support.
"""

import argparse
import logging
from typing import Any, Dict, List, Tuple

from jet.adapters.llama_cpp.config import EMBED_MODEL, PHOENIX_BASE_URL
from jet_telemetry import chain, initialize_telemetry

# Initialize telemetry for the pipeline
initialize_telemetry(service_name="rag-pdf-pipeline", endpoint=PHOENIX_BASE_URL)

from jet.adapters.llama_cpp.chunk_strategies import get_chunker
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


def load_and_extract_pdf(pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Load PDF and extract both markdown and structured elements."""
    extractor = PdfExtractor()
    doc = extractor.extract_from_path(pdf_path)
    markdown_content = extractor.export_to_markdown(doc)
    elements = extractor.extract_elements(doc)
    return markdown_content, elements


def smart_chunk_document(
    markdown_content: str,
    elements: List[Dict[str, Any]],
    model: str = "qwen3.5-uncensored:2b",
) -> List[Document]:
    """Chunk document using SmartChunker with element-aware formatting."""
    chunker = get_chunker(strategy="smart", model=model)

    # SmartChunker can take elements directly for atomic preservation
    chunks = chunker.chunk(
        text=markdown_content,
        chunk_size=512,
        chunk_overlap=50,
        elements=elements,
        retrieval_type="dense",
    )

    # Apply RAG formatting markers based on element types if available
    # Note: SmartChunker already handles some of this, but we can reinforce it
    docs = []
    for i, chunk_text in enumerate(chunks):
        docs.append(Document(page_content=chunk_text, metadata={"chunk_index": i}))

    logger.info(f"Generated {len(docs)} chunks from document.")
    return docs


def build_vectorstore(docs: List[Document], save_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents."""
    texts = [d.page_content for d in docs]
    embeddings = embed(texts, model=EMBED_MODEL, show_progress=True)

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
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("-q", "--query", type=str, default=None, help="Search query")
    parser.add_argument(
        "--save-index", action="store_true", help="Save FAISS index locally"
    )
    return parser.parse_args()


def main(pdf_path: str, query: str, save_index: bool = False):
    """Main pipeline execution."""
    logger.info(f"Starting RAG pipeline for: {pdf_path}")

    # 1. Extract
    markdown_content, elements = load_and_extract_pdf(pdf_path)

    # 2. Chunk
    docs = smart_chunk_document(markdown_content, elements)

    # 3. Index
    vs = build_vectorstore(
        docs, save_path="faiss_docling_index" if save_index else None
    )

    # 4. Query
    if not query:
        query = "How does the authentication workflow handle token refresh?"

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
    main(args.pdf, args.query, args.save_index)
