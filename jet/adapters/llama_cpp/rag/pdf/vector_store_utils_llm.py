"""
vector_store_utils.py
Utilities for building vector stores and executing RAG queries.
"""

import logging
from typing import Any, Dict, List

from jet.adapters.langchain.factory import get_openai_embeddings
from jet.adapters.llama_cpp.config import EMBED_MODEL, LLM_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.llm_reranker_utils import LLMReranker
from jet_telemetry import embedding, reranker, retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Initialize once outside the function if possible, or keep inside for simplicity
_reranker_instance = LLMReranker(model=LLM_MODEL)


@embedding(model_name=EMBED_MODEL)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Wrapper for embedding to ensure telemetry capture."""
    embeddings = embed(texts, model=EMBED_MODEL, show_progress=False, batch_size=64)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return embeddings


def build_vectorstore(docs: List[Document], save_path: str = None) -> FAISS:
    """Build a FAISS vector store from documents using the LangChain factory."""
    if not docs:
        raise ValueError("Cannot build vectorstore: No documents provided.")

    embeddings = get_openai_embeddings(embed_model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(docs, embeddings)

    if save_path:
        vectorstore.save_local(save_path)

    return vectorstore


@retriever(name="vector-search", model_name="cosine-similarity")
def retrieve_candidates(vectorstore: FAISS, query: str, k: int) -> List[Document]:
    """Wrapper for retrieval to ensure telemetry capture."""
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


def execute_rag_query(
    vectorstore: FAISS,
    query: str,
    original_docs: List[Document],
    retrieve_k: int = 50,
    rerank_top_n: int = 10,
) -> Dict[str, Any]:
    """Retrieve, rerank, and prepare context for LLM."""
    candidates = retrieve_candidates(vectorstore, query, k=retrieve_k)

    if not candidates:
        return {"answer": "No relevant context found.", "citations": []}

    doc_texts = [d.page_content for d in candidates]
    ranked_results = rerank_documents(
        query=query, documents=doc_texts, top_n=rerank_top_n
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
