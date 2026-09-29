"""
vector_store_utils.py
Utilities for building vector stores and executing RAG queries.
"""

import logging
from typing import Any, Dict, List

from jet.adapters.langchain.factory import get_openai_embeddings
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.rerank_utils import rerank as rerank_utility
from jet_telemetry import embedding, reranker, retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


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


@reranker(name="cross-encoder-reranker", model_name="auto")
def rerank_documents(query: str, documents: List[str], top_n: int) -> List[Dict]:
    """Wrapper for reranking to ensure telemetry capture."""
    return rerank_utility(query=query, documents=documents, top_n=top_n, method="auto")


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
