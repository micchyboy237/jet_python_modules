"""
Demo: Custom Span Kinds & Sync Functions
Covers: @trace with custom kind, sync decorators, EMBEDDING span kind,
        trace URL returned as part of result dict.
"""

import time

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_BASE_URL_LG,
    EMBED_MODEL,
    EMBED_MODEL_LG,
    PHOENIX_BASE_URL,
)
from jet_telemetry import get_trace_url, initialize_telemetry, tool, trace
from openai import OpenAI

initialize_telemetry(service_name="custom-spans-demo", endpoint=PHOENIX_BASE_URL)

# Initialize Sync OpenAI clients
embed_client_sm = OpenAI(base_url=EMBED_BASE_URL, api_key="sk-local")
embed_client_lg = OpenAI(base_url=EMBED_BASE_URL_LG, api_key="sk-local")


@trace(kind="EMBEDDING", name="embed-query-small")
def embed_query_small(query: str) -> list[float]:
    """Sync embedding with small model - demonstrates sync support + custom kind."""
    resp = embed_client_sm.embeddings.create(model=EMBED_MODEL, input=query)
    return resp.data[0].embedding


@trace(kind="EMBEDDING", name="embed-doc-large")
def embed_doc_large(document: str) -> list[float]:
    """Sync embedding with large model for higher quality."""
    resp = embed_client_lg.embeddings.create(model=EMBED_MODEL_LG, input=document)
    return resp.data[0].embedding


@trace(kind="RETRIEVER", name="hybrid-search")
def hybrid_search(
    query_emb: list[float], doc_embs: list[list[float]], top_k: int = 3
) -> list[int]:
    """Custom RETRIEVER span for hybrid search logic."""
    # Simplified cosine similarity
    scores = []
    for i, doc_emb in enumerate(doc_embs):
        dot = sum(a * b for a, b in zip(query_emb, doc_emb))
        norm_a = sum(a * a for a in query_emb) ** 0.5
        norm_b = sum(b * b for b in doc_emb) ** 0.5
        scores.append((i, dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in scores[:top_k]]


@trace(kind="EVALUATOR", name="relevance-check")
def evaluate_relevance(query: str, retrieved_docs: list[str]) -> dict:
    """Custom EVALUATOR span for quality assessment."""
    # Simple keyword-based evaluation (replace with LLM evaluator in prod)
    keywords = set(query.lower().split())
    scores = []
    for doc in retrieved_docs:
        overlap = len(keywords & set(doc.lower().split()))
        scores.append(min(overlap / max(len(keywords), 1), 1.0))

    avg_score = sum(scores) / len(scores) if scores else 0
    return {"avg_relevance": round(avg_score, 3), "passed": avg_score > 0.3}


@tool(name="batch-embed-and-search")
def batch_embed_and_search(query: str, documents: list[str]) -> dict:
    """Orchestrates sync embedding, retrieval, and evaluation."""
    start = time.time()

    # Embed query and docs (sync)
    q_emb = embed_query_small(query)
    d_embs = [embed_doc_large(doc) for doc in documents]

    # Retrieve
    top_indices = hybrid_search(q_emb, d_embs, top_k=3)
    retrieved = [documents[i] for i in top_indices]

    # Evaluate
    eval_result = evaluate_relevance(query, retrieved)

    elapsed = time.time() - start
    return {
        "retrieved_count": len(retrieved),
        "evaluation": eval_result,
        "latency_ms": round(elapsed * 1000, 1),
        "trace_url": get_trace_url(PHOENIX_BASE_URL),
    }


def main():
    docs = [
        "Machine learning models require careful tuning of hyperparameters.",
        "The weather in San Francisco is typically mild year-round.",
        "Neural networks can approximate complex non-linear functions.",
        "PostgreSQL supports advanced indexing strategies for fast queries.",
        "Transformers revolutionized natural language processing in 2017.",
    ]

    result = batch_embed_and_search("neural network optimization", docs)

    print(f"\n📊 Search Results:")
    print(f"   Retrieved: {result['retrieved_count']} docs")
    print(f"   Relevance: {result['evaluation']['avg_relevance']}")
    print(f"   Latency:   {result['latency_ms']}ms")

    if result.get("trace_url"):
        print(f"🔍 View complete trace: {result['trace_url']}")


if __name__ == "__main__":
    main()
