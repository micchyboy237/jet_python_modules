"""
Demo: Custom Span Kinds & Hybrid Search
Covers: Separate span functions for vector/bm25, manual embedding spans,
        and LLM-powered relevance evaluation as top-level siblings.
"""

import asyncio
import math

import nltk
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_BASE_URL,
)
from jet_telemetry import chain, get_trace_url, initialize_telemetry, tool
from nltk.tokenize import word_tokenize
from openai import AsyncOpenAI, OpenAI
from rank_bm25 import BM25Okapi

# Download NLTK data if not present
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Disable auto_instrument to prevent OpenAI from capturing raw embedding vectors
initialize_telemetry(
    service_name="custom-spans-demo", endpoint=PHOENIX_BASE_URL, auto_instrument=False
)

embed_client = OpenAI(base_url=EMBED_BASE_URL, api_key="sk-local")
llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


@tool(name=EMBED_MODEL)
def embed_text(text: str) -> list[float]:
    """
    Embeds text using the configured model.
    Span is named after the model and excludes raw vector from output.value.
    """
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=text)
    embedding = resp.data[0].embedding

    # Manually set clean metadata instead of letting auto-instrument capture the vector
    from openinference.semconv.trace import SpanAttributes
    from opentelemetry import trace as otel_trace

    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, f"Embedding generated ({len(embedding)} dims)"
        )
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
        span.set_attribute("embedding.model", EMBED_MODEL)
        span.set_attribute("embedding.dimensions", len(embedding))

    return embedding


@tool(name="vector-search-retriever")
def vector_search(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    """
    Performs semantic search using cosine similarity on embeddings.
    Returns top-k documents with their semantic scores.
    """
    query_emb = embed_text(query)
    scored_docs = []

    for i, doc in enumerate(documents):
        doc_emb = embed_text(doc)
        score = _cosine_similarity(query_emb, doc_emb)
        scored_docs.append({"index": i, "content": doc, "sem_score": round(score, 4)})

    scored_docs.sort(key=lambda x: x["sem_score"], reverse=True)
    return scored_docs[:top_k]


@tool(name="bm25-reranker")
def bm25_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Reranks candidate documents using BM25 lexical scoring.
    Takes pre-retrieved candidates and boosts them based on keyword overlap.
    """
    if not candidates:
        return []

    # Tokenize query and documents
    tokenized_query = word_tokenize(query.lower())
    tokenized_docs = [word_tokenize(c["content"].lower()) for c in candidates]

    # Initialize BM25
    bm25 = BM25Okapi(tokenized_docs)

    # Get BM25 scores
    bm25_scores = bm25.get_scores(tokenized_query)

    # Update candidates with BM25 scores
    for i, candidate in enumerate(candidates):
        candidate["lex_score"] = round(float(bm25_scores[i]), 4)
        # Calculate hybrid score (70% Semantic, 30% Lexical)
        candidate["hybrid_score"] = round(
            (0.7 * candidate["sem_score"]) + (0.3 * candidate["lex_score"]), 4
        )

    # Sort by hybrid score
    candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return candidates


@tool(name="hybrid_search")
def hybrid_search_pipeline(query: str, documents: list[str]) -> dict:
    """
    Top-level span that orchestrates vector search and BM25 reranking.
    """
    print(f"\n📐 Running vector search...")
    vector_results = vector_search(query, documents, top_k=3)

    print(f"\n📊 Running BM25 reranking...")
    final_results = bm25_rerank(query, vector_results)

    return {
        "query": query,
        "retrieved": final_results,
        "contents_for_llm": [d["content"] for d in final_results],
    }


@tool(name="llm_relevance_evaluator")
async def llm_relevance_check(query: str, documents: list[str]) -> dict:
    """
    Top-level span that uses an LLM to semantically evaluate relevance.
    """
    doc_context = "\n".join(
        [f"[Doc {i + 1}]: {doc}" for i, doc in enumerate(documents)]
    )

    prompt = f"""
    You are a relevance evaluator. 
    Query: "{query}"
    
    Retrieved Documents:
    {doc_context}
    
    For each document, provide:
    1. A relevance score from 0.0 (irrelevant) to 1.0 (perfectly relevant).
    2. A one-sentence justification.
    
    Return ONLY a valid JSON array of objects:
    [
      {{"doc_index": 0, "score": 0.9, "justification": "..."}}
    ]
    """

    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        import json

        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        return []
    except Exception as e:
        return [{"error": str(e)}]


@chain(name="hybrid-search-workflow")
async def run_search_demo():
    """Root chain for the demo."""
    docs = [
        "Machine learning models require careful tuning of hyperparameters like learning rate.",
        "The weather in San Francisco is typically mild and foggy year-round.",
        "Neural networks can approximate complex non-linear functions using backpropagation.",
        "PostgreSQL supports advanced indexing strategies such as GIN and GiST for fast queries.",
        "Transformers revolutionized natural language processing in 2017 with attention mechanisms.",
    ]

    query = "neural network optimization"
    print(f"\n🔍 Starting hybrid search for: '{query}'")

    # Step 1: Hybrid Retrieval (Top-level span 1)
    retrieval_result = hybrid_search_pipeline(query, docs)

    # Step 2: LLM Relevance Check (Top-level span 2)
    print(f"\n🧠 Evaluating relevance with LLM...")
    relevance_scores = await llm_relevance_check(
        query, retrieval_result["contents_for_llm"]
    )

    print(f"\n📋 Final Results:")
    for i, res in enumerate(retrieval_result["retrieved"]):
        score_info = next((s for s in relevance_scores if s.get("doc_index") == i), {})
        llm_score = score_info.get("score", "N/A")
        justification = score_info.get("justification", "No justification provided.")

        print(f"\n   {i + 1}. [Hybrid: {res['hybrid_score']}] [LLM: {llm_score}]")
        print(f"       Content: {res['content'][:60]}...")
        print(f"       LLM Justification: {justification}")

    if url := get_trace_url(PHOENIX_BASE_URL):
        print(f"\n🔍 View complete trace: {url}")


if __name__ == "__main__":
    asyncio.run(run_search_demo())
