"""
Demo: Custom Span Kinds & LLM-Based Relevance Check
Covers: @trace with custom kind, sync/async mixing, hybrid search logic,
        and LLM-powered relevance evaluation.
"""

import asyncio
import math
import time

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_BASE_URL,
)
from jet_telemetry import chain, get_trace_url, initialize_telemetry, tool
from openai import AsyncOpenAI, OpenAI

initialize_telemetry(service_name="custom-spans-demo", endpoint=PHOENIX_BASE_URL)
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


@tool(name="llm_relevance_evaluator")
async def llm_relevance_check(query: str, documents: list[str]) -> dict:
    """
    Uses an LLM to semantically evaluate the relevance of retrieved documents.
    Returns a structured assessment with scores and justifications.
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
        # Simple extraction for demo purposes; in prod use response_format
        import json

        start = content.find("[")
        end = content.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        return []
    except Exception as e:
        return [{"error": str(e)}]


@tool(name="hybrid_search_pipeline")
def hybrid_search_pipeline(query: str, documents: list[str]) -> dict:
    """
    Performs hybrid search (Semantic + Lexical) and evaluates relevance via LLM.
    """
    start_time = time.time()

    # 1. Embed Query
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=query)
    query_emb = resp.data[0].embedding

    # 2. Score Documents (Semantic + Lexical)
    scored_docs = []
    for i, doc in enumerate(documents):
        doc_resp = embed_client.embeddings.create(model=EMBED_MODEL, input=doc)
        doc_emb = doc_resp.data[0].embedding
        sem_score = _cosine_similarity(query_emb, doc_emb)

        # Lexical Score (BM25-lite)
        q_words = set(query.lower().split())
        d_words = set(doc.lower().split())
        lex_score = len(q_words & d_words) / len(q_words) if q_words else 0

        final_score = (0.7 * sem_score) + (0.3 * lex_score)
        scored_docs.append(
            {
                "index": i,
                "content": doc,
                "hybrid_score": round(final_score, 4),
            }
        )

    # 3. Rank and Retrieve Top-K
    scored_docs.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_k = 3
    retrieved = scored_docs[:top_k]
    retrieved_contents = [d["content"] for d in retrieved]

    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "query": query,
        "retrieved": retrieved,
        "latency_ms": round(elapsed_ms, 1),
        "contents_for_llm": retrieved_contents,
    }


@chain(name="llm-enhanced-search-workflow")
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

    # Step 1: Hybrid Retrieval
    retrieval_result = hybrid_search_pipeline(query, docs)

    # Step 2: LLM Relevance Check
    print(f"\n🧠 Evaluating relevance with LLM...")
    relevance_scores = await llm_relevance_check(
        query, retrieval_result["contents_for_llm"]
    )

    print(f"\n📊 Final Results:")
    print(f"   Latency (Retrieval): {retrieval_result['latency_ms']}ms\n")

    for i, res in enumerate(retrieval_result["retrieved"]):
        score_info = next(
            (s for s in relevance_scores if s.get("doc_index") == res["index"]), {}
        )
        llm_score = score_info.get("score", "N/A")
        justification = score_info.get("justification", "No justification provided.")

        print(f"   {i + 1}. [Hybrid: {res['hybrid_score']}] [LLM: {llm_score}]")
        print(f"       Content: {res['content'][:60]}...")
        print(f"       LLM Justification: {justification}\n")

    if url := get_trace_url(PHOENIX_BASE_URL):
        print(f"🔍 View complete trace: {url}")


if __name__ == "__main__":
    asyncio.run(run_search_demo())
