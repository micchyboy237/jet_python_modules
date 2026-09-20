"""
Demo: Full RAG Pipeline with Telemetry & OpenAI Streaming
Covers: @chain, @tool, @llm, async support, model_name attribution
"""

import asyncio

import httpx
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DOC_PREFIX,
    EMBED_MODEL,
    EMBED_QUERY_PREFIX,
    LLM_BASE_URL,
    LLM_MODEL,
    PHOENIX_BASE_URL,
    RERANK_BASE_URL,
    RERANK_MODEL,
)
from jet_telemetry import chain, initialize_telemetry, llm, tool
from openai import AsyncOpenAI

# Initialize telemetry once at startup
initialize_telemetry(
    service_name="rag-pipeline-demo",
    endpoint=PHOENIX_BASE_URL,
    auto_instrument=True,
    batch=True,
)

# Initialize AsyncOpenAI client
llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")
embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key="sk-local")


@tool(description="Embeds text using local nomic-embed model")
async def embed_text(text: str, is_query: bool = False) -> list[float]:
    prefix = EMBED_QUERY_PREFIX if is_query else EMBED_DOC_PREFIX
    resp = await embed_client.embeddings.create(
        model=EMBED_MODEL, input=f"{prefix}{text}"
    )
    return resp.data[0].embedding


@tool(description="Retrieves top-k documents from vector store")
async def retrieve_documents(
    query_embedding: list[float], top_k: int = 5
) -> list[dict]:
    # Simulated vector DB call
    await asyncio.sleep(0.05)
    return [
        {
            "id": i,
            "content": f"Document {i} content about AI safety and alignment principles.",
            "score": 0.9 - i * 0.1,
        }
        for i in range(top_k)
    ]


@tool(description="Reranks documents using cross-encoder")
async def rerank_documents(
    query: str, documents: list[dict], top_n: int = 3
) -> list[dict]:
    if not RERANK_BASE_URL or not RERANK_MODEL:
        return documents[:top_n]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{RERANK_BASE_URL}/rerank",
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [d["content"] for d in documents],
                "top_n": top_n,
            },
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        return [
            {"content": documents[r["index"]]["content"], "score": r["relevance_score"]}
            for r in results
        ]


@llm(model_name=LLM_MODEL)
async def generate_answer(query: str, context: list[str]) -> str:
    """Streaming LLM call with natural output flushing."""
    messages = [
        {"role": "system", "content": "Answer based ONLY on the provided context."},
        {
            "role": "user",
            "content": f"Context:\n{''.join(context)}\n\nQuestion: {query}",
        },
    ]

    print("\n🤖 LLM Response: ", end="", flush=True)
    collected_content = []

    stream = await llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                # Skip thinking content if present despite disable flag
                continue
            if delta.content:
                collected_content.append(delta.content)
                print(delta.content, end="", flush=True)

    print("\n", flush=True)
    return "".join(collected_content)


@chain(name="full-rag-pipeline")
async def rag_pipeline(query: str) -> dict:
    """End-to-end RAG: Embed → Retrieve → Rerank → Generate"""
    query_emb = await embed_text(query, is_query=True)
    docs = await retrieve_documents(query_emb, top_k=10)
    ranked_docs = await rerank_documents(query, docs, top_n=3)
    context = [d["content"] for d in ranked_docs]
    answer = await generate_answer(query, context)

    return {"query": query, "answer": answer, "sources": len(ranked_docs)}


async def main():
    result = await rag_pipeline("What are the key principles of AI alignment?")
    print(f"\n✅ Sources used: {result['sources']}")
    print(f"🔍 View traces at: {PHOENIX_BASE_URL}")


if __name__ == "__main__":
    asyncio.run(main())
