"""
Demo: Full RAG Pipeline with Telemetry & OpenAI Streaming
Covers: @chain, @embedding, @retriever, @reranker, @llm, async support, model_name attribution
Span Hierarchy:
📦 full-rag-pipeline (CHAIN)
│
├── 🧬 embed_text (EMBEDDING)
│   ├── attr: embedding.model_name = "nomic-embed:2-moe"
│   ├── attr: embedding.text = "What are the key principles..."
│   └── attr: embedding.vector_dimension = 768
│
├── 🔍 vector-store-retrieval (RETRIEVER)
│   ├── attr: retriever.model_name = "cosine-similarity"
│   └── attr: retrieval.document_count = 10
│
├── 📊 cross-encoder-reranker (RERANKER)
│   ├── attr: reranker.model_name = "bge-reranker-v2-m3" (or configured model)
│   ├── attr: reranker.query = "What are the key principles..."
│   └── attr: reranker.output_document_count = 3
│
└── 🤖 generate_answer (LLM)
    ├── attr: llm.model_name = "llama-3.2-3b-instruct"
    ├── attr: llm.provider = "llama_cpp"
    └── attr: llm.input_messages = [...]
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
from jet_telemetry import (
    chain,
    embedding,
    get_trace_url,
    initialize_telemetry,
    llm,
    reranker,
    retriever,
)
from openai import AsyncOpenAI

initialize_telemetry(
    service_name="rag-pipeline-demo",
    endpoint=PHOENIX_BASE_URL,
    auto_instrument=True,
    batch=True,
)


def get_spans_jsonl_download_url(
    phoenix_base_url: str,
    project_name: str,
    span_ids: list[str] | None = None,
    limit: int = 1000,
) -> str:
    """
    Generates a Phoenix REST API URL to download spans as JSONL.
    Args:
        phoenix_base_url: Base URL of the Phoenix instance (e.g., "http://localhost:6006").
        project_name: Name of the Phoenix project.
        span_ids: Optional list of span IDs to filter by.
        limit: Maximum number of spans to return per request.
    Returns:
        A URL to download spans as JSONL.
    """
    base = phoenix_base_url.rstrip("/")
    url = f"{base}/v1/projects/{project_name}/spans?limit={limit}"
    if span_ids:
        span_ids_str = ",".join(span_ids)
        url += f"&span_id={span_ids_str}"
    return url


llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key="sk-local")
embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key="sk-local")


@embedding(model_name=EMBED_MODEL)
async def embed_text(text: str, is_query: bool = False) -> list[float]:
    """Embeds text using local nomic-embed model."""
    prefix = EMBED_QUERY_PREFIX if is_query else EMBED_DOC_PREFIX
    resp = await embed_client.embeddings.create(
        model=EMBED_MODEL, input=f"{prefix}{text}"
    )
    return resp.data[0].embedding


@retriever(name="vector-store-retrieval", model_name="cosine-similarity")
async def retrieve_documents(
    query_embedding: list[float], top_k: int = 5
) -> list[dict]:
    """Retrieves top-k documents from vector store."""
    await asyncio.sleep(0.05)
    return [
        {
            "id": i,
            "content": f"Document {i} content about AI safety and alignment principles.",
            "score": 0.9 - i * 0.1,
        }
        for i in range(top_k)
    ]


@reranker(name="cross-encoder-reranker", model_name=RERANK_MODEL or "unknown")
async def rerank_documents(
    query: str, documents: list[dict], top_n: int = 3
) -> list[dict]:
    """Reranks documents using cross-encoder."""
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
    trace_url = get_trace_url(PHOENIX_BASE_URL)

    # Get the current trace ID for JSONL download
    from opentelemetry import trace as otel_trace

    current_span = otel_trace.get_current_span()
    trace_id_hex = None
    if current_span.is_recording():
        trace_id = current_span.get_span_context().trace_id
        trace_id_hex = format(trace_id, "032x")

    result_dict = {
        "query": query,
        "answer": answer,
        "sources": len(ranked_docs),
        "trace_url": trace_url,
    }
    if trace_id_hex:
        jsonl_download_url = get_spans_jsonl_download_url(
            PHOENIX_BASE_URL, "rag-pipeline-demo", span_ids=[trace_id_hex]
        )
        result_dict["jsonl_download_url"] = jsonl_download_url
    return result_dict


async def main():
    result = await rag_pipeline("What are the key principles of AI alignment?")
    print(f"\n✅ Sources used: {result['sources']}")
    if url := result.get("trace_url"):
        print(f"🔍 View complete trace: {url}")
    if jsonl_url := result.get("jsonl_download_url"):
        print(f"📥 Download spans as JSONL: {jsonl_url}")


if __name__ == "__main__":
    asyncio.run(main())
