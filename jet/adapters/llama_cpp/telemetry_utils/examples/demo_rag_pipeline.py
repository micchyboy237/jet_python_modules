import asyncio

from jet.adapters.llama_cpp import config
from jet_telemetry import chain, initialize_telemetry, llm, tool, trace

# 1. Initialize Telemetry using your existing config variable
# We pass the endpoint directly to ensure it matches your PHOENIX_BASE_URL
initialize_telemetry(
    service_name="llama-cpp-rag-demo",
    endpoint=config.PHOENIX_BASE_URL
)

# --- Mocked External Calls (Replace with actual httpx/requests calls) ---

@tool
async def llama_embed(text: str):
    """Simulates calling your EMBED_BASE_URL"""
    print(f"[Tool] Embedding text via {config.EMBED_BASE_HOST}...")
    await asyncio.sleep(0.2)
    return [0.1, 0.5, 0.9] # Mock vector

@tool
async def llama_rerank(query: str, docs: list[str]):
    """Simulates calling your RERANK_BASE_URL"""
    print(f"[Tool] Reranking via {config.RERANK_BASE_HOST}...")
    await asyncio.sleep(0.3)
    return docs # Mock reranked docs

@llm(model_name=config.LLM_MODEL)
async def llama_generate(prompt: str):
    """Simulates calling your LLM_BASE_URL"""
    print(f"[LLM] Generating with {config.LLM_MODEL} at {config.LLM_BASE_HOST}...")
    await asyncio.sleep(0.5)
    return "This is the generated answer from Qwen3.5."

# --- Orchestration ---

@chain
async def rag_retrieval_step(query: str):
    """Handles the vector search and reranking logic"""
    vectors = await llama_embed(query)
    # In a real app, you'd use vectors to fetch docs from a DB
    mock_docs = ["Doc A about Python", "Doc B about Jets"]
    ranked_docs = await llama_rerank(query, mock_docs)
    return "\n".join(ranked_docs)

@chain
async def run_rag_pipeline(question: str):
    """The main RAG chain"""
    context = await rag_retrieval_step(question)
    prompt = f"Context: {context}\nQuestion: {question}"
    answer = await llama_generate(prompt)
    return answer

# --- Entry Point ---

@trace(name="main_execution")
async def main():
    print("--- Starting RAG Demo ---")
    result = await run_rag_pipeline("How do I configure Llama.cpp?")
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())