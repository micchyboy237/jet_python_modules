import os
import shutil
import logging
from typing import List
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory, StaticMemoryBlock, FactExtractionMemoryBlock, VectorMemoryBlock
from llama_index.core import Settings, VectorStoreIndex
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Setup logging
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(
    OUTPUT_DIR, "main.log"), level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup LLM and embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.llm = OllamaFunctionCallingAdapter(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)


async def run_custom_memory(question: str, tools: List, vector_store) -> str:
    """
    Demonstrates custom memory blocks with short-term and long-term memory.
    """
    blocks = [
        StaticMemoryBlock(
            name="core_info",
            static_content="My name is Logan, and I live in Saskatoon. I work at LlamaIndex.",
            priority=0,
        ),
        FactExtractionMemoryBlock(
            name="extracted_info",
            llm=Settings.llm,
            max_facts=50,
            priority=1,
        ),
        VectorMemoryBlock(
            name="vector_memory",
            vector_store=vector_store,
            priority=2,
            embed_model=Settings.embed_model,
        ),
    ]
    memory = Memory.from_defaults(
        session_id="custom_session",
        token_limit=40000,
        memory_blocks=blocks,
        insert_method="system",
    )
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, memory=memory)
    logger.info(f"Response: {response}")
    return str(response)

if __name__ == "__main__":
    import asyncio
    from jet.db.chroma.adapters.chroma_vector_store_llama_index_adapter import ChromaVectorStore
    import chromadb
    client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(
        chroma_collection=client.create_collection("test_collection"))
    tools = []
    question = "Who am I?"
    asyncio.run(run_custom_memory(question, tools, vector_store))
