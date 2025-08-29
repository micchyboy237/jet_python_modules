import os
import shutil
import logging
from typing import List
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core import Settings
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


async def run_remote_memory(question: str, tools: List) -> str:
    """
    Demonstrates remote memory with PostgreSQL database.
    """
    memory = Memory.from_defaults(
        session_id="remote_session",
        token_limit=40000,
        async_database_uri="postgresql+asyncpg://postgres:mark90@localhost:5432/postgres",
    )
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, memory=memory)
    logger.info(f"Response: {response}")
    return str(response)

if __name__ == "__main__":
    import asyncio
    tools = []
    question = "What's stored in memory?"
    asyncio.run(run_remote_memory(question, tools))
