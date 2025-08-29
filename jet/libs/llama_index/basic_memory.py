import os
import shutil
import logging
from typing import List
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DEFAULT_ASYNC_DB_URI = "postgresql+asyncpg://jethroestrada@localhost:5432/async_db1"

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


async def run_basic_memory(question: str, tools: List) -> str:
    """
    Demonstrates basic memory configuration with default settings.
    """
    memory = Memory.from_defaults(
        session_id="basic_session", token_limit=40000, async_database_uri=DEFAULT_ASYNC_DB_URI)
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, memory=memory)
    logger.info(f"Response: {response}")
    return str(response)

if __name__ == "__main__":
    import asyncio
    # Example tools (replace with actual tools as needed)
    tools = []
    question = "What's the weather like today?"
    asyncio.run(run_basic_memory(question, tools))
