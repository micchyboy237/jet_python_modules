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


async def run_manual_memory(question: str, tools: List) -> List[ChatMessage]:
    """
    Demonstrates manual memory management with put_messages and get.
    """
    memory = Memory.from_defaults(
        session_id="manual_session", token_limit=40000)
    messages = [
        ChatMessage(role="user", content="Hello, world!"),
        ChatMessage(role="assistant", content="Hello, world to you too!"),
    ]
    memory.put_messages(messages)
    chat_history = memory.get()
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, chat_history=chat_history)
    logger.info(f"Chat history: {chat_history}")
    return chat_history

if __name__ == "__main__":
    import asyncio
    tools = []
    question = "What's next?"
    asyncio.run(run_manual_memory(question, tools))
