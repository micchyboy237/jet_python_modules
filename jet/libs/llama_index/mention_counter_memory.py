import os
import shutil
import logging
from typing import Optional, List, Any
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory, BaseMemoryBlock
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
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
Settings.llm = OllamaFunctionCalling(model="llama3.2")
Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)


class MentionCounter(BaseMemoryBlock[str]):
    """A memory block that counts the number of times a user mentions a specific name."""
    mention_name: str = "Logan"
    mention_count: int = 0

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        return f"Logan was mentioned {self.mention_count} times."

    async def _aput(self, messages: List[ChatMessage]) -> None:
        for message in messages:
            if self.mention_name in message.content:
                self.mention_count += 1

    async def atruncate(self, content: str, tokens_to_truncate: int) -> Optional[str]:
        return ""


async def run_mention_counter(question: str, tools: List) -> str:
    """
    Demonstrates a custom MentionCounter memory block.
    """
    memory = Memory.from_defaults(
        session_id="mention_session",
        token_limit=40000,
        memory_blocks=[MentionCounter(name="mention_counter", priority=0)],
    )
    messages = [ChatMessage(role="user", content="Logan is great!")]
    memory.put_messages(messages)
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, memory=memory)
    logger.info(f"Response: {response}")
    return str(response)

if __name__ == "__main__":
    import asyncio
    tools = []
    question = "How many times was Logan mentioned?"
    asyncio.run(run_mention_counter(question, tools))
