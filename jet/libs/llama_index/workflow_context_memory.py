import os
import shutil
import logging
from typing import List
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.workflow import Context
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


async def run_workflow_context_memory(question: str, tools: List, workflow) -> str:
    """
    Demonstrates combining memory with workflow context for human-in-the-loop.
    """
    memory = Memory.from_defaults(
        session_id="workflow_session", token_limit=40000)
    ctx = Context(workflow)
    agent = FunctionAgent(llm=Settings.llm, tools=tools)
    response = await agent.run(question, ctx=ctx, memory=memory)
    ctx_dict = ctx.to_dict()
    logger.info(f"Context serialized: {ctx_dict}")
    ctx_restored = Context.from_dict(workflow, ctx_dict)
    memory_restored = await ctx_restored.store.get("memory")
    logger.info(f"Restored memory: {memory_restored.get()}")
    return str(response)

if __name__ == "__main__":
    import asyncio
    from llama_index.core.workflow import Workflow

    class SimpleWorkflow(Workflow):
        pass
    workflow = SimpleWorkflow()
    tools = []
    question = "Resume the workflow."
    asyncio.run(run_workflow_context_memory(question, tools, workflow))
