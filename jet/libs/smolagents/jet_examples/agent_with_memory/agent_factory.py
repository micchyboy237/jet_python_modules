from callbacks import auto_extract_simple_facts, auto_save_shared_state
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from smolagents import CodeAgent
from tools.memory_tools import (
    LongTermRecallTool,
    LongTermSaveTool,
    SharedStateReadTool,
    SharedStateUpdateTool,
)


def create_local_model(
    temperature: float = 0.7,
    max_tokens: int | None = 8192,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """
    Factory for creating a consistently configured local OpenAI-compatible model
    (llama.cpp server, vLLM, Ollama with OpenAI compat, etc.).
    """
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


def create_memory_enabled_agent(
    model=None, extra_tools=None, max_steps: int = 40, verbosity: int = 1
) -> CodeAgent:
    model = create_local_model()
    tools = [
        LongTermSaveTool(),
        LongTermRecallTool(),
        SharedStateUpdateTool(),
        SharedStateReadTool(),
    ]
    if extra_tools:
        tools.extend(extra_tools)

    return CodeAgent(
        tools=tools,
        model=model,
        step_callbacks=[
            auto_save_shared_state,
            auto_extract_simple_facts,
            # add more callbacks here
        ],
        max_steps=max_steps,
        verbosity_level=verbosity,
    )
