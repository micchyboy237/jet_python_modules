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
    base_url: str = "http://localhost:8000/v1",  # typical llama.cpp server
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
        base_url=base_url,
    )


def create_local_qwen_agent(
    temperature: float = 0.7,
    max_tokens: int | None = 8192,
    max_steps: int = 40,
    verbosity: int = 1,
    extra_tools=None,
    agent_name: str | None = "Qwen-Agent",
) -> CodeAgent:
    model = create_local_model(
        temperature=temperature,
        max_tokens=max_tokens,
        model_id="qwen3-instruct-2507:4b",
        agent_name=agent_name,
    )
    return create_memory_enabled_agent(
        model=model,
        extra_tools=extra_tools,
        max_steps=max_steps,
        verbosity=verbosity,
    )


def create_memory_enabled_agent(
    model=None, extra_tools=None, max_steps: int = 40, verbosity: int = 1
) -> CodeAgent:
    if model is None:  # default remote HF inference
        model = create_local_qwen_agent()
    tools = [
        LongTermSaveTool,
        LongTermRecallTool,
        SharedStateUpdateTool,
        SharedStateReadTool,
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
