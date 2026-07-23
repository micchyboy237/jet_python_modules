from pathlib import Path
from typing import Any, Literal

from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_DIMS,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.adapters.smolagents.code_agent import CodeAgent
from jet.adapters.smolagents.mem0_adapter import Mem0AgentMemory
from jet.db.postgres.cleanup import drop_table_if_exists, drop_type_if_exists
from jet.db.postgres.config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from jet.libs.smolagents.custom_models import OpenAIModel
from smolagents import (
    Model,
    PromptTemplates,
    PythonExecutor,
    Tool,
)


def create_llm_model(
    temperature: float = 0.4,
    max_tokens: int | None = 10000,
    model_id: LLAMACPP_LLM_KEYS = LLM_MODEL,
    agent_name: str | None = None,
    logs_dir: str | Path | None = None,
    enable_thinking: bool = False,
    seed: int | None = 42,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
        logs_dir=logs_dir,
        enable_thinking=enable_thinking,
        seed=seed,
    )


def create_code_agent(
    tools: list[Tool],
    model: Model,
    prompt_templates: PromptTemplates | None = None,
    additional_authorized_imports: list[str] | None = None,
    planning_interval: int | None = None,
    executor: PythonExecutor = None,
    executor_type: Literal["local", "blaxel", "e2b", "modal", "docker"] = "local",
    executor_kwargs: dict[str, Any] | None = None,
    max_print_outputs_length: int | None = None,
    stream_outputs: bool = False,
    use_structured_outputs_internally: bool = False,
    code_block_tags: str | tuple[str, str] | None = ("<code>", "</code>"),
    **kwargs,
):
    agent_args = dict(
        tools=tools,
        model=model,
        prompt_templates=prompt_templates,
        additional_authorized_imports=additional_authorized_imports,
        planning_interval=planning_interval,
        executor=executor,
        executor_type=executor_type,
        executor_kwargs=executor_kwargs,
        max_print_outputs_length=max_print_outputs_length,
        stream_outputs=stream_outputs,
        use_structured_outputs_internally=use_structured_outputs_internally,
        code_block_tags=code_block_tags,
        **kwargs,
    )
    return CodeAgent(**agent_args)


# Add this new function
def create_mem0_agent_memory(
    mem0_config: dict[str, Any] | None = None,
    agent_id: str = "default_agent",
    auto_extract: bool = True,
    auto_store_steps: bool = True,
    reset: bool = True,
) -> Mem0AgentMemory:
    """Factory for creating mem0-backed agent memory."""
    # Default config for local llama.cpp
    collection_name = f"agent_{agent_id}_memories"

    if reset:
        drop_table_if_exists(f"public.{collection_name}_entities")
        drop_type_if_exists(f"public.{collection_name}_entities")

    default_config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": 12000,
                "openai_base_url": LLM_BASE_URL,
                "api_key": "dummy",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBED_MODEL,
                "embedding_dims": EMBED_DIMS,  # fallback 768 if model not in dict
                "openai_base_url": EMBED_BASE_URL,
                "api_key": "dummy",
            },
        },
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "collection_name": collection_name,
                "embedding_model_dims": EMBED_DIMS,
                "dbname": DEFAULT_DB,
                "user": DEFAULT_USER,
                "password": DEFAULT_PASSWORD,
                "host": DEFAULT_HOST,
                "port": DEFAULT_PORT,
            },
        },
    }

    # Merge with provided config (provided config takes precedence)
    final_config = {**default_config, **(mem0_config or {})}

    return Mem0AgentMemory(
        mem0_config=final_config,
        agent_id=agent_id,
        auto_extract=auto_extract,
        auto_store_steps=auto_store_steps,
    )
