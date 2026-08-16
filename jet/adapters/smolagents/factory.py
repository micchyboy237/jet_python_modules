from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Type

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
from smolagents.local_python_executor import PythonExecutor
from smolagents.memory import MemoryStep
from smolagents.models import Model
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.tools import Tool


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
    # MultiStepAgent parameters
    instructions: str | None = None,
    max_steps: int = 20,
    add_base_tools: bool = False,
    verbosity_level: LogLevel = LogLevel.INFO,
    managed_agents: list | None = None,
    step_callbacks: list[Callable]
    | dict[Type[MemoryStep], Callable | list[Callable]]
    | None = None,
    name: str | None = None,
    description: str | None = None,
    provide_run_summary: bool = False,
    final_answer_checks: list[Callable] | None = None,
    return_full_result: bool = False,
    logger: AgentLogger | None = None,
    **kwargs,
) -> CodeAgent:
    """
    Create a CodeAgent instance with the specified configuration.

    Args:
        tools: List of Tool objects the agent can use.
        model: Model that will generate the agent's actions.
        prompt_templates: Optional custom prompt templates.
        additional_authorized_imports: Additional authorized imports for the agent.
        planning_interval: Interval at which the agent will run a planning step.
        executor: Custom Python code executor. If not provided, a default executor will be created.
        executor_type: Type of code executor to use.
        executor_kwargs: Additional arguments to pass to initialize the executor.
        max_print_outputs_length: Maximum length of the print outputs.
        stream_outputs: Whether to stream outputs during execution.
        use_structured_outputs_internally: Whether to use structured generation at each action step.
        code_block_tags: Opening and closing tags for code blocks.

        # MultiStepAgent parameters
        instructions: Custom instructions for the agent, inserted in the system prompt.
        max_steps: Maximum number of steps the agent can take to solve the task.
        add_base_tools: Whether to add the base tools to the agent's tools.
        verbosity_level: Level of verbosity of the agent's logs.
        managed_agents: Managed agents that the agent can call.
        step_callbacks: Callbacks that will be called at each step.
        name: Name by which this agent can be called (for managed agents).
        description: Description of this agent (for managed agents).
        provide_run_summary: Whether to provide a run summary when called as a managed agent.
        final_answer_checks: List of validation functions to run before accepting a final answer.
        return_full_result: Whether to return the full RunResult or just the final answer.
        logger: Custom AgentLogger instance.
        **kwargs: Additional keyword arguments to pass to CodeAgent.

    Returns:
        CodeAgent: A configured CodeAgent instance.
    """
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
        # MultiStepAgent parameters
        instructions=instructions,
        max_steps=max_steps,
        add_base_tools=add_base_tools,
        verbosity_level=verbosity_level,
        managed_agents=managed_agents,
        step_callbacks=step_callbacks,
        name=name,
        description=description,
        provide_run_summary=provide_run_summary,
        final_answer_checks=final_answer_checks,
        return_full_result=return_full_result,
        logger=logger,
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
