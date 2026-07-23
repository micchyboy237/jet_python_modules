from pathlib import Path
from typing import Any, Literal

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.adapters.smolagents.code_agent import CodeAgent
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
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
        logs_dir=logs_dir,
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
