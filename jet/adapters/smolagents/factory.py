from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel


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
