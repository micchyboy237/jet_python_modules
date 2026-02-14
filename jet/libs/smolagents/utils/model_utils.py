# jet.libs.smolagents.utils.model_utils

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 8000,
    model_id: LLAMACPP_LLM_KEYS | None = None,
    agent_name: str | None = None,
    **kwargs,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    if model_id is None:
        model_id = "qwen3-instruct-2507:4b"
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
        **kwargs,
    )
