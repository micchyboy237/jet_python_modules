from typing import Dict, List, Type

from jet.llm.models import OLLAMA_LLM_MODELS
from jet.logger import logger
from llama_index.core.agent import ReActAgent
from jet.llm.ollama.base import Ollama


ALL_MODELS = OLLAMA_LLM_MODELS

AGENTS: Dict[str, Type[ReActAgent]] = {
    "react": ReActAgent,
}


def get_model(model: ALL_MODELS) -> Ollama:
    llm = Ollama(
        model=model,
        temperature=0.01,
        context_window=4096,
    )

    return llm


def is_valid_combination(agent: str, model: str) -> bool:
    if model not in ALL_MODELS.__args__:
        logger.error(f"{agent} does not work with {model}")
        return False
    return True
