# jet.adapters.deepeval.factory

from deepeval.models import DeepEvalBaseLLM
from jet.adapters.deepeval.client import CustomOpenAIClient


def get_chat_openai(**kwargs) -> DeepEvalBaseLLM:
    llm = CustomOpenAIClient(**kwargs)
    return llm
