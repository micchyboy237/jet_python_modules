# jet.adapters.deepeval.factory

from deepeval.models import DeepEvalBaseLLM
from jet.adapters.deepeval.llamacpp_model import LlamacppModel


def get_llm_client(**kwargs) -> DeepEvalBaseLLM:
    llm = LlamacppModel(**kwargs)
    return llm
