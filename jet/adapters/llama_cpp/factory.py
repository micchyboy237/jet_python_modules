# jet.adapters.llama_cpp.factory


from jet.adapters.llama_cpp.config import EMBED_BASE_URL, LLM_BASE_URL
from openai import OpenAI


def get_llm_client() -> OpenAI:
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key="not-needed",
        timeout=120.0,
        max_retries=0,
    )


def get_embedding_client() -> OpenAI:
    return OpenAI(
        base_url=EMBED_BASE_URL,
        api_key="not-needed",
        timeout=30.0,
        max_retries=0,
    )
