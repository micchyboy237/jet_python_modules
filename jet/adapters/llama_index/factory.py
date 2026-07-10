import logging
import random
import time

import nest_asyncio
from jet.adapters.llama_cpp.config import (
    EMBED_BASE_URL,
    EMBED_MODEL,
    LLM_BASE_URL,
    LLM_MODEL,
)
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like.base import OpenAILike

nest_asyncio.apply()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_llama_cpp_llm() -> OpenAILike:
    """Reuse jet's llama_cpp config. OpenAILike skips llama_index's hardcoded
    OpenAI-model-name validation, so local/custom model names (e.g. Qwen3.5-4B) work."""
    logger.info(f"Building LLM client -> model={LLM_MODEL} base_url={LLM_BASE_URL}")
    return OpenAILike(
        model=LLM_MODEL,
        api_base=LLM_BASE_URL,
        api_key="not-needed",
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=8000,  # adjust to match your local model's actual context length
        timeout=120.0,
    )


def get_llama_cpp_embed_model() -> OpenAIEmbedding:
    """Reuse jet's llama_cpp config for embeddings, same pattern as factory.get_embedding_client()."""
    logger.info(
        f"Building embed model client -> model={EMBED_MODEL} base_url={EMBED_BASE_URL}"
    )
    return OpenAIEmbedding(
        model=EMBED_MODEL,
        api_base=EMBED_BASE_URL,
        timeout=30.0,
    )
