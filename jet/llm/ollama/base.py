from typing import Callable, Optional, Sequence, TypedDict, Any, Union
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama as BaseOllama
from llama_index.embeddings.ollama import OllamaEmbedding as BaseOllamaEmbedding
from llama_index.core import Settings

from jet.llm.ollama import (
    base_url,
    large_embed_model,
    DEFAULT_LLM_SETTINGS,
    DEFAULT_EMBED_SETTINGS,
)
from jet.logger import logger


class SettingsDict(TypedDict, total=False):
    llm_model: str
    context_window: int
    request_timeout: float
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    base_url: str
    temperature: float


class EnhancedSettings(Settings):
    model: str
    embedding_model: str
    count_tokens: Callable[[str], int]


def initialize_ollama_settings(settings: SettingsDict = {}):
    embedding_model = settings.get("embedding_model",
                                   DEFAULT_EMBED_SETTINGS['model_name'])
    Settings.embed_model = OllamaEmbedding(
        model_name=DEFAULT_EMBED_SETTINGS['model_name'],
        base_url=base_url,
        embed_batch_size=DEFAULT_EMBED_SETTINGS['embed_batch_size'],
        ollama_additional_kwargs=DEFAULT_EMBED_SETTINGS['ollama_additional_kwargs'],
    )

    llm_model = settings.get("llm_model", DEFAULT_LLM_SETTINGS['model'])
    Settings.llm = create_llm(
        model=llm_model,
        base_url=settings.get("base_url", DEFAULT_LLM_SETTINGS['base_url']),
        temperature=settings.get(
            "temperature", DEFAULT_LLM_SETTINGS['temperature']),
        context_window=settings.get(
            "context_window", DEFAULT_LLM_SETTINGS['context_window']),
        request_timeout=settings.get(
            "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
    )
    Settings.llm = Ollama(
        model=llm_model,
        base_url=settings.get("base_url", DEFAULT_LLM_SETTINGS['base_url']),
        temperature=settings.get(
            "temperature", DEFAULT_LLM_SETTINGS['temperature']),
        context_window=settings.get(
            "context_window", DEFAULT_LLM_SETTINGS['context_window']),
        request_timeout=settings.get(
            "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
    )

    if settings.get("chunk_size"):
        Settings.chunk_size = settings["chunk_size"]

    if settings.get("chunk_overlap"):
        Settings.chunk_overlap = settings["chunk_overlap"]

    EnhancedSettings.model = llm_model
    EnhancedSettings.embedding_model = embedding_model

    def count_tokens(text: str) -> int:
        from jet.token import token_counter
        return token_counter(text, llm_model)
    EnhancedSettings.count_tokens = count_tokens

    return EnhancedSettings


def update_llm_settings(settings: SettingsDict = {}):
    if settings.get("chunk_size"):
        Settings.chunk_size = settings["chunk_size"]

    if settings.get("chunk_overlap"):
        Settings.chunk_overlap = settings["chunk_overlap"]

    if settings.get("embedding_model"):
        Settings.embed_model = create_embed_model(
            model=settings.get("embedding_model",
                               DEFAULT_EMBED_SETTINGS['model_name']),
            base_url=settings.get(
                "base_url", DEFAULT_EMBED_SETTINGS['base_url']),
        )

    if settings.get("llm_model"):
        Settings.llm = create_llm(
            model=settings.get("llm_model", DEFAULT_LLM_SETTINGS['model']),
            base_url=settings.get(
                "base_url", DEFAULT_LLM_SETTINGS['base_url']),
            temperature=settings.get(
                "temperature", DEFAULT_LLM_SETTINGS['temperature']),
            context_window=settings.get(
                "context_window", DEFAULT_LLM_SETTINGS['context_window']),
            request_timeout=settings.get(
                "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
        )

    return Settings


def create_llm(
    model: str = DEFAULT_LLM_SETTINGS['model'],
    base_url: str = base_url,
    temperature: float = DEFAULT_LLM_SETTINGS['temperature'],
    context_window: int = DEFAULT_LLM_SETTINGS['context_window'],
    request_timeout: float = DEFAULT_LLM_SETTINGS['request_timeout'],
    max_tokens: Optional[int] = None
) -> LLM:
    llm = Ollama(
        temperature=temperature,
        context_window=context_window,
        request_timeout=request_timeout,
        model=model,
        base_url=base_url,
        max_tokens=max_tokens,
    )
    Settings.llm = llm
    return llm


def create_embed_model(
    model: str = DEFAULT_EMBED_SETTINGS['model_name'],
    base_url: str = base_url,
    embed_batch_size: int = DEFAULT_EMBED_SETTINGS['embed_batch_size'],
    ollama_additional_kwargs: dict[str,
                                   any] = DEFAULT_EMBED_SETTINGS['ollama_additional_kwargs'],
):
    embed_model = OllamaEmbedding(
        model_name=model,
        base_url=base_url,
        embed_batch_size=embed_batch_size,
        ollama_additional_kwargs=ollama_additional_kwargs,
    )
    Settings.embed_model = embed_model
    return embed_model


class Ollama(BaseOllama):
    def chat(self, messages: Sequence[ChatMessage], max_tokens: Optional[int] = None, **kwargs: Any) -> ChatResponse:
        from jet.token import filter_texts

        logger.info("Calling Ollama chat...")

        if max_tokens:
            messages = filter_texts(
                messages, self.model, max_tokens=max_tokens)

        return super().chat(messages, **kwargs)


class OllamaEmbedding(BaseOllamaEmbedding):
    def get_general_text_embedding(self, texts: Union[str, Sequence[str]] = '',) -> list[float]:
        """Get Ollama embedding with retry mechanism."""
        import time

        logger.info("Calling OllamaEmbedding embed...")

        max_retries = 5
        delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                result = self._client.embed(
                    model=self.model_name, input=texts, options=self.ollama_additional_kwargs
                )
                return result["embeddings"][0]
            except Exception as e:
                if attempt < max_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Raising the exception.")
                    raise e
