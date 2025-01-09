from collections import defaultdict
from typing import Callable, Optional, Sequence, TypedDict, Any, Union
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEvent, CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.llms.ollama import Ollama as BaseOllama
from llama_index.embeddings.ollama import OllamaEmbedding as BaseOllamaEmbedding
from llama_index.core import Settings

from jet.llm.ollama import (
    base_url,
    base_embed_url,
    large_embed_model,
    DEFAULT_LLM_SETTINGS,
    DEFAULT_EMBED_SETTINGS,
    DEFAULT_EMBED_BATCH_SIZE,
)
from jet.llm.ollama.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from jet.logger import logger


# class StreamCallbackManager(CallbackManager):
#     def on_event_start(
#         self,
#         event_type: CBEventType,
#         payload: Optional[dict[str, any]] = None,
#         event_id: str = "",
#         parent_id: str = "",
#         **kwargs: any,
#     ):
#         logger.log("StreamCallbackManager on_event_start:", {
#             "event_type": event_type,
#             "payload": payload,
#             "event_id": event_id,
#             "parent_id": parent_id,
#             **kwargs
#         })

#     def on_event_end(
#         self,
#         event_type: CBEventType,
#         payload: Optional[dict[str, any]] = None,
#         event_id: str = "",
#         **kwargs: any,
#     ):
#         logger.log("StreamCallbackManager on_event_end:", {
#             "event_type": event_type,
#             "payload": str(payload)[:50],
#             "event_id": event_id,
#             **kwargs
#         })


# Settings.callback_manager = StreamCallbackManager()


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
    embed_model: object
    llm: object
    chunk_size: int
    chunk_overlap: int


def initialize_ollama_settings(settings: SettingsDict = {}) -> EnhancedSettings:
    embedding_model = settings.get(
        "embedding_model", DEFAULT_EMBED_SETTINGS['model_name'])
    embed_model = OllamaEmbedding(
        model_name=DEFAULT_EMBED_SETTINGS['model_name'],
        base_url=settings.get("base_url", DEFAULT_EMBED_SETTINGS['base_url']),
        embed_batch_size=DEFAULT_EMBED_SETTINGS['embed_batch_size'],
        ollama_additional_kwargs=DEFAULT_EMBED_SETTINGS['ollama_additional_kwargs'],
    )

    llm_model = settings.get("llm_model", DEFAULT_LLM_SETTINGS['model'])
    llm = Ollama(
        model=llm_model,
        base_url=settings.get("base_url", DEFAULT_LLM_SETTINGS['base_url']),
        temperature=settings.get(
            "temperature", DEFAULT_LLM_SETTINGS['temperature']),
        context_window=settings.get(
            "context_window", DEFAULT_LLM_SETTINGS['context_window']),
        request_timeout=settings.get(
            "request_timeout", DEFAULT_LLM_SETTINGS['request_timeout']),
    )

    chunk_size = settings.get(
        "chunk_size", DEFAULT_LLM_SETTINGS.get('chunk_size'))
    chunk_overlap = settings.get(
        "chunk_overlap", DEFAULT_LLM_SETTINGS.get('chunk_overlap'))

    enhanced_settings = EnhancedSettings()
    enhanced_settings.model = llm_model
    enhanced_settings.embedding_model = embedding_model
    enhanced_settings.embed_model = embed_model
    enhanced_settings.llm = llm
    enhanced_settings.chunk_size = DEFAULT_CHUNK_SIZE
    enhanced_settings.chunk_overlap = DEFAULT_CHUNK_OVERLAP

    def count_tokens(text: str) -> int:
        from jet.token import token_counter
        return token_counter(text, llm_model)

    enhanced_settings.count_tokens = count_tokens

    return enhanced_settings


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
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    result = self._client.embed(
                        model=self.model_name, input=texts, options=self.ollama_additional_kwargs
                    )
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: [texts] if isinstance(texts, str) else texts,
                            EventPayload.EMBEDDINGS: [result["embeddings"][0]],
                        },
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


class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
    ) -> None:
        """Initialize the Stream callback handler."""
        super().__init__(
            event_starts_to_ignore=[],
            event_ends_to_ignore=[],
        )
        self._event_pairs_by_id: dict[str, list[CBEvent]] = defaultdict(list)
        self._trace_map: dict[str, list[str]] = defaultdict(list)

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: any,
    ):
        logger.log("StreamCallbackHandler on_event_start:", {
            "event_type": event_type,
            "payload": payload,
            "event_id": event_id,
            "parent_id": parent_id,
            **kwargs
        })

        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[dict[str, any]] = None,
        event_id: str = "",
        **kwargs: any,
    ):
        logger.log("StreamCallbackHandler on_event_end:", {
            "event_type": event_type,
            "payload": str(payload)[:50],
            "event_id": event_id,
            **kwargs
        })

        event = CBEvent(event_type, payload=payload, id_=event_id)
        self._event_pairs_by_id[event.id_].append(event)
        self._trace_map = defaultdict(list)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        self._trace_map = defaultdict(list)
        return super().start_trace(trace_id)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self._trace_map = trace_map or defaultdict(list)
        return super().end_trace(trace_id, trace_map)

    def build_trace_map(
        self,
        cur_event_id: str,
        trace_map: Any,
    ) -> dict[str, Any]:
        event_pair = self._event_pairs_by_id[cur_event_id]
        if event_pair:
            event_data = {
                "event_type": event_pair[0].event_type,
                "event_id": event_pair[0].id_,
                "children": {},
            }
            trace_map[cur_event_id] = event_data

        child_event_ids = self._trace_map[cur_event_id]
        for child_event_id in child_event_ids:
            self.build_trace_map(child_event_id, event_data["children"])
        return trace_map
