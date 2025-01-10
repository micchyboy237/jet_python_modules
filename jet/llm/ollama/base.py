from collections import defaultdict
from typing import Callable, Optional, Sequence, Type, TypedDict, Any, Union
from jet.decorators.error import wrap_retry
from jet.decorators.function import retry_on_error
from jet.llm.ollama.constants import OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_SMALL_CHUNK_OVERLAP, OLLAMA_SMALL_CHUNK_SIZE, OLLAMA_SMALL_EMBED_MODEL
from jet.logger.timer import sleep_countdown
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEvent, CBEventType, EventPayload
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.types import PydanticProgramMode
from llama_index.llms.ollama import Ollama as BaseOllama
from llama_index.embeddings.ollama import OllamaEmbedding as BaseOllamaEmbedding
from llama_index.core import Settings
from llama_index.core.settings import _Settings

from jet.llm.ollama import (
    base_url,
    base_embed_url,
    large_embed_model,
    DEFAULT_LLM_SETTINGS,
    DEFAULT_EMBED_SETTINGS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from jet.logger import logger
from jet.utils.markdown import extract_json_block_content
from jet.validation.main.json_validation import validate_json
import json
from pydantic.main import BaseModel

dispatcher = get_dispatcher(__name__)


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


class _EnhancedSettings(_Settings):
    model: str
    embedding_model: str
    count_tokens: Callable[[str], int]

    def __setattr__(self, name, value):
        """Override setattr to synchronize with the Settings singleton."""
        super().__setattr__(name, value)
        # Synchronize with the Settings singleton if the attribute exists there
        if hasattr(Settings, name):
            setattr(Settings, name, value)


EnhancedSettings = _EnhancedSettings()


def initialize_ollama_settings(settings: SettingsDict = {}) -> _EnhancedSettings:
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

    chunk_size = settings.get("chunk_size")
    chunk_overlap = settings.get("chunk_overlap")

    if not chunk_size and not chunk_overlap:
        if embedding_model == OLLAMA_LARGE_EMBED_MODEL:
            chunk_size = OLLAMA_LARGE_CHUNK_SIZE
            chunk_overlap = OLLAMA_LARGE_CHUNK_OVERLAP
        elif embedding_model == OLLAMA_SMALL_EMBED_MODEL:
            chunk_size = OLLAMA_SMALL_CHUNK_SIZE
            chunk_overlap = OLLAMA_SMALL_CHUNK_OVERLAP

    def count_tokens(text: str) -> int:
        from jet.token import token_counter
        return token_counter(text, llm_model)

    EnhancedSettings.llm = llm
    EnhancedSettings.embed_model = embed_model
    EnhancedSettings.chunk_size = chunk_size
    EnhancedSettings.chunk_overlap = chunk_overlap
    EnhancedSettings.model = llm_model
    EnhancedSettings.embedding_model = embedding_model
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
    base_url: str = DEFAULT_LLM_SETTINGS['base_url'],
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
    base_url: str = DEFAULT_EMBED_SETTINGS['base_url'],
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
    max_tokens: int | float = 0.4

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from jet.token import filter_texts

        logger.info("Calling Ollama chat...")

        if self.max_tokens:
            messages = filter_texts(
                messages, self.model, max_tokens=self.max_tokens)

        def run():
            from jet.llm import call_ollama_chat

            ollama_messages = self._convert_to_ollama_messages(messages)

            tools = kwargs.pop("tools", None)
            format = kwargs.pop("format", "json" if self.json_mode else None)
            stream = not tools

            response = call_ollama_chat(
                model=self.model,
                messages=ollama_messages,
                stream=stream,
                format=format,
                tools=tools,
                options=self._model_kwargs,
                keep_alive=self.keep_alive,
                full_stream_response=True,
            )

            final_response = {}

            if not stream:
                content = response["message"]["content"]
                role = response["message"]["role"]
                tool_calls = response["message"].get("tool_calls", [])
                token_counts = self._get_response_token_counts(response)
                final_response = {
                    **response.copy(),
                    "usage": token_counts,
                }

            else:
                content = ""
                role = ""
                tool_calls = []
                for chunk in response:
                    content += chunk["message"]["content"]
                    if not role:
                        role = chunk["message"]["role"]
                    if chunk["done"]:
                        token_counts = self._get_response_token_counts(
                            response)
                        final_response = {
                            **chunk.copy(),
                            "usage": token_counts,
                        }

            return ChatResponse(
                message=ChatMessage(
                    content=final_response["message"]["content"],
                    role=final_response["message"]["role"],
                    additional_kwargs={"tool_calls": tool_calls},
                ),
                raw=final_response,
            )

        return wrap_retry(run)

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from jet.token import filter_texts

        logger.info("Calling Ollama achat...")

        if self.max_tokens:
            messages = filter_texts(
                messages, self.model, max_tokens=self.max_tokens)

        def run():
            from jet.llm import call_ollama_chat
            ollama_messages = self._convert_to_ollama_messages(messages)

            tools = kwargs.pop("tools", None)
            format = kwargs.pop("format", "json" if self.json_mode else None)
            stream = not tools

            response = call_ollama_chat(
                model=self.model,
                messages=ollama_messages,
                stream=stream,
                format=format,
                tools=tools,
                options=self._model_kwargs,
                keep_alive=self.keep_alive,
                full_stream_response=True,
            )

            final_response = {}

            if not stream:
                content = response["message"]["content"]
                role = response["message"]["role"]
                tool_calls = response["message"].get("tool_calls", [])
                token_counts = self._get_response_token_counts(response)
                final_response = {
                    **response.copy(),
                    "usage": token_counts,
                }

            else:
                content = ""
                role = ""
                tool_calls = []
                for chunk in response:
                    content += chunk["message"]["content"]
                    if not role:
                        role = chunk["message"]["role"]
                    if chunk["done"]:
                        token_counts = self._get_response_token_counts(
                            response)
                        final_response = {
                            **chunk.copy(),
                            "usage": token_counts,
                        }

            return ChatResponse(
                message=ChatMessage(
                    content=final_response["message"]["content"],
                    role=final_response["message"]["role"],
                    additional_kwargs={"tool_calls": tool_calls},
                ),
                raw=final_response,
            )

        return wrap_retry(run)

    @dispatcher.span
    def structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            llm_kwargs = llm_kwargs or {}
            llm_kwargs["format"] = output_cls.model_json_schema()

            messages = prompt.format_messages(**prompt_args)
            response = self.chat(messages, **llm_kwargs)

            extracted_result = extract_json_block_content(
                response.message.content or "")
            validation_result = validate_json(
                extracted_result, output_cls.model_json_schema())

            return output_cls.model_validate_json(json.dumps(validation_result["data"]))
        else:
            return super().structured_predict(
                output_cls, prompt, llm_kwargs, **prompt_args
            )


class OllamaEmbedding(BaseOllamaEmbedding):
    def get_general_text_embedding(self, texts: Union[str, Sequence[str]] = '',) -> list[float]:
        """Get Ollama embedding with retry mechanism."""
        logger.info("Calling OllamaEmbedding embed...")

        def run():
            with self.callback_manager.event(
                CBEventType.EMBEDDING,
                payload={EventPayload.SERIALIZED: self.to_dict()},
            ) as event:
                result = self._client.embed(
                    model=self.model_name, input=texts, options=self.ollama_additional_kwargs
                )
                embeddings = result["embeddings"][0]
                event.on_end(
                    payload={
                        EventPayload.CHUNKS: [texts] if isinstance(texts, str) else texts,
                        EventPayload.EMBEDDINGS: [embeddings],
                    },
                )

            logger.log("Batch Tokens:", len(embeddings),
                       colors=["DEBUG", "SUCCESS"])
            return embeddings

        return wrap_retry(run)


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
