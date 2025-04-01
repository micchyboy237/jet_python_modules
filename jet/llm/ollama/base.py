from jet.data.utils import generate_unique_hash
from jet.llm.utils.embeddings import get_embedding_function
from jet.token.token_utils import get_model_max_tokens, tokenize
from llama_index.core import VectorStoreIndex as BaseVectorStoreIndex
from collections import defaultdict
from typing import Callable, Optional, Sequence, Type, TypedDict, Any, Union
from jet.decorators.error import wrap_retry
from jet.decorators.function import retry_on_error
from jet.llm.ollama.constants import DEFAULT_BASE_URL, DEFAULT_CONTEXT_WINDOW, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_REQUEST_TIMEOUT, OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_SMALL_CHUNK_OVERLAP, OLLAMA_SMALL_CHUNK_SIZE, OLLAMA_SMALL_EMBED_MODEL
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_CONTEXTS, OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from jet.logger.timer import sleep_countdown, time_it
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEvent, CBEventType, EventPayload
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms.callbacks import llm_chat_callback
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.types import PydanticProgramMode
from llama_index.core.utils import set_global_tokenizer
from llama_index.llms.ollama import Ollama as BaseOllama
from llama_index.embeddings.ollama import OllamaEmbedding as BaseOllamaEmbedding
from llama_index.core import Settings
from llama_index.core.settings import _Settings

from jet.llm.ollama.config import (
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
from pydantic.fields import Field
from pydantic.main import BaseModel
from transformers.tokenization_utils_base import EncodedInput, PreTokenizedInput, TextInput

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
    llm_model: OLLAMA_MODEL_NAMES
    context_window: int
    request_timeout: float
    embedding_model: OLLAMA_EMBED_MODELS
    chunk_size: int
    chunk_overlap: int
    base_url: str
    temperature: float


class _EnhancedSettings(_Settings):
    model: OLLAMA_MODEL_NAMES
    embedding_model: OLLAMA_EMBED_MODELS
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
            "context_window", get_model_max_tokens(llm_model)),
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

    from jet.token.token_utils import get_ollama_tokenizer
    tokenizer = get_ollama_tokenizer(llm_model)
    set_global_tokenizer(tokenizer)
    # EnhancedSettings.tokenizer = get_ollama_tokenizer(llm_model).encode

    from jet.helpers.prompt.custom_prompt_helpers import OllamaPromptHelper
    EnhancedSettings.prompt_helper = OllamaPromptHelper(llm_model)

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
    model: OLLAMA_MODEL_NAMES = DEFAULT_LLM_SETTINGS['model'],
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
    model: OLLAMA_EMBED_MODELS = DEFAULT_EMBED_SETTINGS['model_name'],
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
    model: OLLAMA_MODEL_NAMES = "llama3.1"
    max_tokens: Optional[Union[int, float]] = None
    max_prediction_ratio: Optional[float] = None

    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model=model, **kwargs)

        # Initialize and set tokenizer
        from jet.token.token_utils import get_ollama_tokenizer
        tokenizer = get_ollama_tokenizer(self.model)
        set_global_tokenizer(tokenizer)

    def encode(self, texts: Union[str, Sequence[str]] = ''):
        """Calls get_general_text_embedding to get the embeddings."""
        tokens = tokenize(self.model, texts)
        return tokens

    # @llm_chat_callback()
    def chat(self, messages: str | Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from jet.actions.generation import call_ollama_chat
        from jet.token.token_utils import token_counter

        ollama_messages = self._convert_to_ollama_messages(
            messages) if not isinstance(messages, str) else messages

        tools = kwargs.get("tools", None)
        format = kwargs.get("format", "json" if self.json_mode else None)
        options = kwargs.get("options", {})
        stream = kwargs.get("stream", not tools)

        settings = {
            **kwargs,
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream,
            "format": format,
            "tools": tools,
            "keep_alive": self.keep_alive,
            "full_stream_response": True,
            "options": {
                **self._model_kwargs,
                **options,
            },
        }

        def run():
            response = call_ollama_chat(**settings)

            final_response = {}

            if not stream:
                content = response["message"]["content"]
                role = response["message"]["role"]
                tool_calls = response["message"].get("tool_calls", [])

                final_response_content = content
                final_response_tool_calls = tool_calls
                if final_response_tool_calls:
                    final_response_content += f"\n{final_response_tool_calls}".strip()

                prompt_token_count = token_counter(ollama_messages, self.model)
                response_token_count = token_counter(
                    final_response_content, self.model)

                final_response = {
                    **response.copy(),
                    "usage": {
                        "prompt_tokens": prompt_token_count,
                        "completion_tokens": response_token_count,
                        "total_tokens": prompt_token_count + response_token_count,
                    }
                }

            else:
                content = ""
                role = ""
                tool_calls = []

                if isinstance(response, dict) and "error" in response:
                    raise ValueError(
                        f"Ollama API error:\n{response['error']}")

                for chunk in response:

                    content += chunk["message"]["content"]
                    if not role:
                        role = chunk["message"]["role"]
                    if chunk["done"]:
                        prompt_token_count: int = token_counter(
                            ollama_messages, self.model)
                        response_token_count: int = token_counter(
                            content, self.model)

                        updated_chunk = chunk.copy()
                        updated_chunk["message"]["content"] = content

                        final_response = {
                            **updated_chunk,
                            "usage": {
                                "prompt_tokens": prompt_token_count,
                                "completion_tokens": response_token_count,
                                "total_tokens": prompt_token_count + response_token_count,
                            }
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

    # @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        from jet.actions.generation import call_ollama_chat
        from jet.token.token_utils import token_counter

        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.get("tools", None)
        format = kwargs.get("format", "json" if self.json_mode else None)
        options = kwargs.get("options", {})
        stream = kwargs.get("stream", not tools)

        settings = {
            **kwargs,
            "model": self.model,
            "messages": ollama_messages,
            "stream": stream,
            "format": format,
            "tools": tools,
            "keep_alive": self.keep_alive,
            "full_stream_response": True,
            "options": {
                **self._model_kwargs,
                **options,
            },
        }

        def run():
            response = call_ollama_chat(**settings)

            final_response = {}

            if not stream:
                content = response["message"]["content"]
                role = response["message"]["role"]
                tool_calls = response["message"].get("tool_calls", [])

                final_response_content = content
                final_response_tool_calls = tool_calls
                if final_response_tool_calls:
                    final_response_content += f"\n{final_response_tool_calls}".strip()

                prompt_token_count: int = token_counter(
                    ollama_messages, self.model)
                response_token_count: int = token_counter(
                    final_response_content, self.model)

                final_response = {
                    **response.copy(),
                    "usage": {
                        "prompt_tokens": prompt_token_count,
                        "completion_tokens": response_token_count,
                        "total_tokens": prompt_token_count + response_token_count,
                    }
                }

            else:
                content = ""
                role = ""
                tool_calls = []

                if isinstance(response, dict) and "error" in response:
                    raise ValueError(
                        f"Ollama API error:\n{response['error']}")

                for chunk in response:

                    content += chunk["message"]["content"]
                    if not role:
                        role = chunk["message"]["role"]
                    if chunk["done"]:
                        prompt_token_count = token_counter(
                            ollama_messages, self.model)
                        response_token_count = token_counter(
                            content, self.model)

                        final_response = {
                            **chunk.copy(),
                            "usage": {
                                "prompt_tokens": prompt_token_count,
                                "completion_tokens": response_token_count,
                                "total_tokens": prompt_token_count + response_token_count,
                            }
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
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted by Ollama",
    )
    model_name: OLLAMA_EMBED_MODELS = Field(
        default="mxbai-embed-large",
        description="The Ollama model to use.",
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )

    def encode(self, texts: Union[str, Sequence[str]] = ''):
        """Calls get_general_text_embedding to get the embeddings."""
        tokens = tokenize(self.model_name, texts)
        return tokens

    def get_general_text_embedding(self, texts: Union[str, Sequence[str]] = '',) -> list[float] | list[list[float]]:
        """Get Ollama embedding with retry mechanism."""
        logger.orange("Calling OllamaEmbedding embed...")
        logger.debug(
            "Embed model:",
            self.model_name,
            f"({OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]})",
            colors=["GRAY", "DEBUG", "DEBUG"],
        )
        logger.debug(f"Max Context: {OLLAMA_MODEL_CONTEXTS[self.model_name]}")
        logger.debug(
            f"Embeddings Dim: {OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]}")

        def run():
            with self.callback_manager.event(
                CBEventType.EMBEDDING,
                payload={EventPayload.SERIALIZED: self.to_dict()},
            ) as event:
                embed_func = get_embedding_function(
                    model_name=self.model_name
                )
                embeddings = embed_func(texts)

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


def embed_nodes(
    nodes: Sequence[BaseNode] | Sequence[str], embed_model: OLLAMA_EMBED_MODELS | str, show_progress: bool = False
) -> dict[str, list[float]]:
    """Get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode] | Sequence[str]): The nodes or texts to embed.
        embed_model (OLLAMA_EMBED_MODELS): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        dict[str, list[float]]: A map from node id to embedding.
    """
    id_to_embed_map: dict[str, list[float]] = {}

    texts_to_embed: list[str] = []
    ids_to_embed: list[str] = []

    if isinstance(nodes[0], BaseNode):
        texts_to_embed = [node.text for node in nodes]
        ids_to_embed = [node.node_id for node in nodes]
    else:
        texts_to_embed = nodes
        ids_to_embed = [generate_unique_hash(text) for text in nodes]

    embedding_function = get_embedding_function(
        model_name=embed_model,
    )
    new_embeddings = embedding_function(texts_to_embed)

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map


async def async_embed_nodes(
    nodes: Sequence[BaseNode] | Sequence[str], embed_model: OLLAMA_EMBED_MODELS | str, show_progress: bool = False
) -> dict[str, list[float]]:
    """Async get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode] | Sequence[str]): The nodes or texts to embed.
        embed_model (OLLAMA_EMBED_MODELS): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    Returns:
        dict[str, list[float]]: A map from node id to embedding.
    """
    id_to_embed_map: dict[str, list[float]] = {}

    texts_to_embed: list[str] = []
    ids_to_embed: list[str] = []

    if isinstance(nodes[0], BaseNode):
        texts_to_embed = [node.text for node in nodes]
        ids_to_embed = [node.node_id for node in nodes]
    else:
        texts_to_embed = nodes
        ids_to_embed = [generate_unique_hash(text) for text in nodes]

    embedding_function = get_embedding_function(
        model_name=embed_model
    )
    new_embeddings = embedding_function(texts_to_embed)

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id] = text_embedding

    return id_to_embed_map


class VectorStoreIndex(BaseVectorStoreIndex):
    model_name: OLLAMA_EMBED_MODELS = Field(
        default="mxbai-embed-large",
        description="The Ollama model to use.",
    )

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> list[BaseNode]:
        """
        Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = embed_nodes(
            nodes, self._embed_model.model_name, show_progress=show_progress
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.model_copy()
            result.embedding = embedding
            results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> list[BaseNode]:
        """
        Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = await async_embed_nodes(
            nodes=nodes,
            embed_model=self._embed_model.model_name,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = node.model_copy()
            result.embedding = embedding
            results.append(result)
        return results


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
