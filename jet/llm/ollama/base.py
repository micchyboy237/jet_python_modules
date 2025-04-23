from jet.token.token_utils import get_model_max_tokens
from typing import Optional, Any, AsyncGenerator
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from jet.llm.llm_types import MessageRole
from langchain_postgres import PostgresChatMessageHistory
import psycopg
import uuid
from jet.data.utils import generate_unique_hash
from jet.llm.utils.embeddings import get_embedding_function
from jet.token.token_utils import get_model_max_tokens, tokenize
from jet.transformers.object import make_serializable
from llama_index.core import VectorStoreIndex as BaseVectorStoreIndex
from collections import defaultdict
from typing import AsyncGenerator, Callable, Dict, Optional, Sequence, Type, TypedDict, Any, Union
from jet.decorators.error import wrap_retry
from jet.decorators.function import retry_on_error
from jet.llm.ollama.constants import DEFAULT_BASE_URL, DEFAULT_CONTEXT_WINDOW, DEFAULT_EMBED_BATCH_SIZE, DEFAULT_REQUEST_TIMEOUT, OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_SMALL_CHUNK_OVERLAP, OLLAMA_SMALL_CHUNK_SIZE, OLLAMA_SMALL_EMBED_MODEL
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_CONTEXTS, OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from jet.logger.timer import sleep_countdown, time_it
from llama_index.core.base.llms.types import (
    ChatMessage, ChatResponse as BaseChatResponse,
    ImageBlock,
    TextBlock,
)
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
from ollama import Message as OllamaMessage, ChatResponse as OllamaChatResponse
from jet.llm.tools.types import BaseTool

NON_DETERMINISTIC_LLM_SETTINGS = {
    # "seed": random.randint(0, 1000),
    "temperature": 0.6,
    "num_keep": 0,
    "num_predict": -1,
}

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


class ChatResponse(BaseChatResponse):
    """Extended chat response that adds the content attribute."""

    @property
    def content(self) -> str:
        """Returns the content from the associated ChatMessage."""
        return self.message.content

    def __str__(self) -> str:
        return self.content


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


DEFAULT_DB = "chat_history_db1"
DEFAULT_USER = "jethroestrada"
DEFAULT_PASSWORD = ""
DEFAULT_HOST = "jetairm1"
DEFAULT_PORT = 5432
DEFAULT_TABLE_NAME = "chat_history"

# Initialize database connection
sync_connection = psycopg.connect(
    dbname=DEFAULT_DB,
    user=DEFAULT_USER,
    password=DEFAULT_PASSWORD,
    host=DEFAULT_HOST,
    port=DEFAULT_PORT
)

# Create table if it doesn't exist
PostgresChatMessageHistory.create_tables(sync_connection, DEFAULT_TABLE_NAME)


class ChatHistory:
    def __init__(self, session_id: Optional[str] = None, table_name: str = DEFAULT_TABLE_NAME):
        if not session_id:
            session_id = generate_unique_hash()
        self.session_id = session_id
        self.table_name = table_name
        self.history = PostgresChatMessageHistory(
            table_name,
            session_id,
            sync_connection=sync_connection
        )

    def get_messages(self) -> list[dict]:
        """Retrieve chat history messages, converted to Ollama-compatible format."""
        messages = self.history.get_messages()
        return [
            {
                "role": "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            }
            for msg in messages
        ]

    def get_turn_count(self) -> int:
        """Return the number of turns, excluding the system message."""
        messages = self.history.get_messages()
        return len([msg for msg in messages if not isinstance(msg, SystemMessage)])

    def add_messages(self, messages: list[dict]) -> None:
        """Add messages to the chat history."""
        for msg in messages:
            if msg["role"] == "system":
                self.history.add_messages(
                    [SystemMessage(content=msg["content"])])
            elif msg["role"] == "user":
                self.history.add_messages(
                    [HumanMessage(content=msg["content"])])
            elif msg["role"] == "assistant":
                self.history.add_messages([AIMessage(content=msg["content"])])

    def clear(self) -> None:
        """Clear the chat history."""
        self.history.clear()

    def add_system_message(self, content: str) -> None:
        """Add a system message to the chat history."""
        self.history.add_messages([SystemMessage(content=content)])


class Ollama(BaseOllama, BaseModel):
    model: str = Field(default="llama3.1")
    max_tokens: Optional[Union[int, float]] = Field(default=None)
    max_prediction_ratio: Optional[float] = Field(default=None)
    system_prompt: Optional[str] = Field(default=None)
    session_id: str = Field(default_factory=generate_unique_hash)
    table_name: str = Field(default="chat_history")
    chat_history: Optional[ChatHistory] = Field(default=None)

    def __init__(self, model: str, system: Optional[str] = None, session_id: Optional[str] = None, table_name: str = "chat_history", **kwargs) -> None:
        if not session_id:
            session_id = generate_unique_hash()

        context_window = kwargs.get("context_window")
        temperature = kwargs.get("temperature", 0.3)
        max_model_tokens = get_model_max_tokens(model)
        if not context_window or context_window > max_model_tokens:
            context_window = max_model_tokens
        kwargs = {
            **kwargs,
            "context_window": context_window,
            "temperature": temperature,
        }

        # Initialize chat_history
        chat_history = ChatHistory(
            session_id=session_id, table_name=table_name)

        # Initialize Pydantic model with all fields
        super().__init__(
            model=model,
            system_prompt=system,
            session_id=session_id,
            table_name=table_name,
            chat_history=chat_history,
            **kwargs
        )

    async def stream_chat(self, query: str, context: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> AsyncGenerator[str, None]:
        from jet.actions.generation import call_ollama_chat
        from jet.token.token_utils import token_counter, get_ollama_tokenizer

        # Initialize tokenizer
        tokenizer = get_ollama_tokenizer(self.model)
        set_global_tokenizer(tokenizer)

        model = model or self.model
        tools = kwargs.get("tools", None)
        format = kwargs.get("format", "json" if self.json_mode else None)
        options = kwargs.get("options", {})
        system = kwargs.get("system", self.system_prompt)

        # Get history messages
        history_messages = self.chat_history.get_messages()

        user_input = query
        new_user_msg = {"role": "user", "content": user_input}

        messages = history_messages + [new_user_msg]

        if system and not any(msg["role"] == "system" for msg in history_messages):
            system_msg = {"role": "system", "content": system}
            messages.insert(0, system_msg)

        settings = {
            **kwargs,
            "model": model,
            "messages": messages,
            "context": context,
            "stream": True,
            "format": format,
            "tools": tools,
            "keep_alive": self.keep_alive,
            "full_stream_response": True,
            "options": {
                **self._model_kwargs,
                **options,
            },
        }

        response = call_ollama_chat(**settings)

        if isinstance(response, dict) and "error" in response:
            raise ValueError(f"Ollama API error:\n{response['error']}")

        content = ""
        role = ""
        tool_calls = []

        for chunk in response:
            chunk_content = chunk["message"]["content"]
            content += chunk_content
            yield chunk_content

            if not role:
                role = chunk["message"]["role"]

            if chunk["done"]:
                # Save history
                if system and not any(isinstance(msg, SystemMessage) for msg in self.chat_history.history.get_messages()):
                    self.chat_history.clear()
                    self.chat_history.add_system_message(system)

                self.chat_history.add_messages([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": content},
                ])

    def chat(self, messages: str | Sequence[ChatMessage] | PromptTemplate, **kwargs: Any) -> ChatResponse:
        from jet.actions.generation import call_ollama_chat
        from jet.token.token_utils import token_counter, get_ollama_tokenizer

        # Initialize and set tokenizer
        tokenizer = get_ollama_tokenizer(self.model)
        set_global_tokenizer(tokenizer)

        tools = kwargs.get("tools", None)
        format = kwargs.get("format", "json" if self.json_mode else None)
        options = kwargs.get("options", {})
        stream = kwargs.get("stream", not tools)
        template_vars = kwargs.get("template_vars", {})
        system = kwargs.get("system", self.system_prompt)

        # Get existing messages from history
        history_messages = self.chat_history.get_messages()

        if history_messages:
            history_messages = _convert_to_ollama_messages(history_messages)

        ollama_messages = messages
        if isinstance(messages, list) and isinstance(messages[0], (ChatMessage, BaseMessage)):
            ollama_messages = _convert_to_ollama_messages(messages)
        elif isinstance(messages, PromptTemplate):
            ollama_messages = messages.format(**template_vars)

        # Combine history with new messages
        if isinstance(ollama_messages, list):
            combined_messages = history_messages + ollama_messages
        else:
            combined_messages = history_messages + \
                [{"role": "user", "content": ollama_messages}]

        system_messages = [system] if system else []

        if isinstance(messages, list):
            system_messages = system_messages + [
                m['content'] for m in combined_messages if m['role'] == MessageRole.SYSTEM]

            system = "\n\n".join(system_messages)
            # Remove all system messages from the original list
            combined_messages = [
                m for m in combined_messages if m['role'] != MessageRole.SYSTEM]

        settings = {
            **kwargs,
            "system": system,
            "model": self.model,
            "messages": combined_messages,
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

                prompt_token_count = token_counter(
                    combined_messages, self.model)
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

                # Save messages to history
                user_message = ollama_messages if isinstance(
                    ollama_messages, str) else ollama_messages[-1]['content']

                if system:
                    chat_messages = self.chat_history.get_messages()
                    has_system_message = any(isinstance(
                        message, SystemMessage) for message in chat_messages)
                    if not has_system_message:
                        # Insert system message at the beginning of history
                        self.chat_history.clear()  # Clear existing messages
                        self.chat_history.add_messages([
                            SystemMessage(content=system),
                        ] + chat_messages)

                self.chat_history.add_messages([
                    HumanMessage(content=user_message),
                    AIMessage(content=content),
                ])

            else:
                content = ""
                role = ""
                tool_calls = []

                if isinstance(response, dict) and "error" in response:
                    raise ValueError(f"Ollama API error:\n{response['error']}")

                for chunk in response:
                    content += chunk["message"]["content"]
                    if not role:
                        role = chunk["message"]["role"]
                    if chunk["done"]:
                        prompt_token_count = token_counter(
                            combined_messages, self.model)
                        response_token_count = token_counter(
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

                        # Save messages to history
                        user_message = ollama_messages if isinstance(
                            ollama_messages, str) else ollama_messages[-1]['content']

                        if system:
                            self.chat_history.add_messages([
                                SystemMessage(content=system),
                            ])
                        self.chat_history.add_messages([
                            HumanMessage(content=user_message),
                            AIMessage(content=content),
                        ])

            chat_response = ChatResponse(
                message=ChatMessage(
                    role=role,
                    content=content,
                    additional_kwargs={"tool_calls": tool_calls},
                ),
                raw=final_response,
            )
            return chat_response

        return wrap_retry(run)

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        model: Optional[OLLAMA_MODEL_NAMES] = None,
        llm_kwargs: dict[str, Any] = {},
        **prompt_args: Any,
    ) -> BaseModel:
        # Get existing messages from history
        history_messages = self.chat_history.get_messages()

        if self.pydantic_program_mode == PydanticProgramMode.DEFAULT:
            llm_kwargs["format"] = output_cls.model_json_schema()

            llm_kwargs = {
                **llm_kwargs,
                "model": llm_kwargs.get("model", model),
                "system": llm_kwargs.get("system", self.system_prompt),
            }

            messages = prompt.format_messages(**prompt_args)
            # Combine with history
            combined_messages = history_messages + messages
            response = self.chat(combined_messages, **llm_kwargs)

            extracted_result = extract_json_block_content(
                response.message.content or "")
            validation_result = validate_json(
                extracted_result, output_cls.model_json_schema())

            # Save messages to history
            self.chat_history.add_messages(messages + [response.message])

            return output_cls.model_validate_json(json.dumps(validation_result["data"]))
        else:
            return super().structured_predict(output_cls, prompt, llm_kwargs, **prompt_args)

    def encode(self, texts: Union[str, Sequence[str]] = '') -> list[int] | list[list[int]]:
        """Calls get_general_text_embedding to get the embeddings."""
        tokens = tokenize(self.model, texts)
        return tokens


class OllamaEmbedding(BaseOllamaEmbedding):
    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted by Ollama",
    )
    model_name: OLLAMA_EMBED_MODELS = Field(
        default=OLLAMA_SMALL_EMBED_MODEL,
        description="The Ollama model to use.",
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )

    def encode(self, texts: Union[str, Sequence[str]] = '') -> list[int] | list[list[int]]:
        """Calls get_general_text_embedding to get the embeddings."""
        tokens = tokenize(self.model_name, texts)
        return tokens

    def get_general_text_embedding(self, texts: Union[str, Sequence[str]] = '',) -> list[float] | list[list[float]]:
        """Get Ollama embedding with retry mechanism."""
        # logger.orange("Calling OllamaEmbedding embed...")
        # logger.debug(
        #     "Embed model:",
        #     self.model_name,
        #     f"({OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]})",
        #     colors=["GRAY", "DEBUG", "DEBUG"],
        # )
        # logger.debug(f"Max Context: {OLLAMA_MODEL_CONTEXTS[self.model_name]}")
        # logger.debug(
        #     f"Embeddings Dim: {OLLAMA_MODEL_EMBEDDING_TOKENS[self.model_name]}")

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

            # logger.log("Batch Tokens:", len(embeddings),
            #            colors=["DEBUG", "SUCCESS"])
            return embeddings

        return wrap_retry(run)


def _convert_to_ollama_messages(messages: Sequence[ChatMessage] | list[BaseMessage]) -> Dict:
    ollama_messages = []
    for message in messages:
        if isinstance(message, BaseMessage):
            content = message.content
            if isinstance(message, SystemMessage):
                role = MessageRole.SYSTEM
            elif isinstance(message, HumanMessage):
                role = MessageRole.USER
            elif isinstance(message, AIMessage):
                role = MessageRole.ASSISTANT

            cur_ollama_message = {
                "role": role,
                "content": content,
            }

            if "tool_calls" in message.additional_kwargs:
                cur_ollama_message["tool_calls"] = message.additional_kwargs[
                    "tool_calls"
                ]

            ollama_messages.append(cur_ollama_message)

        else:
            cur_ollama_message = {
                "role": message.role.value,
                "content": "",
            }
            for block in message.blocks:
                if isinstance(block, TextBlock):
                    cur_ollama_message["content"] += block.text
                elif isinstance(block, ImageBlock):
                    if "images" not in cur_ollama_message:
                        cur_ollama_message["images"] = []
                    cur_ollama_message["images"].append(
                        block.resolve_image(
                            as_base64=True).read().decode("utf-8")
                    )
                else:
                    raise ValueError(f"Unsupported block type: {type(block)}")

            if "tool_calls" in message.additional_kwargs:
                cur_ollama_message["tool_calls"] = message.additional_kwargs[
                    "tool_calls"
                ]

            ollama_messages.append(cur_ollama_message)

    return ollama_messages


def chat(
    messages: str | Sequence[ChatMessage] | Sequence[OllamaMessage] | PromptTemplate,
    model: str = "llama3.2",
    *,
    system: Optional[str] = None,
    context: Optional[str] = None,
    format: Optional[Union[str, dict]] = None,
    stream: bool = True,
    tools=[],
    **kwargs: Any,
) -> ChatResponse:
    from jet.actions.generation import call_ollama_chat
    # from jet.token.token_utils import token_counter

    stream = stream if not tools else False
    template_vars = kwargs.pop("template_vars", {})

    ollama_messages = messages
    if isinstance(messages, list) and isinstance(messages[0], ChatMessage):
        ollama_messages = _convert_to_ollama_messages(
            messages) if not isinstance(messages, str) else messages
    elif isinstance(messages, PromptTemplate):
        ollama_messages = messages.format(**template_vars)

    settings = {
        "model": model,
        "messages": ollama_messages,
        "system": system,
        "context": context,
        "stream": stream,
        "format": format,
        "tools": tools,
        "full_stream_response": True,
        "options": {
            **NON_DETERMINISTIC_LLM_SETTINGS,
            **kwargs,
        },
    }

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

        # prompt_token_count = token_counter(messages, model)
        # response_token_count = token_counter(
        #     final_response_content, model)

        final_response = {
            **response.copy(),
            # "usage": {
            #     "prompt_tokens": prompt_token_count,
            #     "completion_tokens": response_token_count,
            #     "total_tokens": prompt_token_count + response_token_count,
            # }
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
                # prompt_token_count: int = token_counter(
                #     messages, model)
                # response_token_count: int = token_counter(
                #     content, model)

                updated_chunk = chunk.copy()
                updated_chunk["message"]["content"] = content

                final_response = {
                    **updated_chunk,
                    # "usage": {
                    #     "prompt_tokens": prompt_token_count,
                    #     "completion_tokens": response_token_count,
                    #     "total_tokens": prompt_token_count + response_token_count,
                    # }
                }

    final_response.pop("message")
    chat_response = OllamaChatResponse(
        message=OllamaMessage(
            role=role,
            content=content,
            tool_calls=tool_calls,
        ),
        **final_response
    )
    return chat_response


async def achat(messages: str | Sequence[ChatMessage] | Sequence[OllamaMessage] | PromptTemplate, **kwargs: Any) -> ChatResponse:
    return chat(messages, **kwargs)


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
    def __init__(
        self,
        *args,
        # Accept model_name explicitly
        model_name: OLLAMA_EMBED_MODELS = OLLAMA_SMALL_EMBED_MODEL,
        embed_model: Optional[EmbedType] = None,
        **kwargs
    ):
        if not embed_model:
            embed_model = OllamaEmbedding(model_name=model_name)

        super().__init__(
            *args,
            embed_model=embed_model,
            **kwargs
        )

        self.model_name = self._embed_model.model_name

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
