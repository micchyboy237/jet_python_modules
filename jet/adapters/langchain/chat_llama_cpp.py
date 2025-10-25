"""Llama.cpp chat model integration with OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import (
    JsonOutputKeyToolsParser,
    JsonOutputParser,
    PydanticOutputParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from openai import Client, AsyncClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from pydantic import BaseModel, PrivateAttr
from pydantic.json_schema import JsonSchemaValue
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import Self, is_typeddict

from langchain_core.messages.tool import tool_call

from operator import itemgetter

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from pydantic import model_validator

from jet.utils.object import remove_null_keys

log = logging.getLogger(__name__)

def _get_usage_metadata_from_generation_info(
    generation_info: Optional[Union[ChatCompletion, ChatCompletionChunk, dict[str, Any]]],
) -> Optional[UsageMetadata]:
    """Get usage metadata from OpenAI generation info."""
    if generation_info is None:
        return None
    usage = None
    if isinstance(generation_info, (ChatCompletion, ChatCompletionChunk)):
        usage = generation_info.usage
    elif isinstance(generation_info, dict) and "usage" in generation_info:
        usage = generation_info["usage"]
    if not usage:
        return None
    return UsageMetadata(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )

def _lc_tool_call_to_openai_tool_call(tool_call_: ToolCall) -> dict:
    """Convert a LangChain tool call to an OpenAI tool call format."""
    return {
        "type": "function",
        "id": tool_call_["id"],
        "function": {
            "name": tool_call_["name"],
            "arguments": tool_call_["args"],
        },
    }

def _get_image_from_data_content_block(block: dict) -> str:
    """Format standard data content block to OpenAI-compatible format."""
    if block["type"] == "image":
        if block["source_type"] == "base64":
            return block["data"]
        raise ValueError("Image data only supported through in-line base64 format.")
    raise ValueError(f"Blocks of type {block['type']} not supported.")

def _get_tool_calls_from_response(
    response: Union[ChatCompletion, dict[str, Any]],
) -> list[ToolCall]:
    """Get tool calls from OpenAI response."""
    tool_calls = []
    if isinstance(response, dict) and "choices" in response and response["choices"]:
        choice = response["choices"][0]
        if "message" in choice and (raw_tool_calls := choice["message"].get("tool_calls")):
            tool_calls.extend(
                [
                    tool_call(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        args=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in raw_tool_calls
                ]
            )
    elif isinstance(response, ChatCompletionChunk) and response.choices:
        choice = response.choices[0]
        if choice.delta.tool_calls:
            tool_calls.extend(
                [
                    tool_call(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments or "{}"),
                    )
                    for tc in choice.delta.tool_calls
                    if tc.function and tc.function.name
                ]
            )
    return tool_calls

def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)

class ChatLlamaCpp(BaseChatModel):
    r"""Llama.cpp chat model integration with OpenAI-compatible API.

    .. dropdown:: Setup
        :open:

        Ensure the llama.cpp server is running with an OpenAI-compatible API at
        `http://shawn-pc.local:8080/v1` and the desired model is loaded.

        Install the required package:

        .. code-block:: bash

            pip install -U openai

    Key init args — completion params:
        model: str
            Name of the llama.cpp model to use.
        reasoning: Optional[Union[bool, str]]
            Controls reasoning/thinking mode for supported models.

            - ``True``: Enables reasoning mode. The model's reasoning process will be
              captured and returned in the ``additional_kwargs`` of the response message,
              under ``reasoning_content``. The main response content will not include
              reasoning tags.
            - ``False``: Disables reasoning mode. No reasoning content is included.
            - ``None`` (Default): Uses the model's default reasoning behavior, which may
              include reasoning content in the main response.
            - ``str``: e.g., ``'low'``, ``'medium'``, ``'high'``. Enables reasoning with
              a custom intensity level, if supported by the model.
        temperature: Optional[float]
            Sampling temperature. Ranges from ``0.0`` to ``2.0``. Default: ``0.8``.
        max_tokens: Optional[int]
            Maximum number of tokens to generate. Default: ``128``.

    Instantiate:
        .. code-block:: python

            from chat_llama_cpp import ChatLlamaCpp

            llm = ChatLlamaCpp(
                model="gpt-oss:20b",
                validate_model_on_init=True,
                temperature=0.8,
                max_tokens=256,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='J\'adore programmer.', response_metadata={'model': 'gpt-oss:20b', 'created': 1698162061, 'usage': {'prompt_tokens': 32, 'completion_tokens': 71, 'total_tokens': 103}}, id='run-ba48f958-6402-41a5-b461-5e250a4ebd36-0')

    Stream:
        .. code-block:: python

            for chunk in llm.stream("Return the words Hello World!"):
                print(chunk.text, end="")

        .. code-block:: python

            Hello World!

    Async:
        .. code-block:: python

            await llm.ainvoke("Hello how are you!")

        .. code-block:: python

            AIMessage(content="Hi there! I'm ready to help with any questions or tasks you may have!", response_metadata={'model': 'gpt-oss:20b', 'created': 1698162061, 'usage': {'prompt_tokens': 10, 'completion_tokens': 47, 'total_tokens': 57}}, id='run-29c510ae-49a4-4cdd-8f23-b972bfab1c49-0')

    JSON mode:
        .. code-block:: python

            json_llm = ChatLlamaCpp(format="json")
            llm.invoke(
                "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. "
                "Respond using JSON only."
            ).content

        .. code-block:: python

            '{"location": "Pune, India", "time_of_day": "morning"}'

    Tool Calling:
        .. code-block:: python

            from chat_llama_cpp import ChatLlamaCpp
            from pydantic import BaseModel, Field

            class Multiply(BaseModel):
                a: int = Field(..., description="First integer")
                b: int = Field(..., description="Second integer")

            ans = await llm.invoke("What is 45*67")
            ans.tool_calls

        .. code-block:: python

            [
                {
                    "name": "Multiply",
                    "args": {"a": 45, "b": 67},
                    "id": "call_Ao02pnFYXD6GN1yzc0uXPsvF",
                    "type": "tool_call",
                }
            ]

    Thinking / Reasoning:
        You can enable reasoning mode for models that support it by setting
        the ``reasoning`` parameter to ``True`` in either the constructor or
        the ``invoke``/``stream`` methods. This will enable the model to think
        through the problem and return the reasoning process separately in the
        ``additional_kwargs`` of the response message, under ``reasoning_content``.

        If ``reasoning`` is set to ``None``, the model will use its default reasoning
        behavior, and any reasoning content will be included in the main response content.

        .. code-block:: python

            from chat_llama_cpp import ChatLlamaCpp

            llm = ChatLlamaCpp(
                model="gpt-oss:20b",
                validate_model_on_init=True,
                reasoning=True,
            )

            llm.invoke("How many r's in the word strawberry?")

        .. code-block:: python

            AIMessage(content='The word "strawberry" contains **three r letters**.', additional_kwargs={'reasoning_content': 'Let\'s break down the word "strawberry": s-t-r-a-w-b-e-r-r-y. Counting the r\'s: 1 (t-r), 2 (r-r), 3 (r-y). Thus, there are three r\'s.'}, response_metadata={'model': 'gpt-oss:20b', 'created': 1698162061, 'usage': {'prompt_tokens': 10, 'completion_tokens': 3615, 'total_tokens': 3625}}, id='run-18f8269f-6a35-4a7c-826d-b89d52c753b3-0')

    Structured Output:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class AnswerWithJustification(BaseModel):
                answer: str
                justification: str

            llm = ChatLlamaCpp(model="gpt-oss:20b", temperature=0)
            structured_llm = llm.with_structured_output(AnswerWithJustification)
            structured_llm.invoke("What weighs more: a pound of bricks or a pound of feathers?")

        .. code-block:: python

            AnswerWithJustification(
                answer='They weigh the same',
                justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            )
    """

    model: str
    """Model name to use."""

    reasoning: Optional[Union[bool, str]] = None
    """Controls reasoning/thinking mode for supported models.

    - ``True``: Enables reasoning mode. The model's reasoning process will be
      captured and returned in the ``additional_kwargs`` of the response message,
      under ``reasoning_content``. The main response content will not include
      reasoning tags.
    - ``False``: Disables reasoning mode. No reasoning content is included.
    - ``None`` (Default): Uses the model's default reasoning behavior, which may
      include reasoning content in the main response.
    - ``str``: e.g., ``'low'``, ``'medium'``, ``'high'``. Enables reasoning with
      a custom intensity level, if supported by the model.
    """

    validate_model_on_init: bool = False
    """Whether to validate the model exists on the server on initialization."""

    temperature: Optional[float] = None
    """Sampling temperature. Ranges from ``0.0`` to ``2.0``. Default: ``0.8``."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate. Default: ``128``."""

    top_p: Optional[float] = None
    """Top-p sampling probability. Ranges from ``0.0`` to ``1.0``. Default: ``0.9``."""

    frequency_penalty: Optional[float] = None
    """Penalizes token repetition. Ranges from ``-2.0`` to ``2.0``. Default: ``0.0``."""

    stop: Optional[list[str]] = None
    """Stop tokens to use."""

    format: Optional[Union[Literal["", "json"], JsonSchemaValue]] = None
    """Specify the format of the output (options: ``'json'``, JSON schema)."""

    base_url: Optional[str] = None
    """Base URL the model is hosted under. Default: ``http://shawn-pc.local:8080/v1``."""

    client_kwargs: Optional[dict] = {}
    """Additional kwargs to pass to the httpx clients."""

    async_client_kwargs: Optional[dict] = {}
    """Additional kwargs for the httpx AsyncClient."""

    sync_client_kwargs: Optional[dict] = {}
    """Additional kwargs for the httpx Client."""

    _client: Client = PrivateAttr()
    """The client to use for making requests."""

    _async_client: AsyncClient = PrivateAttr()
    """The async client to use for making requests."""

    @model_validator(mode="after")
    def _set_clients(self) -> Self:
        """Set clients for OpenAI-compatible llama.cpp server."""
        client_kwargs = self.client_kwargs or {}

        sync_client_kwargs = client_kwargs
        if self.sync_client_kwargs:
            sync_client_kwargs = {**sync_client_kwargs, **self.sync_client_kwargs}

        async_client_kwargs = client_kwargs
        if self.async_client_kwargs:
            async_client_kwargs = {**async_client_kwargs, **self.async_client_kwargs}

        base_url = self.base_url or "http://shawn-pc.local:8080/v1"
        self._client = Client(base_url=base_url, **sync_client_kwargs)
        self._async_client = AsyncClient(base_url=base_url, **async_client_kwargs)
        if self.validate_model_on_init:
            try:
                self._client.models.list()
            except Exception as e:
                raise ValueError(f"Failed to validate model on server: {e}")
        return self

    def _convert_messages_to_openai_messages(
        self, messages: list[BaseMessage]
    ) -> Sequence[ChatCompletionMessageParam]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for message in messages:
            role: str
            tool_call_id: Optional[str] = None
            tool_calls: Optional[list[dict[str, Any]]] = None
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
                tool_calls = (
                    [
                        _lc_tool_call_to_openai_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ]
                    if message.tool_calls
                    else None
                )
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, ChatMessage):
                role = message.role
            elif isinstance(message, ToolMessage):
                role = "tool"
                tool_call_id = message.tool_call_id
            else:
                raise ValueError(f"Received unsupported message type: {type(message)}")

            content: Union[str, list[dict[str, Any]]] = ""
            if isinstance(message.content, str):
                content = message.content
            else:
                content = []
                for content_part in message.content:
                    if isinstance(content_part, str):
                        content.append({"type": "text", "text": content_part})
                    elif content_part.get("type") == "text":
                        content.append({"type": "text", "text": content_part["text"]})
                    elif content_part.get("type") == "tool_use":
                        continue
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = temp_image_url
                        elif (
                            isinstance(temp_image_url, dict)
                            and "url" in temp_image_url
                            and isinstance(temp_image_url["url"], str)
                        ):
                            image_url = temp_image_url["url"]
                        else:
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )
                        content.append({"type": "image_url", "image_url": {"url": image_url}})
                    elif is_data_content_block(content_part):
                        image = _get_image_from_data_content_block(content_part)
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must be 'text', 'image_url', or a valid data content block."
                        )
                if not content:
                    content = ""

            msg: ChatCompletionMessageParam = {"role": role, "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
            openai_messages.append(msg)

        return openai_messages

    def _chat_params(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        openai_messages = self._convert_messages_to_openai_messages(messages)

        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        if self.stop is not None:
            stop = self.stop

        params = {
            "messages": openai_messages,
            "stream": kwargs.pop("stream", True),
            "model": kwargs.pop("model", self.model),
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "top_p": kwargs.pop("top_p", self.top_p),
            "frequency_penalty": kwargs.pop("frequency_penalty", self.frequency_penalty),
            "stop": stop,
            "tool_choice": "auto",
        }

        if tools := kwargs.get("tools"):
            params["tools"] = tools

        if self.format:
            params["response_format"] = (
                {"type": "json_object"} if self.format == "json" else self.format
            )

        if reasoning := kwargs.get("reasoning", self.reasoning):
            params["messages"].insert(
                0,
                {
                    "role": "system",
                    "content": f"Enable reasoning mode: {reasoning}. Provide detailed reasoning steps before answering.",
                },
            )

        return remove_null_keys(params)

    def _create_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[ChatCompletionChunk, dict[str, Any]]]:
        chat_params = self._chat_params(messages, stop, **kwargs)
        if chat_params["stream"]:
            yield from self._client.chat.completions.create(**chat_params)
        else:
            yield self._client.chat.completions.create(**chat_params).dict()

    async def _acreate_chat_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[ChatCompletionChunk, dict[str, Any]]]:
        chat_params = self._chat_params(messages, stop, **kwargs)
        if chat_params["stream"]:
            async for part in await self._async_client.chat.completions.create(**chat_params):
                yield part
        else:
            yield (await self._async_client.chat.completions.create(**chat_params)).dict()

    def _chat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("No data received from llama.cpp stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    chunk=chunk,
                    verbose=verbose,
                )
        if final_chunk is None:
            raise ValueError("No data received from llama.cpp stream.")

        return final_chunk

    def _iterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if isinstance(stream_resp, str):
                continue
            choices = stream_resp.choices if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.delta if isinstance(choice, Choice) else choice.get("delta", {})
            content = delta.content if isinstance(delta, ChoiceDelta) else delta.get("content", "")

            generation_info = None
            if choice.finish_reason:
                generation_info = {
                    "model_name": stream_resp.model if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("model"),
                    "created": stream_resp.created if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("created"),
                    "usage": stream_resp.usage if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("usage"),
                }

            additional_kwargs = {}
            if reasoning and content:
                additional_kwargs["reasoning_content"] = content

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content or "",
                    additional_kwargs=additional_kwargs,
                    usage_metadata=_get_usage_metadata_from_generation_info(stream_resp),
                    tool_calls=_get_tool_calls_from_response(stream_resp),
                ),
                generation_info=generation_info,
            )
            yield chunk

    async def _aiterate_over_stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reasoning = kwargs.get("reasoning", self.reasoning)
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if isinstance(stream_resp, str):
                continue
            choices = stream_resp.choices if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("choices", [])
            if not choices:
                continue

            choice = choices[0]
            delta = choice.delta if isinstance(choice, Choice) else choice.get("delta", {})
            content = delta.content if isinstance(delta, ChoiceDelta) else delta.get("content", "")

            generation_info = None
            if choice.finish_reason:
                generation_info = {
                    "model_name": stream_resp.model if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("model"),
                    "created": stream_resp.created if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("created"),
                    "usage": stream_resp.usage if isinstance(stream_resp, ChatCompletionChunk) else stream_resp.get("usage"),
                }

            additional_kwargs = {}
            if reasoning and content:
                additional_kwargs["reasoning_content"] = content

            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=content or "",
                    additional_kwargs=additional_kwargs,
                    usage_metadata=_get_usage_metadata_from_generation_info(stream_resp),
                    tool_calls=_get_tool_calls_from_response(stream_resp),
                ),
                generation_info=generation_info,
            )
            yield chunk

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="llamacpp",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = self._chat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(AIMessageChunk, final_chunk.message).usage_metadata,
                tool_calls=cast(AIMessageChunk, final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        result = ChatResult(generations=[chat_generation])
        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._iterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for chunk in self._aiterate_over_stream(messages, stop, **kwargs):
            if run_manager:
                await run_manager.on_llm_new_token(
                    chunk.text,
                    verbose=self.verbose,
                )
            yield chunk

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = await self._achat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        generation_info = final_chunk.generation_info
        chat_generation = ChatGeneration(
            message=AIMessage(
                content=final_chunk.text,
                usage_metadata=cast(AIMessageChunk, final_chunk.message).usage_metadata,
                tool_calls=cast(AIMessageChunk, final_chunk.message).tool_calls,
                additional_kwargs=final_chunk.message.additional_kwargs,
            ),
            generation_info=generation_info,
        )
        return ChatResult(generations=[chat_generation])

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-llamacpp"

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "any"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: If provided, specifies which tool to call. Options: 'auto', 'any', tool name, or True.
            kwargs: Additional parameters passed to ``self.bind(**kwargs)``.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: Union[dict, type],
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema (Pydantic class, JSON schema, TypedDict, or OpenAI tool schema).
            method: The method for steering model generation ('json_schema', 'function_calling', 'json_mode').
            include_raw: If True, returns raw response, parsed output, and parsing error.
            kwargs: Additional keyword args are not supported.

        Returns:
            A Runnable outputting structured data according to the schema.
        """
        _ = kwargs.pop("strict", None)
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError("schema must be specified when method is not 'json_mode'. Received None.")
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": formatted_tool,
                },
            )
            output_parser: Runnable = (
                PydanticToolsParser(tools=[schema], first_tool_only=True)
                if is_pydantic_schema
                else JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            )
        elif method == "json_mode":
            llm = self.bind(
                format="json",
                ls_structured_output_format={
                    "kwargs": {"method": method},
                    "schema": schema,
                },
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        elif method == "json_schema":
            if schema is None:
                raise ValueError("schema must be specified when method is not 'json_mode'. Received None.")
            if is_pydantic_schema:
                schema = cast(TypeBaseModel, schema)
                if issubclass(schema, BaseModelV1):
                    response_format = schema.schema()
                else:
                    response_format = schema.model_json_schema()
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": schema,
                    },
                )
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                if is_typeddict(schema):
                    response_format = convert_to_json_schema(schema)
                    if "required" not in response_format:
                        response_format["required"] = list(response_format["properties"].keys())
                else:
                    response_format = cast(dict, schema)
                llm = self.bind(
                    format=response_format,
                    ls_structured_output_format={
                        "kwargs": {"method": method},
                        "schema": response_format,
                    },
                )
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling', "
                f"'json_schema', or 'json_mode'. Received: '{method}'"
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser
