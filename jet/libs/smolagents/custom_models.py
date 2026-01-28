# custom_models.py

import json
import logging
import os
from pathlib import Path
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from smolagents.monitoring import TokenUsage
from smolagents.tools import Tool
from smolagents.utils import (
    RateLimiter,
    Retrying,
    _is_package_available,
    encode_image_base64,
    make_image_url,
    parse_json_blob,
)


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

RETRY_WAIT = 60
RETRY_MAX_ATTEMPTS = 3
RETRY_EXPONENTIAL_BASE = 2
RETRY_JITTER = True
CODEAGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "additionalProperties": False,
            "properties": {
                "thought": {
                    "description": "A free form text description of the thought process.",
                    "title": "Thought",
                    "type": "string",
                },
                "code": {
                    "description": "Valid Python code snippet implementing the thought.",
                    "title": "Code",
                    "type": "string",
                },
            },
            "required": ["thought", "code"],
            "title": "ThoughtAndCodeAnswer",
            "type": "object",
        },
        "name": "ThoughtAndCodeAnswer",
        "strict": True,
    },
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


def remove_content_after_stop_sequences(
    content: str | None, stop_sequences: list[str] | None
) -> str | None:
    """Remove content after any stop sequence is encountered.

    Some providers may return ``None`` content (for example when responding purely with tool calls),
    so we skip processing in that case.
    """
    if content is None or not stop_sequences:
        return content

    for stop_seq in stop_sequences:
        split = content.split(stop_seq)
        content = split[0]
    return content


@dataclass
class ChatMessageToolCallFunction:
    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallFunction
    id: str
    type: str

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


@dataclass
class ChatMessage:
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API
    token_usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            return
        self.tool_calls = [
            _coerce_tool_call(tool_call) for tool_call in self.tool_calls
        ]

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(
        cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None
    ) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]),
                    id=tc["id"],
                    type=tc["type"],
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        return get_dict_from_nested_dataclasses(self)

    def render_as_markdown(self) -> str:
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps(
                        {
                            "tool": tool.function.name,
                            "arguments": tool.function.arguments,
                        }
                    )
                    for tool in self.tool_calls
                ]
            )
        return rendered


def _coerce_tool_call(tool_call: Any) -> ChatMessageToolCall:
    if isinstance(tool_call, ChatMessageToolCall):
        return tool_call

    if isinstance(tool_call, dict):
        tool_call_dict = tool_call
    elif hasattr(tool_call, "model_dump"):
        tool_call_dict = tool_call.model_dump()
    elif hasattr(tool_call, "dict") and callable(tool_call.dict):
        tool_call_dict = tool_call.dict()

    return ChatMessageToolCall(
        function=ChatMessageToolCallFunction(
            arguments=tool_call_dict["function"]["arguments"],
            name=tool_call_dict["function"]["name"],
        ),
        id=tool_call_dict["id"],
        type=tool_call_dict["type"],
    )


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageToolCallStreamDelta:
    """Represents a streaming delta for tool calls during generation."""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ChatMessageToolCallFunction | None = None


@dataclass
class ChatMessageStreamDelta:
    content: str | None = None
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None
    token_usage: TokenUsage | None = None


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta],
    role: MessageRole = MessageRole.ASSISTANT,
) -> ChatMessage:
    """
    Agglomerate a list of stream deltas into a single stream delta.
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in (
                stream_delta.tool_calls
            ):  # ?ormally there should be only one call at a time
                # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = (
                            ChatMessageToolCallStreamDelta(
                                id=tool_call_delta.id,
                                type=tool_call_delta.type,
                                function=ChatMessageToolCallFunction(
                                    name="", arguments=""
                                ),
                            )
                        )
                    # Update the tool call at the specific index
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if (
                            tool_call_delta.function.name
                            and len(tool_call_delta.function.name) > 0
                        ):
                            tool_call.function.name = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += (
                                tool_call_delta.function.arguments
                            )
                else:
                    raise ValueError(
                        f"Tool call index is not provided in tool delta: {tool_call_delta}"
                    )

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)

        # parse anyOf
        if "anyOf" in value:
            types = []
            enum = None
            for t in value["anyOf"]:
                if t["type"] == "null":
                    value["nullable"] = True
                    continue
                if t["type"] == "any":
                    types.append("string")
                else:
                    types.append(t["type"])
                if "enum" in t:  # assuming there is only one enum in anyOf
                    enum = t["enum"]

            value["type"] = types if len(types) > 1 else types[0]
            if enum is not None:
                value["enum"] = enum

            value.pop("anyOf")

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def get_clean_message_list(
    message_list: list[ChatMessage | dict],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """
    Creates a list of messages to give as input to the LLM. These messages are dictionaries and chat template compatible with transformers LLM chat template.
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`list[ChatMessage | dict]`): List of chat messages. Mixed types are allowed.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(
                f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
            )

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), (
                    "Error: this element should be a dict:" + str(element)
                )
                if element["type"] == "image":
                    assert not flatten_messages_as_text, (
                        f"Cannot use images with {flatten_messages_as_text=}"
                    )
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": make_image_url(
                                        encode_image_base64(element.pop("image"))
                                    )
                                },
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if (
            len(output_message_list) > 0
            and message.role == output_message_list[-1]["role"]
        ):
            assert isinstance(message.content, list), "Error: wrong content:" + str(
                message.content
            )
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                for el in message.content:
                    if (
                        el["type"] == "text"
                        and output_message_list[-1]["content"][-1]["type"] == "text"
                    ):
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += (
                            "\n" + el["text"]
                        )
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(
    text: str, tool_name_key: str, tool_arguments_key: str
) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and the openai/gpt-5 series (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    if model_name == "o3-mini":
        return True
    # o3* (except mini), o4*, all grok-* models, and the gpt-5* family (including versioned variants) don't support stop parameter
    openai_model_pattern = r"(o3(?:$|[-.].*)|o4(?:$|[-.].*)|gpt-5.*)"
    grok_model_pattern = r"([A-Za-z][A-Za-z0-9_-]*\.)?grok-[A-Za-z0-9][A-Za-z0-9_.-]*"
    pattern = rf"^({openai_model_pattern}|{grok_model_pattern})$"

    return not re.match(pattern, model_name)


class _ParameterRemove:
    """Sentinel value to indicate a parameter should be removed."""

    def __repr__(self):
        return "REMOVE_PARAMETER"


# Singleton instance for removing parameters
REMOVE_PARAMETER = _ParameterRemove()


def get_next_call_number(logs_dir: Path) -> int:
    """Find the next available llm_call_NNNN number"""
    if not logs_dir.exists():
        return 1

    existing = [
        int(d.name.split("_")[-1])
        for d in logs_dir.iterdir()
        if d.is_dir() and d.name.startswith("llm_call_") and d.name[9:].isdigit()
    ]
    return max(existing, default=0) + 1


def save_llm_call(
    logs_dir: Path,
    call_number: int,
    is_stream: bool,
    request_data: dict,
    response_data: Any | None = None,
    stream_deltas: list | None = None,
) -> None:
    """Save complete call information in structured files"""
    subdir_name = f"llm_call_{call_number:04d}"
    target_dir = (
        logs_dir / ("generate_stream" if is_stream else "generate") / subdir_name
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. request
    (target_dir / "request.json").write_text(json.dumps(request_data, indent=2))

    # 2. final response (non-stream)
    if response_data is not None and not is_stream:
        if hasattr(response_data, "model_dump_json"):
            (target_dir / "response.json").write_text(response_data.model_dump_json())
        else:
            (target_dir / "response.json").write_text(
                json.dumps(response_data, indent=2, default=str)
            )

    # 3. stream deltas (one line per delta)
    if is_stream and stream_deltas:
        with (target_dir / "stream_deltas.ndjson").open("w", encoding="utf-8") as f:
            for delta in stream_deltas:
                f.write(json.dumps(delta, default=str) + "\n")


class Model:
    """Base class for all language model implementations.

    This abstract class defines the core interface that all model implementations must follow
    to work with agents. It provides common functionality for message handling, tool integration,
    and model configuration while allowing subclasses to implement their specific generation logic.

    Parameters:
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten complex message content into plain text format.
        tool_name_key (`str`, default `"name"`):
            The key used to extract tool names from model responses.
        tool_arguments_key (`str`, default `"arguments"`):
            The key used to extract tool arguments from model responses.
        verbose (`bool`, default `False`):
            Whether to enable verbose logging of model operations.
        logs_dir (`str`, optional):
            Directory for logging model call details. If None, no logs are written.
        model_id (`str`, *optional*):
            Identifier for the specific model being used.
        **kwargs:
            Additional keyword arguments to forward to the underlying model completion call.

    Note:
        This is an abstract base class. Subclasses must implement the `generate()` method
        to provide actual model inference capabilities.

    Example:
        ```python
        class CustomModel(Model):
            def generate(self, messages, **kwargs):
                # Implementation specific to your model
                pass
        ```
    """

    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        verbose: bool = False,
        logs_dir: Optional[str] = None,
        model_id: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self.verbose = verbose
        self.logs_dir = Path(logs_dir).resolve() if logs_dir else None
        self._call_counter: dict[bool, int] = {False: 0, True: 0}  # non-stream / stream
        self.model_id: str | None = model_id

    @property
    def supports_stop_parameter(self) -> bool:
        return supports_stop_parameter(self.model_id or "")

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str
        | dict
        | None = "required",  # Configurable tool_choice parameter
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation.

        Parameter priority (highest to lowest):
        1. self.kwargs (model defaults)
        2. Explicitly passed kwargs
        3. Specific parameters (stop_sequences, response_format, etc.)
        """
        # Clean and standardize the message list
        flatten_messages_as_text = kwargs.pop(
            "flatten_messages_as_text", self.flatten_messages_as_text
        )
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Start with messages
        completion_kwargs = {
            "messages": messages_as_dicts,
        }
        # Override with specific parameters
        if stop_sequences is not None and self.supports_stop_parameter:
            # Some models do not support stop parameter
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format
        if tools_to_call_from:
            completion_kwargs["tools"] = [
                get_tool_json_schema(tool) for tool in tools_to_call_from
            ]
            if tool_choice is not None:
                completion_kwargs["tool_choice"] = tool_choice
        # Override with passed-in kwargs
        completion_kwargs.update(kwargs)
        # Override with self.kwargs
        for kwarg_name, kwarg_value in self.kwargs.items():
            if kwarg_value is REMOVE_PARAMETER:
                completion_kwargs.pop(kwarg_name, None)  # Remove parameter if present
            else:
                completion_kwargs[kwarg_name] = kwarg_value  # Set/override parameter
        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`list[dict[str, str | list[dict]]] | list[ChatMessage]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            response_format (`dict[str, str]`, *optional*):
                The response format to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Sometimes APIs do not return the tool call as a specific object, so we need to parse it."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, (
                "Message contains no content and no tool calls"
            )
            message.tool_calls = [
                get_tool_call_from_text(
                    message.content, self.tool_name_key, self.tool_arguments_key
                )
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(
                tool_call.function.arguments
            )
        return message

    def to_dict(self) -> dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        return cls(**{k: v for k, v in model_dictionary.items()})


class ApiModel(Model):
    """
    Base class for API-based language models.

    This class serves as a foundation for implementing models that interact with
    external APIs. It handles the common functionality for managing model IDs,
    custom role mappings, and API client connections.

    Parameters:
        model_id (`str`):
            The identifier for the model to be used with the API.
        custom_role_conversions (`dict[str, str`], **optional**):
            Mapping to convert  between internal role names and API-specific role names. Defaults to None.
        client (`Any`, **optional**):
            Pre-configured API client instance. If not provided, a default client will be created. Defaults to None.
        requests_per_minute (`float`, **optional**):
            Rate limit in requests per minute.
        retry (`bool`, **optional**):
            Wether to retry on rate limit errors, up to RETRY_MAX_ATTEMPTS times. Defaults to True.
        **kwargs:
            Additional keyword arguments to forward to the underlying model completion call.
    """

    def __init__(
        self,
        model_id: str,
        custom_role_conversions: dict[str, str] | None = None,
        client: Any | None = None,
        requests_per_minute: float | None = None,
        retry: bool = True,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.retryer = Retrying(
            max_attempts=RETRY_MAX_ATTEMPTS if retry else 1,
            wait_seconds=RETRY_WAIT,
            exponential_base=RETRY_EXPONENTIAL_BASE,
            jitter=RETRY_JITTER,
            retry_predicate=is_rate_limit_error,
            reraise=True,
            before_sleep_logger=(logger, logging.INFO),
            after_logger=(logger, logging.INFO),
        )

    def create_client(self):
        """Create the API client for the specific service."""
        raise NotImplementedError(
            "Subclasses must implement this method to create a client"
        )

    def _apply_rate_limit(self):
        """Apply rate limiting before making API calls."""
        self.rate_limiter.throttle()


class OpenAIModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-5").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to forward to the underlying OpenAI API completion call, for instance `temperature`.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # --- BEGIN add logging ---
        if self.logs_dir:
            self._call_counter[True] += 1
            call_num = (
                get_next_call_number(self.logs_dir)
                if self._call_counter[True] == 1
                else self._call_counter[True]
            )
            request_data = {
                "model": completion_kwargs.get("model"),
                "messages": completion_kwargs.get("messages"),
                "kwargs": {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ("messages", "model")
                },
            }
            deltas_collected = []
        # --- END add logging ---

        self._apply_rate_limit()
        try:
            for event in self.retryer(
                self.client.chat.completions.create,
                **completion_kwargs,
                stream=True,
                stream_options={"include_usage": True},
            ):
                delta = None
                if event.usage:
                    delta = ChatMessageStreamDelta(
                        content="",
                        token_usage=TokenUsage(
                            input_tokens=event.usage.prompt_tokens,
                            output_tokens=event.usage.completion_tokens,
                        ),
                    )
                    if self.logs_dir:
                        deltas_collected.append(delta)
                    yield delta
                if event.choices:
                    choice = event.choices[0]
                    if choice.delta:
                        delta = ChatMessageStreamDelta(
                            content=choice.delta.content,
                            tool_calls=[
                                ChatMessageToolCallStreamDelta(
                                    index=delta_item.index,
                                    id=delta_item.id,
                                    type=delta_item.type,
                                    function=delta_item.function,
                                )
                                for delta_item in choice.delta.tool_calls
                            ]
                            if choice.delta.tool_calls
                            else None,
                        )
                        if self.logs_dir:
                            deltas_collected.append(delta)
                        yield delta
                    else:
                        if not getattr(choice, "finish_reason", None):
                            raise ValueError(
                                f"No content or tool calls in event: {event}"
                            )
        finally:
            if self.logs_dir:
                save_llm_call(
                    logs_dir=self.logs_dir,
                    call_number=call_num,
                    is_stream=True,
                    request_data=request_data,
                    stream_deltas=deltas_collected,
                )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if self.verbose:
            print(
                f"[OpenAIModel] Generating non-stream response for model {self.model_id}"
            )

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # --- BEGIN add logging ---
        if self.logs_dir:
            self._call_counter[False] += 1
            call_num = (
                get_next_call_number(self.logs_dir)
                if self._call_counter[False] == 1
                else self._call_counter[False]
            )
            request_data = {
                "model": completion_kwargs.get("model"),
                "messages": completion_kwargs.get("messages"),
                "kwargs": {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ("messages", "model")
                },
            }
        # --- END add logging ---

        self._apply_rate_limit()
        response = self.retryer(
            self.client.chat.completions.create, **completion_kwargs
        )

        if self.logs_dir:
            save_llm_call(
                logs_dir=self.logs_dir,
                call_number=call_num,
                is_stream=False,
                request_data=request_data,
                response_data=response,
            )

        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)
        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


def is_rate_limit_error(exception: BaseException) -> bool:
    """Check if the exception is a rate limit error."""
    error_str = str(exception).lower()
    return (
        "429" in error_str
        or "rate limit" in error_str
        or "too many requests" in error_str
        or "rate_limit" in error_str
    )


OpenAIServerModel = OpenAIModel


# Model Registry for secure deserialization
# This registry maps model class names to their actual classes.
# Only classes listed here can be instantiated during deserialization (from_dict).
# This prevents arbitrary code execution via importlib-based dynamic loading.
MODEL_REGISTRY = {
    "OpenAIModel": OpenAIModel,
}

__all__ = [
    "REMOVE_PARAMETER",
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "ApiModel",
    "OpenAIServerModel",
    "OpenAIModel",
    "ChatMessage",
]
