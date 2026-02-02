# custom_models.py

import json
import logging
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS, LLAMACPP_LLM_TYPES
from jet.transformers.object import make_serializable
from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name
from smolagents.models import (
    REMOVE_PARAMETER,
    RETRY_EXPONENTIAL_BASE,
    RETRY_JITTER,
    RETRY_MAX_ATTEMPTS,
    RETRY_WAIT,
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCallStreamDelta,
    MessageRole,
    get_clean_message_list,
    get_tool_call_from_text,
    get_tool_json_schema,
    is_rate_limit_error,
    parse_json_if_needed,
    remove_content_after_stop_sequences,
    supports_stop_parameter,
    tool_role_conversions,
)
from smolagents.monitoring import TokenUsage
from smolagents.tools import Tool
from smolagents.utils import (
    RateLimiter,
    Retrying,
)

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = os.getenv("LLAMA_CPP_LLM_URL")
DEFAULT_MODEL_ID: LLAMACPP_LLM_TYPES = "qwen3-instruct-2507:4b"
DEFAULT_API_KEY = None


def get_next_call_number(logs_dir: Path) -> int:
    """Find the next available call number using 4-digit prefix (aligned with save_step_state)."""
    if not logs_dir.exists():
        return 1
    existing = [
        int(d.name.split("_")[0])
        for d in logs_dir.iterdir()
        if d.is_dir() and len(d.name) >= 5 and d.name[4] == "_" and d.name[:4].isdigit()
    ]
    return max(existing, default=0) + 1


def _get_llm_call_subdir(
    logs_dir: Path,
    call_number: int,
    is_stream: bool,
    agent_name: str | None = None,
) -> Path:
    # prefix = "generate_stream" if is_stream else "generate"
    prefix = f"{call_number:04d}"

    cleaned = "default"
    if agent_name:
        cleaned = (
            str(agent_name)
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
    elif hasattr(logs_dir, "name") and "_" in logs_dir.name:
        # fallback: try to reuse parent agent folder name if possible
        cleaned = logs_dir.name.split("_")[-1]

    subdir_name = f"{prefix}_{cleaned}"
    target_dir = logs_dir / subdir_name
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_request_llm_call(
    logs_dir: Path,
    call_number: int,
    is_stream: bool,
    request_data: dict,
    agent_name: str | None = None,
) -> None:
    target_dir = _get_llm_call_subdir(
        logs_dir,
        call_number,
        is_stream,
        agent_name,
    )
    (target_dir / "request.json").write_text(
        json.dumps(request_data, indent=2, ensure_ascii=False)
    )


def save_response_llm_call(
    logs_dir: Path,
    call_number: int,
    is_stream: bool,
    response_data: Any | None = None,
    stream_deltas: list | None = None,
    agent_name: str | None = None,
) -> None:
    target_dir = _get_llm_call_subdir(
        logs_dir,
        call_number,
        is_stream,
        agent_name,
    )

    if response_data is not None and not is_stream:
        if hasattr(response_data, "model_dump_json"):
            text = response_data.model_dump_json()
        elif hasattr(response_data, "json") and callable(response_data.json):
            text = response_data.json()
        else:
            text = response_data
        (target_dir / "response.json").write_text(
            json.dumps(make_serializable(text), indent=2, ensure_ascii=False)
        )

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
        verbose (`bool`, default `True`):
            Whether to enable verbose logging of model operations.
        logs_dir (`str`, optional):
            Directory for logging model call details. If None, a default location is used
            based on the entry script location.
        model_id (`str`, optional):
            Identifier for the specific model being used.
        agent_name (`str`, optional):
            Name of the agent using this model instance. Used to create agent-specific
            subdirectories in the logging folder (e.g. `.../my_research_agent/0001_generate_...`).
            If not provided, a generic/default subdirectory name is used.
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
        verbose: bool = True,
        logs_dir: str | None = None,
        model_id: LLAMACPP_KEYS | None = None,
        agent_name: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self.verbose = verbose
        self.agent_name = agent_name

        # cleaned_agent = None
        # if agent_name:
        #     cleaned_agent = (
        #         str(agent_name)
        #         .strip()
        #         .replace(" ", "_")
        #         .replace("-", "_")
        #         .replace(".", "_")
        #         .lower()
        #     )

        _caller_base_dir = (
            Path(get_entry_file_dir())
            / "generated"
            / Path(get_entry_file_name()).stem
            / "llm_calls"
        )
        self.logs_dir = Path(logs_dir).resolve() if logs_dir else _caller_base_dir

        # if cleaned_agent:
        #     self.logs_dir = self.logs_dir / cleaned_agent

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._call_counter: dict[bool, int] = {False: 0, True: 0}
        self.model_id: LLAMACPP_KEYS | None = model_id

    def _clean_agent_name(self, name: str) -> str:
        return str(name).strip().replace(" ", "_").replace("-", "_").lower()

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
    custom role mappings, rate limiting, retry logic, and API client connections.

    Parameters:
        model_id (`str`):
            The identifier for the model to be used with the API.
        custom_role_conversions (`dict[str, str]`, optional):
            Mapping to convert between internal role names and API-specific role names.
            Defaults to None.
        client (`Any`, optional):
            Pre-configured API client instance. If not provided, a default client
            will be created. Defaults to None.
        requests_per_minute (`float`, optional):
            Rate limit in requests per minute.
        retry (`bool`, default `True`):
            Whether to retry on rate limit errors, up to `RETRY_MAX_ATTEMPTS` times.
        agent_name (`str`, optional):
            Name of the agent using this model. Used to organize logs into agent-specific
            subfolders (e.g. `.../math_solver/0003_generate_stream_...`). Passed down from
            higher-level agent configuration.
        **kwargs:
            Additional keyword arguments forwarded to the underlying model completion call.
    """

    def __init__(
        self,
        model_id: LLAMACPP_KEYS,
        custom_role_conversions: dict[str, str] | None = None,
        client: Any | None = None,
        requests_per_minute: float | None = None,
        retry: bool = True,
        agent_name: str | None = None,  # ← forward
        **kwargs,
    ):
        super().__init__(
            model_id=model_id,
            agent_name=agent_name,  # ← pass down
            **kwargs,
        )
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
    """This model connects to an OpenAI-compatible API server (including local servers
    like llama.cpp server, vLLM, LM Studio, TabbyAPI, etc.).

    Parameters:
        model_id (`str`, default `"qwen3-instruct-2507:4b"`):
            The model identifier to use on the server (e.g. "gpt-4o", "llama-3.1-8b", ...).
        api_base (`str`, optional):
            The base URL of the OpenAI-compatible API server.
            Defaults to value of environment variable `LLAMA_CPP_LLM_URL` (if set).
        api_key (`str`, optional):
            The API key to use for authentication (often not needed for local servers).
        organization (`str`, optional):
            The organization to use for the API request (OpenAI-specific).
        project (`str`, optional):
            The project to use for the API request (OpenAI-specific).
        client_kwargs (`dict[str, Any]`, optional):
            Additional keyword arguments to pass to the OpenAI client constructor
            (e.g. `organization`, `project`, `max_retries`, `timeout`, ...).
        custom_role_conversions (`dict[str, str]`, optional):
            Custom mapping to convert message roles. Useful for models/servers that
            do not support "system" role or use different role names.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten structured message content into plain text before sending.
        agent_name (`str`, optional):
            Name of the agent using this model instance. If provided, LLM call logs
            are saved into agent-specific subdirectories (e.g. `0004_generate_my_agent/`).
            Helps separate logs when multiple agents run in the same process.
        **kwargs:
            Additional keyword arguments forwarded to every `chat.completions.create`
            call (e.g. `temperature`, `max_tokens`, `top_p`, ...).
    """

    def __init__(
        self,
        model_id: LLAMACPP_KEYS = DEFAULT_MODEL_ID,
        api_base: str | None = DEFAULT_API_BASE,
        api_key: str | None = DEFAULT_API_KEY,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        agent_name: str | None = None,  # ← accept here too
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
            agent_name=agent_name,  # ← forward to ApiModel → Model
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
            call_num = get_next_call_number(self.logs_dir)
            formatted_messages = []
            for msg in completion_kwargs.get("messages", []):
                role = (
                    msg["role"].value if hasattr(msg["role"], "value") else msg["role"]
                )

                content = msg["content"]
                if isinstance(content, str):
                    content_for_tokenizer = content
                elif (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], dict)
                ):
                    # extract text parts only (ignore images for token counting)
                    content_for_tokenizer = " ".join(
                        part["text"] for part in content if part.get("type") == "text"
                    )
                else:
                    content_for_tokenizer = str(content)  # fallback

                formatted_messages.append(
                    {"role": role, "content": content_for_tokenizer}
                )
            input_tokens = count_tokens(
                formatted_messages,
                model=self.model_id,
            )
            request_data = {
                "model": completion_kwargs.get("model"),
                "messages": completion_kwargs.get("messages"),
                "token_counts": {
                    "input_tokens": input_tokens,
                },
                "kwargs": {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ("messages", "model")
                },
            }
            save_request_llm_call(
                logs_dir=self.logs_dir,
                call_number=call_num,
                is_stream=False,
                request_data=request_data,
                agent_name=self.agent_name,
            )
        # --- END add logging ---

        self._apply_rate_limit()
        response = self.retryer(
            self.client.chat.completions.create, **completion_kwargs
        )

        if self.logs_dir:
            # Attach response input and output token count for easier debugging and analysis
            response.token_counts = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            save_response_llm_call(
                logs_dir=self.logs_dir,
                call_number=call_num,
                is_stream=False,
                response_data=response,
                agent_name=self.agent_name,
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
            call_num = get_next_call_number(self.logs_dir)
            formatted_messages = []
            for msg in completion_kwargs.get("messages", []):
                role = (
                    msg["role"].value if hasattr(msg["role"], "value") else msg["role"]
                )

                content = msg["content"]
                if isinstance(content, str):
                    content_for_tokenizer = content
                elif (
                    isinstance(content, list)
                    and content
                    and isinstance(content[0], dict)
                ):
                    # extract text parts only (ignore images for token counting)
                    content_for_tokenizer = " ".join(
                        part["text"] for part in content if part.get("type") == "text"
                    )
                else:
                    content_for_tokenizer = str(content)  # fallback

                formatted_messages.append(
                    {"role": role, "content": content_for_tokenizer}
                )
            input_tokens = count_tokens(
                formatted_messages,
                model=self.model_id,
            )
            request_data = {
                "model": completion_kwargs.get("model"),
                "messages": completion_kwargs.get("messages"),
                "token_counts": {
                    "input_tokens": input_tokens,
                },
                "kwargs": {
                    k: v
                    for k, v in completion_kwargs.items()
                    if k not in ("messages", "model")
                },
            }
            deltas_collected = []
            save_request_llm_call(
                logs_dir=self.logs_dir,
                call_number=call_num,
                is_stream=True,
                request_data=request_data,
                agent_name=self.agent_name,
            )
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
                save_response_llm_call(
                    logs_dir=self.logs_dir,
                    call_number=call_num,
                    is_stream=True,
                    stream_deltas=deltas_collected,
                    agent_name=self.agent_name,
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
