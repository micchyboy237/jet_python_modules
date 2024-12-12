from enum import Enum
import json
import requests
from typing import Generator, Literal, Optional, TypedDict, Union
from jet.logger import logger


class ToolFunctionParameters(TypedDict):
    type: Literal["object"]
    properties: dict
    required: list[str]


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: ToolFunctionParameters


class Tool(TypedDict):
    type: Literal["function"]
    function: ToolFunction


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    images: Optional[list[str]]
    tool_calls: Optional[list[Tool]]


class OllamaChatOptions(TypedDict):
    mirostat: Optional[int]
    mirostat_eta: Optional[float]
    mirostat_tau: Optional[float]
    num_ctx: Optional[int]
    repeat_last_n: Optional[int]
    repeat_penalty: Optional[float]
    temperature: Optional[float]
    seed: Optional[int]
    stop: Optional[str]
    tfs_z: Optional[float]
    num_predict: Optional[int]
    top_k: Optional[int]
    top_p: Optional[float]
    min_p: Optional[float]


class OllamaChatRequest(TypedDict):
    model: str
    messages: list[Message]
    tools: Optional[list[Tool]]
    format: Optional[Union[str, dict]]  # Can be "json" or a JSON schema
    options: Optional[OllamaChatOptions]
    stream: Optional[bool]  # Defaults to True if not specified
    keep_alive: Optional[Union[int, str]]  # Defaults to "5m" if not specified


class OllamaChatResponseMessage(TypedDict):
    role: Literal["assistant", "system", "user", "tool"]
    content: str
    images: Optional[list[str]]
    tool_calls: Optional[list[dict]]


class OllamaChatResponse(TypedDict):
    model: str
    created_at: str
    message: OllamaChatResponseMessage
    done_reason: Optional[str]
    done: bool
    total_duration: Optional[int]
    load_duration: Optional[int]
    prompt_eval_count: Optional[int]
    prompt_eval_duration: Optional[int]
    eval_count: Optional[int]
    eval_duration: Optional[int]


class Track(TypedDict):
    repo: str
    run_hash: Optional[str]
    read_only: Optional[bool]
    experiment: Optional[str]
    force_resume: Optional[bool]
    system_tracking_interval: Optional[int]
    log_system_params: Optional[bool]
    # Custom
    run_name: Optional[str]
    metadata: Optional[dict]
    format: Optional[str]


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class OllamaChatMessage(TypedDict):
    content: str
    role: MessageRole
    images: Optional[list[str]]
    tool_calls: Optional[list[Tool]]
