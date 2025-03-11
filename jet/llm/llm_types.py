import sys
import json
from enum import Enum
from typing import Any, Optional, TypedDict, Sequence, Literal, Mapping

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired
else:
    from typing import NotRequired


# Ollama LLM Types
class BaseGenerateResponse(TypedDict):
    model: str
    created_at: str
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int


class GenerateResponse(BaseGenerateResponse):
    response: str
    context: Sequence[int]


class ToolCallFunction(TypedDict):
    name: str
    arguments: NotRequired[Mapping[str, Any]]


class ToolCall(TypedDict):
    function: ToolCallFunction


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class Message(TypedDict):
    role: Literal['user', 'assistant', 'system', 'tool'] | MessageRole
    content: NotRequired[str]
    images: NotRequired[Sequence[Any]]
    tool_calls: NotRequired[Sequence[ToolCall]]


class Property(TypedDict):
    type: str
    description: str
    enum: NotRequired[Sequence[str]]


class Parameters(TypedDict):
    type: str
    required: Sequence[str]
    properties: Mapping[str, Property]


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: Parameters


class Tool(TypedDict):
    type: str
    function: ToolFunction


class OllamaChatResponse(BaseGenerateResponse):
    message: Message


class ProgressResponse(TypedDict):
    status: str
    completed: int
    total: int
    digest: str


class OllamaChatOptions(TypedDict, total=False):
    numa: bool
    num_ctx: int
    num_batch: int
    num_gpu: int
    main_gpu: int
    low_vram: bool
    f16_kv: bool
    logits_all: bool
    vocab_only: bool
    use_mmap: bool
    use_mlock: bool
    embedding_only: bool
    num_thread: int
    num_keep: int
    seed: int
    num_predict: int
    top_k: int
    top_p: float
    tfs_z: float
    typical_p: float
    repeat_last_n: int
    temperature: float
    repeat_penalty: float
    presence_penalty: float
    frequency_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float
    penalize_newline: bool
    stop: Sequence[str]


class ChatResponseInfo(OllamaChatResponse):
    options: NotRequired[OllamaChatOptions]

# Exception Definitions


class RequestError(Exception):
    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


class ResponseError(Exception):
    def __init__(self, error: str, status_code: int = -1):
        try:
            error = json.loads(error).get('error', error)
        except json.JSONDecodeError:
            pass
        super().__init__(error)
        self.error = error
        self.status_code = status_code


# Custom Types

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
