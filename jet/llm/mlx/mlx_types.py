from typing import Literal, Union, Dict, List, Optional, TypedDict, Any, get_args
from enum import Enum
from jet.llm.mlx.helpers.detect_repetition import NgramRepeat
from transformers import PreTrainedTokenizer
from jet.models.model_types import LLMModelType
from mlx_lm.tokenizer_utils import TokenizerWrapper

# Type definitions

MLXTokenizer = Union[TokenizerWrapper, PreTrainedTokenizer]


class Message(TypedDict):
    role: str
    content: str


class Delta(TypedDict):
    role: Optional[str]
    content: Optional[str]


class Tool(TypedDict):
    type: str
    function: Dict[str, Any]


class RoleMapping(TypedDict, total=False):
    system_prompt: str
    system: str
    user: str
    assistant: str
    stop: str


class Logprobs(TypedDict):
    token_logprobs: List[float]
    top_logprobs: List[Dict[int, float]]
    tokens: List[int]


class Choice(TypedDict):
    index: int
    logprobs: Logprobs
    finish_reason: Optional[Literal["length", "stop"]]
    message: Optional[Message]
    delta: Optional[Delta]
    text: Optional[str]


class Usage(TypedDict):
    prompt_tokens: int
    prompt_tps: float
    completion_tokens: int
    completion_tps: float
    total_tokens: int
    peak_memory: float


class ModelInfo(TypedDict):
    id: str
    object: str
    created: int


class ModelsResponse(TypedDict):
    object: str
    data: List[ModelInfo]


class CompletionResponse(TypedDict):
    id: str
    system_fingerprint: str
    object: str
    model: LLMModelType
    created: int
    usage: Optional[Usage]
    content: str
    repetitions: Optional[List[NgramRepeat]]
    choices: List[Choice]
