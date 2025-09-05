from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, Literal

from jet.llm.mlx.mlx_types import ToolCall


class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, str]]]
    tool_calls: Optional[List[ToolCall]]


class ChatCompletionRequest(TypedDict):
    messages: List[Message]
    model: Optional[str]
    draft_model: Optional[str]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[int]
    xtc_probability: Optional[float]
    xtc_threshold: Optional[float]
    logit_bias: Optional[Dict[int, float]]
    logprobs: Optional[int]
    seed: Optional[int]
    stop: Optional[Union[str, List[str]]]
    stream: Optional[bool]
    stream_options: Optional[Dict[str, bool]]
    role_mapping: Optional[Dict[str, str]]
    tools: Optional[List[Dict[str, Any]]]


class ChatChoice(TypedDict):
    index: int
    message: Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]
    finish_reason: Optional[Literal["stop", "length"]]
    logprobs: Optional[Dict[str, Union[List[float],
                                       List[List[Tuple[int, float]]], List[int]]]]


class ChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]]
    history: Optional[List[Dict]]
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


class TextCompletionRequest(TypedDict):
    prompt: str
    model: Optional[str]
    draft_model: Optional[str]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    min_p: Optional[float]
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[int]
    xtc_probability: Optional[float]
    xtc_threshold: Optional[float]
    logit_bias: Optional[Dict[int, float]]
    logprobs: Optional[int]
    seed: Optional[int]
    stop: Optional[Union[str, List[str]]]
    stream: Optional[bool]
    stream_options: Optional[Dict[str, bool]]


class TextChoice(TypedDict):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]]
    logprobs: Optional[Dict[str, Union[List[float],
                                       List[List[Tuple[int, float]]], List[int]]]]


class TextCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[TextChoice]
    usage: Optional[Dict[str, int]]
    content: Optional[str]


class Model(TypedDict):
    id: str
    object: Literal["model"]
    created: int


class ModelsResponse(TypedDict):
    object: Literal["list"]
    data: List[Model]


class HealthResponse(TypedDict):
    status: Literal["ok"]
