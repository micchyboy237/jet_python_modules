from typing import Dict, List, Optional, TypedDict, Union, Literal


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Dict[str, str]]]


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
    stream_options: Optional[Dict[str, bool]]
    role_mapping: Optional[Dict[str, str]]
    tools: Optional[List[Dict[str, str]]]


class ChatChoice(TypedDict):
    index: int
    message: Dict[str, Union[str, List[Dict[str, str]]]]
    finish_reason: Optional[Literal["stop", "length"]]
    logprobs: Optional[Dict[str, Union[List[float],
                                       List[Dict[int, float]], List[int]]]]


class ChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatChoice]
    usage: Optional[Dict[str, int]]


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
    stream_options: Optional[Dict[str, bool]]


class TextChoice(TypedDict):
    index: int
    text: str
    finish_reason: Optional[Literal["stop", "length"]]
    logprobs: Optional[Dict[str, Union[List[float],
                                       List[Dict[int, float]], List[int]]]]


class TextCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[TextChoice]
    usage: Optional[Dict[str, int]]


class Model(TypedDict):
    id: str
    object: Literal["model"]
    created: int


class ModelsResponse(TypedDict):
    object: Literal["list"]
    data: List[Model]


class HealthResponse(TypedDict):
    status: Literal["ok"]
