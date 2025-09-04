import logging

from typing import Dict, List, Optional, Union, Literal, Iterator

from jet.llm.mlx.remote.client import MLXRemoteClient
from jet.llm.mlx.remote.types import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    ModelsResponse,
    HealthResponse,
)
from jet.llm.mlx.chat_history import ChatHistory


def get_models(
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    repo_id: Optional[str] = None,
    verbose: bool = False,
) -> ModelsResponse:
    """Retrieve available models from the remote MLX server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    return client.list_models(repo_id=repo_id)


def health_check(
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
) -> HealthResponse:
    """Health check for the remote MLX server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    return client.health_check()


def prepare_messages(
    messages: Union[str, List[Message]],
    history: ChatHistory,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
) -> List[Message]:
    """Prepare messages with optional history and system prompt, ensuring tool_calls are included."""
    if system_prompt and not any(m["role"] == "system" for m in history.get_messages()):
        if with_history:
            history.add_message("system", system_prompt)

    if isinstance(messages, str):
        if with_history:
            history.add_message("user", messages)
        all_messages: List[Message] = (
            history.get_messages()
            if with_history
            else [{"role": "user", "content": messages, "tool_calls": []}]
        )
    elif isinstance(messages, list):
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must include 'role' and 'content'.")
            if with_history:
                # Include tool_calls if present, default to empty list if not
                tool_calls = msg.get("tool_calls", [])
                history.add_message(
                    msg["role"],
                    msg["content"],
                    tool_calls=tool_calls if isinstance(
                        tool_calls, list) else []
                )
        all_messages = history.get_messages() if with_history else messages
    else:
        raise TypeError("messages must be a string or a list of Message dicts")

    # Convert messages to match Message TypedDict, including only role, content, tool_calls
    formatted_messages: List[Message] = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "tool_calls": msg.get("tool_calls", [])
        }
        for msg in all_messages
    ]

    if system_prompt and not with_history:
        formatted_messages = [
            {"role": "system", "content": system_prompt, "tool_calls": []}
        ] + formatted_messages
    return formatted_messages


def chat(
    messages: Union[str, List[Message]],
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[List[Dict[str, str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
) -> ChatCompletionResponse:
    """Create a chat completion via the remote server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    hist = history or ChatHistory()
    request_messages = prepare_messages(
        messages, hist, system_prompt, with_history)
    req: ChatCompletionRequest = {
        "messages": request_messages,
        "model": model,
        "draft_model": draft_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "repetition_context_size": repetition_context_size,
        "xtc_probability": xtc_probability,
        "xtc_threshold": xtc_threshold,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "seed": seed,
        "stop": stop,
        "stream": False,
        "stream_options": None,
        "role_mapping": role_mapping,
        "tools": tools,
    }
    req = {k: v for k, v in req.items() if v is not None}
    response = list(client.create_chat_completion(req, stream=False))[0]
    new_choices = []
    for choice in response["choices"]:
        if isinstance(choice, dict) and "delta" in choice:
            delta = choice.get("delta", {})
            new_choice = {k: v for k, v in choice.items() if k != "delta"}
            if delta:
                if "message" not in new_choice:
                    new_choice["message"] = {}
                for k, v in delta.items():
                    new_choice["message"][k] = v
            if "message" in new_choice:
                new_choice["message"] = {
                    k: v for k, v in new_choice["message"].items()
                    if k in ["role", "content", "tool_calls"]
                }
            new_choices.append(new_choice)
        else:
            if isinstance(choice, dict) and "message" in choice:
                choice["message"] = {
                    k: v for k, v in choice["message"].items()
                    if k in ["role", "content", "tool_calls"]
                }
            new_choices.append(choice)
    response["choices"] = new_choices
    return response


def stream_chat(
    messages: Union[str, List[Message]],
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    role_mapping: Optional[Dict[str, str]] = None,
    tools: Optional[List[Dict[str, str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
) -> Iterator[ChatCompletionResponse]:
    """Stream chat completion chunks via the remote server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    hist = history or ChatHistory()
    request_messages = prepare_messages(
        messages, hist, system_prompt, with_history)
    req: ChatCompletionRequest = {
        "messages": request_messages,
        "model": model,
        "draft_model": draft_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "repetition_context_size": repetition_context_size,
        "xtc_probability": xtc_probability,
        "xtc_threshold": xtc_threshold,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "seed": seed,
        "stop": stop,
        "stream": True,
        "stream_options": None,
        "role_mapping": role_mapping,
        "tools": tools,
    }
    req = {k: v for k, v in req.items() if v is not None}
    chunks = client.create_chat_completion(
        req, stream=True)
    for chunk in chunks:
        new_choices = []
        for choice in chunk["choices"]:
            if isinstance(choice, dict) and "delta" in choice:
                delta = choice.get("delta", {})
                new_choice = {k: v for k, v in choice.items() if k != "delta"}
                if delta:
                    if "message" not in new_choice:
                        new_choice["message"] = {}
                    for k, v in delta.items():
                        new_choice["message"][k] = v
                if "message" in new_choice:
                    new_choice["message"] = {
                        k: v for k, v in new_choice["message"].items()
                        if k in ["role", "content", "tool_calls"]
                    }
                new_choices.append(new_choice)
            else:
                if isinstance(choice, dict) and "message" in choice:
                    choice["message"] = {
                        k: v for k, v in choice["message"].items()
                        if k in ["role", "content", "tool_calls"]
                    }
                new_choices.append(choice)
        chunk["choices"] = new_choices
        yield chunk


def generate(
    prompt: str,
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> TextCompletionResponse:
    """Create a text completion via the remote server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    req: TextCompletionRequest = {
        "prompt": prompt,
        "model": model,
        "draft_model": draft_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "repetition_context_size": repetition_context_size,
        "xtc_probability": xtc_probability,
        "xtc_threshold": xtc_threshold,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "seed": seed,
        "stop": stop,
        "stream": False,
        "stream_options": None,
    }
    req = {k: v for k, v in req.items() if v is not None}
    return list(client.create_text_completion(req, stream=False))[0]


def stream_generate(
    prompt: str,
    model: Optional[str] = None,
    draft_model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    xtc_probability: Optional[float] = None,
    xtc_threshold: Optional[float] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    logprobs: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[TextCompletionResponse]:
    """Stream text completion chunks via the remote server."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    req: TextCompletionRequest = {
        "prompt": prompt,
        "model": model,
        "draft_model": draft_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "repetition_context_size": repetition_context_size,
        "xtc_probability": xtc_probability,
        "xtc_threshold": xtc_threshold,
        "logit_bias": logit_bias,
        "logprobs": logprobs,
        "seed": seed,
        "stop": stop,
        "stream": True,
        "stream_options": None,
    }
    req = {k: v for k, v in req.items() if v is not None}
    chunks = client.create_text_completion(
        req, stream=True)
    for chunk in chunks:
        yield chunk


__all__ = [
    "prepare_messages",
    "get_models",
    "health_check",
    "chat",
    "stream_chat",
    "generate",
    "stream_generate",
]
