from typing import Callable, Dict, List, Optional, Union, Literal, Iterator, Any

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
from jet.llm.mlx.mlx_utils import has_tools
from jet.utils.inspect_utils import get_method_info
from jet.logger import logger


def prepare_messages(
    messages: Union[str, List[Message]],
    history: ChatHistory,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
) -> List[Message]:
    """Prepare messages with optional history and system prompt, ensuring tool_calls and alternating roles."""
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
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for call in tool_calls:
                    if not isinstance(call, dict) or "function" not in call or call.get("type") != "function":
                        raise ValueError(
                            "Invalid tool_call format: must include 'function' and 'type'='function'.")
            if with_history:
                history.add_message(
                    msg["role"],
                    msg["content"],
                    tool_calls=tool_calls if isinstance(
                        tool_calls, list) else []
                )
        all_messages = history.get_messages() if with_history else messages
    else:
        raise TypeError("messages must be a string or a list of Message dicts")
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
    non_system_messages = [
        m for m in formatted_messages if m["role"] != "system"]
    if non_system_messages:
        for i, msg in enumerate(non_system_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if msg["role"] != expected_role:
                logger.error(
                    f"Invalid role sequence at index {i}: expected '{expected_role}', got '{msg['role']}'"
                )
                raise ValueError(
                    "Conversation roles must alternate user/assistant/user/assistant after an optional system message."
                )
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
    tools: Optional[List[Callable]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
) -> ChatCompletionResponse:
    """Create a chat completion via the remote server with tool usage support."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    hist = history or ChatHistory()

    # Format tools for the request if model supports tools
    formatted_tools = []
    if tools and model and has_tools(model):
        if not all(callable(tool) for tool in tools):
            raise ValueError("All tools must be callable functions")
        formatted_tools = [
            {
                "type": "function",
                "function": get_method_info(tool)
            }
            for tool in tools
        ]

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
        "tools": formatted_tools if formatted_tools else None,
    }
    req = {k: v for k, v in req.items() if v is not None}
    response = list(client.create_chat_completion(req, stream=False))[0]
    assistant_content = []
    assistant_tool_calls = []
    if with_history and response.get("choices"):
        for choice in response["choices"]:
            if choice.get("message", {}).get("role") == "assistant":
                content = choice["message"].get("content", "")
                tool_calls = choice["message"].get("tool_calls", [])
                if content:
                    assistant_content.append(content)
                if tool_calls:
                    for call in tool_calls:
                        if isinstance(call, dict) and call.get("type") == "function":
                            assistant_tool_calls.append(call)
                hist.add_message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls
                )
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
                message = choice["message"]
                formatted_tool_calls = []
                if message.get("tool_calls"):
                    for call in message["tool_calls"]:
                        if isinstance(call, dict) and call.get("type") == "function":
                            formatted_tool_calls.append(call)
                new_message = {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                    "tool_calls": formatted_tool_calls
                }
                choice["message"] = new_message
            new_choices.append(choice)
    response["choices"] = new_choices
    response["content"] = "".join(
        assistant_content) if assistant_content else None
    response["tool_calls"] = assistant_tool_calls if assistant_tool_calls else None
    if with_history:
        response["history"] = hist.get_messages()
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
    tools: Optional[List[Callable]] = None,
    client: Optional[MLXRemoteClient] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    with_history: bool = False,
    history: Optional[ChatHistory] = None,
    verbose: bool = False,
) -> Iterator[ChatCompletionResponse]:
    """Stream chat completion chunks via the remote server with tool usage support."""
    if client is None:
        client = MLXRemoteClient(base_url=base_url, verbose=verbose)
    hist = history or ChatHistory()

    # Format tools for the request if model supports tools
    formatted_tools = []
    if tools and model and has_tools(model):
        if not all(callable(tool) for tool in tools):
            raise ValueError("All tools must be callable functions")
        formatted_tools = [
            {
                "type": "function",
                "function": get_method_info(tool)
            }
            for tool in tools
        ]

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
        "tools": formatted_tools if formatted_tools else None,
    }
    req = {k: v for k, v in req.items() if v is not None}
    chunks = client.create_chat_completion(req, stream=True)
    assistant_content = []
    assistant_tool_calls = []
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
                    formatted_tool_calls = []
                    if new_choice["message"].get("tool_calls"):
                        for call in new_choice["message"]["tool_calls"]:
                            if isinstance(call, dict) and call.get("type") == "function":
                                formatted_tool_calls.append(call)
                    new_message = {
                        "role": new_choice["message"].get("role", "assistant"),
                        "content": new_choice["message"].get("content", ""),
                        "tool_calls": formatted_tool_calls
                    }
                    new_choice["message"] = new_message
                    if new_choice["message"].get("role") == "assistant":
                        if new_choice["message"]["content"]:
                            assistant_content.append(
                                new_choice["message"]["content"])
                        if new_choice["message"]["tool_calls"]:
                            assistant_tool_calls.extend(formatted_tool_calls)
                new_choices.append(new_choice)
            else:
                if isinstance(choice, dict) and "message" in choice:
                    message = choice["message"]
                    formatted_tool_calls = []
                    if message.get("tool_calls"):
                        for call in message["tool_calls"]:
                            if isinstance(call, dict) and call.get("type") == "function":
                                formatted_tool_calls.append(call)
                    new_message = {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                        "tool_calls": formatted_tool_calls
                    }
                    choice["message"] = new_message
                    if choice["message"].get("role") == "assistant":
                        if choice["message"]["content"]:
                            assistant_content.append(
                                choice["message"]["content"])
                        if choice["message"]["tool_calls"]:
                            assistant_tool_calls.extend(formatted_tool_calls)
                new_choices.append(choice)
            if choice.get("finish_reason") is not None and with_history:
                if assistant_content or assistant_tool_calls:
                    hist.add_message(
                        role="assistant",
                        content="".join(
                            assistant_content) if assistant_content else "",
                        tool_calls=assistant_tool_calls if assistant_tool_calls else None
                    )
        chunk["choices"] = new_choices
        chunk["content"] = "".join(
            assistant_content) if assistant_content else None
        chunk["tool_calls"] = assistant_tool_calls if assistant_tool_calls else None
        if with_history:
            chunk["history"] = hist.get_messages()
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
    response = list(client.create_text_completion(req, stream=False))[0]
    # Collect content from choices
    text_content = []
    for choice in response["choices"]:
        if choice.get("text"):
            text_content.append(choice["text"])
    response["content"] = "".join(text_content) if text_content else None
    return response


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
    chunks = client.create_text_completion(req, stream=True)
    text_content = []
    for chunk in chunks:
        for choice in chunk["choices"]:
            if choice.get("text"):
                text_content.append(choice["text"])
            if choice.get("finish_reason") is not None:
                chunk["content"] = "".join(
                    text_content) if text_content else None
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
