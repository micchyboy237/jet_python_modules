# jet_python_modules/jet/libs/llama_cpp/usage/chat_stream.py
"""Pure LLM Streaming Engine — No Observability Dependencies.

Provides sync/async chat and text completion streaming with tool-use loops,
vision input, and structured output parsing. Completely free of OpenTelemetry,
Phoenix, or any tracing side effects.

Use this module directly when you need raw streaming without observability overhead.
For traced execution, use chat_stream_observability.py instead.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import inspect
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable

import httpx
import requests
from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_types import (
    StreamCompletionResult,
    ToolCallResult,
)
from jet.libs.llama_cpp.usage.structured_output import (
    OutputFormat,
    parse_structured_content,
    resolve_response_format,
)
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = "qwen3.5-uncensored:2b"
MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", DEFAULT_MODEL)


def encode_image_to_base64(image_source: str | Path | bytes) -> tuple[str, str]:
    """Encode a local file, remote URL, or raw bytes to base64 for vision API."""
    if isinstance(image_source, (str, Path)):
        source = str(image_source)
        if source.startswith(("http://", "https://")):
            try:
                resp = requests.get(source, timeout=30)
                resp.raise_for_status()
                img_bytes = resp.content
                content_type = (
                    resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                )
                valid_mimes = {"image/png", "image/jpeg", "image/webp", "image/gif"}
                mime = content_type if content_type in valid_mimes else "image/jpeg"
            except RequestException as exc:
                raise ValueError(f"Failed to fetch image from {source}: {exc}") from exc
        else:
            path = Path(source).expanduser()
            img_bytes = path.read_bytes()
            suffix = path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
    elif isinstance(image_source, bytes):
        img_bytes = image_source
        mime = "image/jpeg"
    else:
        raise ValueError("image_source must be str/Path (local/remote) or bytes")
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return base64_data, mime


async def encode_image_to_base64_async(
    image_source: str | Path | bytes,
) -> tuple[str, str]:
    """Async version of encode_image_to_base64 using httpx."""
    if isinstance(image_source, (str, Path)):
        source = str(image_source)
        if source.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(source)
                    response.raise_for_status()
                    img_bytes = response.content
                    content_type = (
                        response.headers.get("Content-Type", "")
                        .split(";")[0]
                        .strip()
                        .lower()
                    )
                    valid_mimes = {"image/png", "image/jpeg", "image/webp", "image/gif"}
                    mime = content_type if content_type in valid_mimes else "image/jpeg"
            except httpx.HTTPError as exc:
                raise ValueError(f"Failed to fetch image from {source}: {exc}") from exc
        else:
            path = Path(source).expanduser()
            img_bytes = path.read_bytes()
            suffix = path.suffix.lower()
            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/jpeg")
    elif isinstance(image_source, bytes):
        img_bytes = image_source
        mime = "image/jpeg"
    else:
        raise ValueError("image_source must be str/Path (local/remote) or bytes")
    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return base64_data, mime


def execute_tool(
    tool_name: str,
    tool_arguments: dict[str, Any] | str,
    executor: Callable[..., Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Execute a tool function synchronously with error handling."""
    if isinstance(tool_arguments, str):
        try:
            tool_arguments = json.loads(tool_arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            if strict:
                raise
            return {"error": f"Invalid JSON arguments: {e}", "tool": tool_name}
    try:
        result = executor(**tool_arguments)
        return result
    except TypeError as exc:
        error_msg = f"Argument mismatch: {exc}"
        logger.warning(f"Tool '{tool_name}' argument error: {exc}")
        if strict:
            raise
        return {"error": error_msg, "tool": tool_name}
    except Exception as exc:
        logger.exception(f"Tool '{tool_name}' failed")
        if strict:
            raise
        return {"error": str(exc), "tool": tool_name}


async def execute_tool_async(
    tool_name: str,
    tool_arguments: dict[str, Any] | str,
    executor: Callable[..., Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Execute a tool function asynchronously (handles sync and async executors)."""
    if isinstance(tool_arguments, str):
        try:
            tool_arguments = json.loads(tool_arguments)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
            if strict:
                raise
            return {"error": f"Invalid JSON arguments: {e}", "tool": tool_name}
    try:
        if inspect.iscoroutinefunction(executor):
            result = await executor(**tool_arguments)
        else:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: executor(**tool_arguments)
            )
        return result
    except TypeError as exc:
        error_msg = f"Argument mismatch: {exc}"
        logger.warning(f"Tool '{tool_name}' argument error: {exc}")
        if strict:
            raise
        return {"error": error_msg, "tool": tool_name}
    except Exception as exc:
        logger.exception(f"Tool '{tool_name}' failed")
        if strict:
            raise
        return {"error": str(exc), "tool": tool_name}


def _build_messages(
    prompt: str | None,
    messages: list[dict[str, Any]] | None,
    image_source: str | None,
    system_prompt_addition: str | None,
    image_encoder: Callable,
) -> list[dict[str, Any]]:
    """Build the initial messages list from prompt/messages/image inputs."""
    current_messages: list[dict[str, Any]] | None = messages
    if current_messages is None:
        if image_source and prompt:
            base64_img, mime_type = image_encoder(image_source)
            content: Any = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                },
            ]
            current_messages = [{"role": "user", "content": content}]
        elif prompt:
            current_messages = [{"role": "user", "content": prompt}]
        else:
            current_messages = []
    if system_prompt_addition and current_messages:
        existing_system_idx = next(
            (
                i
                for i, msg in enumerate(current_messages)
                if msg.get("role") == "system"
            ),
            None,
        )
        if existing_system_idx is not None:
            existing_content = current_messages[existing_system_idx].get("content", "")
            current_messages[existing_system_idx]["content"] = (
                f"{existing_content}\n{system_prompt_addition}"
            )
        else:
            current_messages.insert(
                0, {"role": "system", "content": system_prompt_addition}
            )
    return current_messages


def _parse_tool_calls_from_accumulator(
    tool_calls_acc: dict[int, dict[str, Any]],
) -> list[ToolCallResult]:
    """Convert accumulated streaming tool call deltas into parsed ToolCallResult list."""
    parsed: list[ToolCallResult] = []
    for idx in sorted(tool_calls_acc):
        tc = tool_calls_acc[idx]
        fn = tc["function"]
        try:
            parsed_args = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            parsed_args = {}
        parsed.append(
            ToolCallResult(
                id=tc.get("id", ""),
                type=tc.get("type", "function"),
                name=fn.get("name", ""),
                arguments=parsed_args,
                raw_arguments=fn.get("arguments", ""),
            )
        )
    return parsed


def make_console_chat_printer() -> tuple[
    Callable[[ChatCompletionChunk], None], dict[str, Any]
]:
    """Create a plain-stdout on_chunk callback for live chat streaming output.

    Mirrors chat_stream_observability's `_make_chat_chunk_handler`, minus
    the rich/OTel dependency, so `python chat_stream.py "..."` streams
    tokens live the same way the observability variant does.

    Returns:
        Tuple of (callback_fn, state_dict). state_dict tracks
        "first_token_at" (perf_counter timestamp of first token, or None)
        and "in_think_block" (bool) for post-stream metrics/cleanup.
    """
    state: dict[str, Any] = {"first_token_at": None, "in_think_block": False}

    def on_chunk(chunk: ChatCompletionChunk) -> None:
        if not chunk.choices:
            return
        delta = chunk.choices[0].delta
        if not delta:
            return
        if state["first_token_at"] is None and (
            getattr(delta, "content", None)
            or getattr(delta, "reasoning_content", None)
            or getattr(delta, "tool_calls", None)
        ):
            state["first_token_at"] = time.perf_counter()
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if not state["in_think_block"]:
                print("<think>", end="", flush=True)
                state["in_think_block"] = True
            print(delta.reasoning_content, end="", flush=True)
        elif state["in_think_block"]:
            print("</think>", end="", flush=True)
            state["in_think_block"] = False
        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)

    return on_chunk, state


def make_console_generate_printer() -> tuple[Callable[[Any], None], dict[str, Any]]:
    """Create a plain-stdout on_chunk callback for raw text-completion streaming.

    Mirrors chat_stream_observability's `_make_generate_chunk_handler`.

    Returns:
        Tuple of (callback_fn, state_dict). state_dict tracks
        "first_token_at" (perf_counter timestamp of first token, or None).
    """
    state: dict[str, Any] = {"first_token_at": None}

    def on_chunk(chunk: Any) -> None:
        if not chunk.choices:
            return
        delta = chunk.choices[0].text
        if delta:
            if state["first_token_at"] is None:
                state["first_token_at"] = time.perf_counter()
            print(delta, end="", flush=True)

    return on_chunk, state


def print_stream_summary(
    result: StreamCompletionResult,
    total_secs: float,
    first_token_at: float | None,
    t_start: float,
) -> None:
    """Print a 📊 Summary block, matching chat_stream_observability's shape
    (tokens, throughput, duration, TTFT, response length, finish reason)
    but via plain print() — no rich/OTel dependency.
    """
    ttft = (first_token_at - t_start) if first_token_at is not None else None
    print("─" * 60)
    print("📊 Summary")
    if result.usage:
        tok_per_sec = (
            result.usage.get("completion_tokens", 0) / total_secs
            if total_secs > 0
            else 0.0
        )
        print(
            f"   Tokens           : {result.usage.get('prompt_tokens', 0)}p / "
            f"{result.usage.get('completion_tokens', 0)}c / "
            f"{result.usage.get('total_tokens', 0)}t"
        )
        print(f"   Throughput       : {tok_per_sec:.1f} tok/s")
    print(f"   Duration         : {total_secs:.2f}s")
    if ttft is not None:
        print(f"   Time to first token: {ttft:.2f}s")
    print(f"   Response length  : {len(result.content)} chars")
    if result.finish_reason:
        print(f"   Finish reason    : {result.finish_reason}")
    if result.has_tool_calls:
        print(f"   Tool calls       : {len(result.tool_calls)}")
    if result.structured:
        status = "✅" if result.structured.success else "⚠️"
        print(f"   Structured       : {status} {result.structured.format_used.value}")
    print("─" * 60)


def run_chat_stream(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    image_source: str | None = None,
    client: OpenAI | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_registry: dict[str, Callable[..., Any]] | None = None,
    response_format: Any = None,
    max_tool_rounds: int = 10,
    extra_body_params: dict[str, Any] | None = None,
    on_chunk: Callable[[ChatCompletionChunk], None] | None = None,
) -> StreamCompletionResult:
    """Pure synchronous chat streaming with tool loops and structured output.

    Args:
        on_chunk: Optional callback invoked for each streamed chunk. Used by
            the observability wrapper for real-time console flushing without
            coupling this module to rich/logging. Also usable directly via
            make_console_chat_printer() for plain-stdout streaming.
    """
    resolved_fmt = resolve_response_format(response_format)
    api_response_format = resolved_fmt.api_format
    if resolved_fmt.output_format == OutputFormat.GRAMMAR:
        grammar_str = (api_response_format or {}).get("_grammar", "")
        if grammar_str:
            if extra_body_params is None:
                extra_body_params = {}
            extra_body_params["grammar"] = grammar_str
        api_response_format = None
    if client is None:
        client = get_llm_client()
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
    else:
        messages = prompt_or_messages
    current_messages = _build_messages(
        prompt,
        messages,
        image_source,
        resolved_fmt.system_prompt_addition,
        encode_image_to_base64,
    )
    is_agentic = tool_registry is not None
    last_result: StreamCompletionResult | None = None
    round_num = 0
    while round_num < max_tool_rounds:
        round_num += 1
        extra_body: dict[str, Any] = {
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if min_p > 0.0:
            extra_body["min_p"] = min_p
        if repeat_penalty != 1.1:
            extra_body["repeat_penalty"] = repeat_penalty
        if extra_body_params:
            extra_body.update(extra_body_params)
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": current_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "seed": seed,
            "stop": stop,
            "extra_body": extra_body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            api_kwargs["tools"] = tools
        if tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice
        if api_response_format:
            api_kwargs["response_format"] = api_response_format
        collected_content: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = None
        finish_reason: str | None = None
        stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
            **api_kwargs
        )
        in_think_block = False
        for chunk in stream:
            if on_chunk is not None:
                on_chunk(chunk)
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                in_think_block = True
                collected_content.append(delta.reasoning_content)
            elif in_think_block:
                in_think_block = False
            if hasattr(delta, "content") and delta.content:
                collected_content.append(delta.content)
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "type": tc_delta.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )
        full_response = "".join(collected_content)
        parsed_tool_calls = _parse_tool_calls_from_accumulator(tool_calls_acc)
        last_result = StreamCompletionResult(
            content=full_response,
            tool_calls=parsed_tool_calls,
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else None,
            finish_reason=finish_reason,
        )
        if resolved_fmt.output_format != OutputFormat.TEXT and not parsed_tool_calls:
            last_result.structured = parse_structured_content(
                full_response, resolved_fmt
            )
        if not last_result.has_tool_calls:
            break
        if not is_agentic:
            break
        assistant_tc_message: dict[str, Any] = {
            "role": "assistant",
            "content": last_result.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.name, "arguments": tc.raw_arguments},
                }
                for tc in last_result.tool_calls
            ],
        }
        current_messages.append(assistant_tc_message)
        for tc in last_result.tool_calls:
            executor = tool_registry.get(tc.name)
            if executor is None:
                tool_result: Any = {
                    "error": f"Unknown tool: {tc.name}",
                    "available_tools": list(tool_registry.keys()),
                }
            else:
                tool_result = execute_tool(
                    tc.name, tc.arguments, executor, strict=False
                )
            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                }
            )
    return last_result or StreamCompletionResult(
        content="", finish_reason="no_response"
    )


def run_generate_stream(
    prompt: str,
    model: str = MODEL,
    *,
    client: OpenAI | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra_body_params: dict[str, Any] | None = None,
    on_chunk: Callable[[Any], None] | None = None,
) -> StreamCompletionResult:
    """Pure synchronous raw text completion streaming.

    Args:
        on_chunk: Optional callback invoked for each streamed chunk. Also
            usable directly via make_console_generate_printer() for
            plain-stdout streaming.
    """
    if client is None:
        client = get_llm_client()
    extra_body: dict[str, Any] = {"top_k": top_k}
    if min_p > 0.0:
        extra_body["min_p"] = min_p
    if repeat_penalty != 1.1:
        extra_body["repeat_penalty"] = repeat_penalty
    if extra_body_params:
        extra_body.update(extra_body_params)
    api_kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "seed": seed,
        "stop": stop,
        "extra_body": extra_body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    collected_content: list[str] = []
    usage = None
    finish_reason: str | None = None
    stream = client.completions.create(**api_kwargs)
    for chunk in stream:
        if on_chunk is not None:
            on_chunk(chunk)
        if not chunk.choices:
            usage = getattr(chunk, "usage", None)
            continue
        delta = chunk.choices[0].text
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta:
            collected_content.append(delta)
    full_response = "".join(collected_content)
    return StreamCompletionResult(
        content=full_response,
        tool_calls=[],
        usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage
        else None,
        finish_reason=finish_reason,
    )


async def run_chat_stream_async(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    image_source: str | None = None,
    client: AsyncOpenAI | None = None,
    enable_thinking: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_registry: dict[str, Callable[..., Any]] | None = None,
    response_format: Any = None,
    max_tool_rounds: int = 10,
    extra_body_params: dict[str, Any] | None = None,
    on_chunk: Callable[[ChatCompletionChunk], None] | None = None,
) -> StreamCompletionResult:
    """Pure asynchronous chat streaming with tool loops and structured output.

    Args:
        on_chunk: Optional callback invoked for each streamed chunk. Used by
            the observability wrapper for real-time console flushing.

    Note:
        Stream and client lifecycle are managed internally by the OpenAI SDK.
        Do NOT call stream.aclose() or client.close() manually — doing so
        conflicts with httpcore2's safe_async_iterate context manager and
        causes RuntimeError: generator didn't stop after athrow().
    """
    resolved_fmt = resolve_response_format(response_format)
    api_response_format = resolved_fmt.api_format
    if resolved_fmt.output_format == OutputFormat.GRAMMAR:
        grammar_str = (api_response_format or {}).get("_grammar", "")
        if grammar_str:
            if extra_body_params is None:
                extra_body_params = {}
            extra_body_params["grammar"] = grammar_str
        api_response_format = None
    if client is None:
        client = get_async_llm_client()
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
    else:
        messages = prompt_or_messages
    encoded_image = None
    if image_source:
        encoded_image = await encode_image_to_base64_async(image_source)
    current_messages = _build_messages(
        prompt,
        messages,
        image_source,
        resolved_fmt.system_prompt_addition,
        lambda src: encoded_image if encoded_image else ("", "image/jpeg"),
    )
    is_agentic = tool_registry is not None
    last_result: StreamCompletionResult | None = None
    round_num = 0
    while round_num < max_tool_rounds:
        round_num += 1
        extra_body: dict[str, Any] = {
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if min_p > 0.0:
            extra_body["min_p"] = min_p
        if repeat_penalty != 1.1:
            extra_body["repeat_penalty"] = repeat_penalty
        if extra_body_params:
            extra_body.update(extra_body_params)
        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": current_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "seed": seed,
            "stop": stop,
            "extra_body": extra_body,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            api_kwargs["tools"] = tools
        if tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice
        if api_response_format:
            api_kwargs["response_format"] = api_response_format
        collected_content: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = None
        finish_reason: str | None = None
        stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            **api_kwargs
        )
        in_think_block = False
        async for chunk in stream:
            if on_chunk is not None:
                on_chunk(chunk)
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                in_think_block = True
                collected_content.append(delta.reasoning_content)
            elif in_think_block:
                in_think_block = False
            if hasattr(delta, "content") and delta.content:
                collected_content.append(delta.content)
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "type": tc_delta.type or "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["function"]["name"] += (
                                tc_delta.function.name
                            )
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )
        full_response = "".join(collected_content)
        parsed_tool_calls = _parse_tool_calls_from_accumulator(tool_calls_acc)
        last_result = StreamCompletionResult(
            content=full_response,
            tool_calls=parsed_tool_calls,
            usage={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage
            else None,
            finish_reason=finish_reason,
        )
        if resolved_fmt.output_format != OutputFormat.TEXT and not parsed_tool_calls:
            last_result.structured = parse_structured_content(
                full_response, resolved_fmt
            )
        if not last_result.has_tool_calls:
            break
        if not is_agentic:
            break
        assistant_tc_message: dict[str, Any] = {
            "role": "assistant",
            "content": last_result.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.name, "arguments": tc.raw_arguments},
                }
                for tc in last_result.tool_calls
            ],
        }
        current_messages.append(assistant_tc_message)
        for tc in last_result.tool_calls:
            executor = tool_registry.get(tc.name)
            if executor is None:
                tool_result: Any = {
                    "error": f"Unknown tool: {tc.name}",
                    "available_tools": list(tool_registry.keys()),
                }
            else:
                tool_result = await execute_tool_async(
                    tc.name, tc.arguments, executor, strict=False
                )
            current_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                }
            )
    return last_result or StreamCompletionResult(
        content="", finish_reason="no_response"
    )


async def run_generate_stream_async(
    prompt: str,
    model: str = MODEL,
    *,
    client: AsyncOpenAI | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repeat_penalty: float = 1.1,
    presence_penalty: float = 1.5,
    frequency_penalty: float = 0.0,
    logit_bias: dict[str, int] | None = None,
    seed: int | None = None,
    stop: list[str] | None = None,
    extra_body_params: dict[str, Any] | None = None,
    on_chunk: Callable[[Any], None] | None = None,
) -> StreamCompletionResult:
    """Pure asynchronous raw text completion streaming.

    Args:
        on_chunk: Optional callback invoked for each streamed chunk.

    Note:
        Stream and client lifecycle are managed internally by the OpenAI SDK.
        Do NOT call stream.aclose() or client.close() manually.
    """
    if client is None:
        client = get_async_llm_client()
    extra_body: dict[str, Any] = {"top_k": top_k}
    if min_p > 0.0:
        extra_body["min_p"] = min_p
    if repeat_penalty != 1.1:
        extra_body["repeat_penalty"] = repeat_penalty
    if extra_body_params:
        extra_body.update(extra_body_params)
    api_kwargs: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "seed": seed,
        "stop": stop,
        "extra_body": extra_body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    collected_content: list[str] = []
    usage = None
    finish_reason: str | None = None
    stream = await client.completions.create(**api_kwargs)
    async for chunk in stream:
        if on_chunk is not None:
            on_chunk(chunk)
        if not chunk.choices:
            usage = getattr(chunk, "usage", None)
            continue
        delta = chunk.choices[0].text
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if delta:
            collected_content.append(delta)
    full_response = "".join(collected_content)
    return StreamCompletionResult(
        content=full_response,
        tool_calls=[],
        usage={
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage
        else None,
        finish_reason=finish_reason,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure LLM streaming engine (no observability)."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="What is OpenTelemetry in one sentence?",
        help="Prompt for the chat/image analysis or raw completion.",
    )
    parser.add_argument(
        "-i",
        "--image-source",
        type=str,
        default=None,
        help="Path or URL to an image to analyze. Omit for text-only. (chat mode only)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Use raw text completion (run_generate_stream) instead of chat mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to request (env: LLAMA_CPP_VISION_MODEL).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=LLAMA_CPP_BASE_URL,
        help="OpenAI-compatible server base URL (env: LLAMA_CPP_VISION_URL).",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stop", type=str, nargs="+", default=None)
    parser.add_argument(
        "--logit-bias",
        type=str,
        default=None,
        help="JSON dict of token_id:bias pairs, e.g. '{\"1234\": -100}'",
    )
    parser.add_argument(
        "--tools-json",
        type=str,
        default=None,
        help="JSON array of tool definitions for function calling. (chat mode only)",
    )
    parser.add_argument(
        "--tool-choice",
        type=str,
        default=None,
        help='"auto", "none", "required", or JSON object. (chat mode only)',
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default=None,
        help='JSON response format, e.g. \'{"type": "json_object"}\'. (chat mode only)',
    )
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = get_args()

    parsed_logit_bias: dict[str, int] | None = None
    if args.logit_bias:
        try:
            parsed_logit_bias = json.loads(args.logit_bias)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid logit_bias JSON: {e}", file=sys.stderr)
            raise SystemExit(1)

    client = get_llm_client(base_url=args.base_url, timeout=args.timeout)

    print("─" * 60)
    print(f"🤖 Model        : {args.model}")
    print(
        f"🎛️  Sampling     : temp={args.temperature} top_p={args.top_p} top_k={args.top_k}"
    )

    t_start = time.perf_counter()

    if args.generate:
        # ── Raw text completion mode ──────────────────────────────────
        on_chunk, chunk_state = make_console_generate_printer()
        print("Response: ", end="", flush=True)
        result = run_generate_stream(
            args.prompt,
            model=args.model,
            client=client,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            logit_bias=parsed_logit_bias,
            seed=args.seed,
            stop=args.stop,
            on_chunk=on_chunk,
        )
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(result, total_secs, chunk_state["first_token_at"], t_start)
        print(
            f"📋 Result: {len(result.content)} chars, "
            f"finish_reason={result.finish_reason}"
        )
    else:
        # ── Chat streaming mode ───────────────────────────────────────
        parsed_tools: list[dict[str, Any]] | None = None
        if args.tools_json:
            try:
                parsed_tools = json.loads(args.tools_json)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid tools JSON: {e}", file=sys.stderr)
                raise SystemExit(1)
        parsed_tool_choice: str | dict[str, Any] | None = args.tool_choice
        if parsed_tool_choice and parsed_tool_choice.startswith("{"):
            try:
                parsed_tool_choice = json.loads(parsed_tool_choice)
            except json.JSONDecodeError:
                pass
        parsed_response_format: dict[str, Any] | None = None
        if args.response_format:
            try:
                parsed_response_format = json.loads(args.response_format)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid response_format JSON: {e}", file=sys.stderr)
                raise SystemExit(1)

        on_chunk, chunk_state = make_console_chat_printer()
        print("Response: ", end="", flush=True)
        result = run_chat_stream(
            args.prompt,
            model=args.model,
            image_source=args.image_source,
            client=client,
            enable_thinking=args.enable_thinking,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            logit_bias=parsed_logit_bias,
            seed=args.seed,
            stop=args.stop,
            tools=parsed_tools,
            tool_choice=parsed_tool_choice,
            response_format=parsed_response_format,
            tool_registry=None,
            on_chunk=on_chunk,
        )
        if chunk_state["in_think_block"]:
            print("</think>", end="")
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(result, total_secs, chunk_state["first_token_at"], t_start)

        if result.has_tool_calls:
            print(
                f"📋 Result: {len(result.tool_calls)} tool call(s), "
                f"finish_reason={result.finish_reason}"
            )
        else:
            print(
                f"📋 Result: {len(result.content)} chars, "
                f"finish_reason={result.finish_reason}"
            )
