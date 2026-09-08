"""Shared utilities for LLM streaming engines.
Pure helpers with ZERO observability, rich, or logging side effects.
Used by both chat_stream.py (pure) and chat_stream_observability.py (traced).
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

import httpx
import requests
from jet.libs.llama_cpp.usage.chat_stream_types import ToolCallResult
from jet.libs.llama_cpp.usage.structured_output import (
    OutputFormat,
    ResolvedFormat,
    resolve_response_format,
)
from openai.types.chat import ChatCompletionChunk
from requests.exceptions import RequestException

LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = "qwen3.5-uncensored:2b"
MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def execute_tool(
    tool_name: str,
    tool_arguments: dict[str, Any] | str,
    executor: Callable[..., Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Execute a tool function synchronously with error handling."""
    import logging

    logger = logging.getLogger(__name__)

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
    import logging

    logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def build_messages(
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


# ---------------------------------------------------------------------------
# Tool call accumulation
# ---------------------------------------------------------------------------


def parse_tool_calls_from_accumulator(
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


def accumulate_tool_call_delta(
    tool_calls_acc: dict[int, dict[str, Any]],
    tc_delta: Any,
) -> None:
    """Merge a single streaming tool_call delta into the accumulator dict."""
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
            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
        if tc_delta.function.arguments:
            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments


# ---------------------------------------------------------------------------
# Response format → API kwargs preparation
# ---------------------------------------------------------------------------


def prepare_response_format(
    response_format: Any,
    extra_body_params: dict[str, Any] | None,
) -> tuple[ResolvedFormat, dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve response_format and route grammar into extra_body if needed.

    Returns:
        (resolved_fmt, api_response_format, updated_extra_body_params)
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
    return resolved_fmt, api_response_format, extra_body_params


# ---------------------------------------------------------------------------
# Extra body & API kwargs builders
# ---------------------------------------------------------------------------


def build_chat_extra_body(
    *,
    top_k: int,
    enable_thinking: bool,
    min_p: float,
    repeat_penalty: float,
    extra_body_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct the extra_body dict for chat completion requests."""
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
    return extra_body


def build_generate_extra_body(
    *,
    top_k: int,
    min_p: float,
    repeat_penalty: float,
    extra_body_params: dict[str, Any] | None,
) -> dict[str, Any]:
    """Construct the extra_body dict for text completion requests."""
    extra_body: dict[str, Any] = {"top_k": top_k}
    if min_p > 0.0:
        extra_body["min_p"] = min_p
    if repeat_penalty != 1.1:
        extra_body["repeat_penalty"] = repeat_penalty
    if extra_body_params:
        extra_body.update(extra_body_params)
    return extra_body


def build_chat_api_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    logit_bias: dict[str, int] | None,
    seed: int | None,
    stop: list[str] | None,
    extra_body: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    api_response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the full kwargs dict for client.chat.completions.create()."""
    api_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
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
    return api_kwargs


def build_generate_api_kwargs(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    logit_bias: dict[str, int] | None,
    seed: int | None,
    stop: list[str] | None,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full kwargs dict for client.completions.create()."""
    return {
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


# ---------------------------------------------------------------------------
# Chunk processing helpers
# ---------------------------------------------------------------------------


def extract_content_from_chat_chunk(
    chunk: ChatCompletionChunk,
    collected_content: list[str],
    tool_calls_acc: dict[int, dict[str, Any]],
) -> tuple[str | None, bool]:
    """Process a single chat completion chunk, accumulating content and tool calls.

    Returns:
        (finish_reason, in_think_block) — finish_reason is set when the chunk
        signals completion; in_think_block tracks reasoning token state.
    """
    finish_reason: str | None = None
    in_think_block = False

    if not chunk.choices:
        return finish_reason, in_think_block

    delta = chunk.choices[0].delta
    if not delta:
        return finish_reason, in_think_block

    if chunk.choices[0].finish_reason:
        finish_reason = chunk.choices[0].finish_reason

    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        in_think_block = True
        collected_content.append(delta.reasoning_content)

    if hasattr(delta, "content") and delta.content:
        collected_content.append(delta.content)

    if hasattr(delta, "tool_calls") and delta.tool_calls:
        for tc_delta in delta.tool_calls:
            accumulate_tool_call_delta(tool_calls_acc, tc_delta)

    return finish_reason, in_think_block


def extract_content_from_generate_chunk(
    chunk: Any,
    collected_content: list[str],
) -> str | None:
    """Process a single text completion chunk, accumulating content.

    Returns:
        finish_reason or None.
    """
    if not chunk.choices:
        return None
    delta = chunk.choices[0].text
    finish_reason: str | None = None
    if chunk.choices[0].finish_reason:
        finish_reason = chunk.choices[0].finish_reason
    if delta:
        collected_content.append(delta)
    return finish_reason


# ---------------------------------------------------------------------------
# Agentic tool loop helpers
# ---------------------------------------------------------------------------


def build_assistant_tool_message(
    content: str | None,
    tool_calls: list[ToolCallResult],
) -> dict[str, Any]:
    """Build the assistant message containing tool calls for the agentic loop."""
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.name, "arguments": tc.raw_arguments},
            }
            for tc in tool_calls
        ],
    }


def build_tool_result_message(
    tool_call_id: str,
    tool_result: Any,
) -> dict[str, Any]:
    """Build a tool role message from an execution result."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(tool_result, default=str),
    }
