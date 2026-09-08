"""Pure LLM Streaming Engine — No Observability Dependencies.
Provides sync/async chat and text completion streaming with tool-use loops,
vision input, and structured output parsing. Completely free of OpenTelemetry,
Phoenix, or any tracing side effects.

Uses chat_stream_utils for all shared logic. When no on_chunk callback is
provided, defaults to a plain-stdout printer with post-stream summary.

For traced execution, use chat_stream_observability.py instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Callable

from jet.adapters.llama_cpp.factory import get_async_llm_client, get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet.libs.llama_cpp.usage.chat_stream_utils import (
    MODEL,
    build_assistant_tool_message,
    build_chat_api_kwargs,
    build_chat_extra_body,
    build_generate_api_kwargs,
    build_generate_extra_body,
    build_messages,
    build_tool_result_message,
    encode_image_to_base64,
    encode_image_to_base64_async,
    execute_tool,
    execute_tool_async,
    extract_content_from_chat_chunk,
    extract_content_from_generate_chunk,
    parse_tool_calls_from_accumulator,
    prepare_response_format,
)
from jet.libs.llama_cpp.usage.structured_output import (
    OutputFormat,
    parse_structured_content,
)
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default console printers (no rich dependency)
# ---------------------------------------------------------------------------


def make_console_chat_printer() -> tuple[
    Callable[[ChatCompletionChunk], None], dict[str, Any]
]:
    """Create a plain-stdout on_chunk callback for live chat streaming output.
    Returns:
        Tuple of (callback_fn, state_dict). state_dict tracks
        "first_token_at" and "in_think_block" for post-stream metrics.
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
    Returns:
        Tuple of (callback_fn, state_dict). state_dict tracks "first_token_at".
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
    """Print a 📊 Summary block via plain print() — no rich/OTel dependency."""
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


# ---------------------------------------------------------------------------
# Sync chat stream
# ---------------------------------------------------------------------------


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
        on_chunk: Optional callback invoked for each streamed chunk. When None,
            defaults to a plain-stdout printer with post-stream summary. Pass a
            custom callback (e.g. from chat_stream_observability) to override.
    """
    _using_default_printer = on_chunk is None
    chunk_state: dict[str, Any] = {"first_token_at": None, "in_think_block": False}
    if _using_default_printer:
        on_chunk, chunk_state = make_console_chat_printer()

    resolved_fmt, api_response_format, extra_body_params = prepare_response_format(
        response_format, extra_body_params
    )

    if client is None:
        client = get_llm_client()

    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
    else:
        messages = prompt_or_messages

    current_messages = build_messages(
        prompt,
        messages,
        image_source,
        resolved_fmt.system_prompt_addition,
        encode_image_to_base64,
    )

    is_agentic = tool_registry is not None
    last_result: StreamCompletionResult | None = None
    round_num = 0
    t_start = time.perf_counter()

    while round_num < max_tool_rounds:
        round_num += 1
        extra_body = build_chat_extra_body(
            top_k=top_k,
            enable_thinking=enable_thinking,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            extra_body_params=extra_body_params,
        )
        api_kwargs = build_chat_api_kwargs(
            model=model,
            messages=current_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            seed=seed,
            stop=stop,
            extra_body=extra_body,
            tools=tools,
            tool_choice=tool_choice,
            api_response_format=api_response_format,
        )

        collected_content: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = None
        finish_reason: str | None = None

        stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
            **api_kwargs
        )
        for chunk in stream:
            on_chunk(chunk)
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                continue
            fr, _ = extract_content_from_chat_chunk(
                chunk, collected_content, tool_calls_acc
            )
            if fr:
                finish_reason = fr

        full_response = "".join(collected_content)
        parsed_tool_calls = parse_tool_calls_from_accumulator(tool_calls_acc)
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

        if not last_result.has_tool_calls or not is_agentic:
            break

        current_messages.append(
            build_assistant_tool_message(last_result.content, last_result.tool_calls)
        )
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
            current_messages.append(build_tool_result_message(tc.id, tool_result))

    result = last_result or StreamCompletionResult(
        content="", finish_reason="no_response"
    )

    if _using_default_printer:
        if chunk_state.get("in_think_block"):
            print("</think>", end="", flush=True)
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(
            result, total_secs, chunk_state.get("first_token_at"), t_start
        )

    return result


# ---------------------------------------------------------------------------
# Sync generate stream
# ---------------------------------------------------------------------------


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
        on_chunk: Optional callback. When None, defaults to plain-stdout printer.
    """
    _using_default_printer = on_chunk is None
    chunk_state: dict[str, Any] = {"first_token_at": None}
    if _using_default_printer:
        on_chunk, chunk_state = make_console_generate_printer()

    if client is None:
        client = get_llm_client()

    extra_body = build_generate_extra_body(
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        extra_body_params=extra_body_params,
    )
    api_kwargs = build_generate_api_kwargs(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body=extra_body,
    )

    collected_content: list[str] = []
    usage = None
    finish_reason: str | None = None
    t_start = time.perf_counter()

    stream = client.completions.create(**api_kwargs)
    for chunk in stream:
        on_chunk(chunk)
        if not chunk.choices:
            usage = getattr(chunk, "usage", None)
            continue
        fr = extract_content_from_generate_chunk(chunk, collected_content)
        if fr:
            finish_reason = fr

    result = StreamCompletionResult(
        content="".join(collected_content),
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

    if _using_default_printer:
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(
            result, total_secs, chunk_state.get("first_token_at"), t_start
        )

    return result


# ---------------------------------------------------------------------------
# Async chat stream
# ---------------------------------------------------------------------------


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
        on_chunk: Optional callback. When None, defaults to plain-stdout printer.
    Note:
        Stream and client lifecycle are managed internally by the OpenAI SDK.
        Do NOT call stream.aclose() or client.close() manually.
    """
    _using_default_printer = on_chunk is None
    chunk_state: dict[str, Any] = {"first_token_at": None, "in_think_block": False}
    if _using_default_printer:
        on_chunk, chunk_state = make_console_chat_printer()

    resolved_fmt, api_response_format, extra_body_params = prepare_response_format(
        response_format, extra_body_params
    )

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

    current_messages = build_messages(
        prompt,
        messages,
        image_source,
        resolved_fmt.system_prompt_addition,
        lambda src: encoded_image if encoded_image else ("", "image/jpeg"),
    )

    is_agentic = tool_registry is not None
    last_result: StreamCompletionResult | None = None
    round_num = 0
    t_start = time.perf_counter()

    while round_num < max_tool_rounds:
        round_num += 1
        extra_body = build_chat_extra_body(
            top_k=top_k,
            enable_thinking=enable_thinking,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            extra_body_params=extra_body_params,
        )
        api_kwargs = build_chat_api_kwargs(
            model=model,
            messages=current_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            seed=seed,
            stop=stop,
            extra_body=extra_body,
            tools=tools,
            tool_choice=tool_choice,
            api_response_format=api_response_format,
        )

        collected_content: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = None
        finish_reason: str | None = None

        stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            **api_kwargs
        )
        async for chunk in stream:
            on_chunk(chunk)
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                continue
            fr, _ = extract_content_from_chat_chunk(
                chunk, collected_content, tool_calls_acc
            )
            if fr:
                finish_reason = fr

        full_response = "".join(collected_content)
        parsed_tool_calls = parse_tool_calls_from_accumulator(tool_calls_acc)
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

        if not last_result.has_tool_calls or not is_agentic:
            break

        current_messages.append(
            build_assistant_tool_message(last_result.content, last_result.tool_calls)
        )
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
            current_messages.append(build_tool_result_message(tc.id, tool_result))

    result = last_result or StreamCompletionResult(
        content="", finish_reason="no_response"
    )

    if _using_default_printer:
        if chunk_state.get("in_think_block"):
            print("</think>", end="", flush=True)
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(
            result, total_secs, chunk_state.get("first_token_at"), t_start
        )

    return result


# ---------------------------------------------------------------------------
# Async generate stream
# ---------------------------------------------------------------------------


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
        on_chunk: Optional callback. When None, defaults to plain-stdout printer.
    Note:
        Stream and client lifecycle are managed internally by the OpenAI SDK.
        Do NOT call stream.aclose() or client.close() manually.
    """
    _using_default_printer = on_chunk is None
    chunk_state: dict[str, Any] = {"first_token_at": None}
    if _using_default_printer:
        on_chunk, chunk_state = make_console_generate_printer()

    if client is None:
        client = get_async_llm_client()

    extra_body = build_generate_extra_body(
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        extra_body_params=extra_body_params,
    )
    api_kwargs = build_generate_api_kwargs(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body=extra_body,
    )

    collected_content: list[str] = []
    usage = None
    finish_reason: str | None = None
    t_start = time.perf_counter()

    stream = await client.completions.create(**api_kwargs)
    async for chunk in stream:
        on_chunk(chunk)
        if not chunk.choices:
            usage = getattr(chunk, "usage", None)
            continue
        fr = extract_content_from_generate_chunk(chunk, collected_content)
        if fr:
            finish_reason = fr

    result = StreamCompletionResult(
        content="".join(collected_content),
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

    if _using_default_printer:
        print()
        total_secs = time.perf_counter() - t_start
        print_stream_summary(
            result, total_secs, chunk_state.get("first_token_at"), t_start
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    parser.add_argument("-i", "--image-source", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--base-url", type=str, default=LLAMA_CPP_BASE_URL)
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
    parser.add_argument("--logit-bias", type=str, default=None)
    parser.add_argument("--tools-json", type=str, default=None)
    parser.add_argument("--tool-choice", type=str, default=None)
    parser.add_argument("--response-format", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    import sys

    from jet.libs.llama_cpp.usage.chat_stream_utils import LLAMA_CPP_BASE_URL

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

    if args.generate:
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
        )
    else:
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
        )

    if result.has_tool_calls:
        print(
            f"📋 Result: {len(result.tool_calls)} tool call(s), finish_reason={result.finish_reason}"
        )
    else:
        print(
            f"📋 Result: {len(result.content)} chars, finish_reason={result.finish_reason}"
        )
