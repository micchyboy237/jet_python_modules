"""Observability Wrapper for chat_stream.py using jet_telemetry.
Adds OpenTelemetry tracing, Phoenix integration, rich console logging,
and PII redaction around the pure streaming engine. All actual LLM logic
is delegated to chat_stream pure functions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Coroutine

# Import pure engine
from jet.libs.llama_cpp.usage.chat_stream import (
    run_chat_stream as _pure_run_chat_stream,
)
from jet.libs.llama_cpp.usage.chat_stream import (
    run_chat_stream_async as _pure_run_chat_stream_async,
)
from jet.libs.llama_cpp.usage.chat_stream import (
    run_generate_stream as _pure_run_generate_stream,
)
from jet.libs.llama_cpp.usage.chat_stream import (
    run_generate_stream_async as _pure_run_generate_stream_async,
)
from jet.libs.llama_cpp.usage.chat_stream_types import StreamCompletionResult
from jet.libs.llama_cpp.usage.chat_stream_utils import MODEL
from jet.libs.llama_cpp.usage.structured_output import (
    OutputFormat,
    resolve_response_format,
)

# Import jet_telemetry components
from jet_telemetry import get_trace_url, initialize_telemetry, redact
from openai import AsyncOpenAI, OpenAI
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

# Setup Rich Logging
from rich.console import Console
from rich.logging import RichHandler

console = Console(force_terminal=True, highlight=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("chat-stream-obs")

PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


def _ensure_telemetry(project_name: str, phoenix_url: str):
    """Ensure telemetry is initialized."""
    if project_name:
        initialize_telemetry(service_name=project_name, endpoint=phoenix_url)


def _extract_messages(
    prompt_or_messages: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize input to message list."""
    if isinstance(prompt_or_messages, list):
        return prompt_or_messages
    return [{"role": "user", "content": prompt_or_messages}]


def _prepare_span_attributes(
    model: str,
    messages: list[dict[str, Any]],
    is_agentic: bool,
    session_id: str | None,
    max_tool_rounds: int | None,
    span_kind: OpenInferenceSpanKindValues,
    provider: str = "llama_cpp",
) -> dict[str, Any]:
    """Prepare common span attributes with PII redaction."""
    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind.value,
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.LLM_PROVIDER: provider,
    }

    safe_messages = [
        {"role": m["role"], "content": redact(str(m.get("content", "")))}
        for m in messages
    ]

    attributes[SpanAttributes.INPUT_VALUE] = json.dumps(safe_messages)
    attributes[SpanAttributes.INPUT_MIME_TYPE] = "application/json"
    attributes[SpanAttributes.LLM_INPUT_MESSAGES] = json.dumps(safe_messages)

    if is_agentic:
        attributes[SpanAttributes.SESSION_ID] = session_id or "default-session"
        if max_tool_rounds:
            attributes["agent.max_steps"] = max_tool_rounds

    return attributes


def _record_result(
    span: Any, result: StreamCompletionResult, total_secs: float, ttft: float | None
):
    """Record results, metrics, and status to the span."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, redact(result.content[:4000]))
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "text/plain")
    span.set_attribute(
        SpanAttributes.LLM_OUTPUT_MESSAGES,
        json.dumps([{"role": "assistant", "content": redact(result.content)}]),
    )
    span.set_status(Status(StatusCode.OK))

    if result.usage:
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT, result.usage.get("prompt_tokens", 0)
        )
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
            result.usage.get("completion_tokens", 0),
        )
        span.set_attribute(
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL, result.usage.get("total_tokens", 0)
        )

    span.set_attribute("llm.latency.total_s", round(total_secs, 4))
    if ttft is not None:
        span.set_attribute("llm.latency.time_to_first_token_s", round(ttft, 4))

    if result.structured:
        span.set_attribute("llm.structured_output.success", result.structured.success)
        span.set_attribute(
            "llm.structured_output.format", result.structured.format_used.value
        )
        if result.structured.error:
            span.set_attribute(
                "llm.structured_output.error", redact(result.structured.error)
            )


def _make_chat_chunk_handler() -> tuple[Callable[[Any], None], dict[str, Any]]:
    """Create a per-chunk callback that flushes tokens to rich console."""
    state: dict[str, Any] = {"first_token_at": None, "in_think_block": False}

    def on_chunk(chunk: Any) -> None:
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
                console.print("[bold orange1]<think>[/bold orange1]", end="")
                state["in_think_block"] = True
            console.print(
                f"[bold orange1]{delta.reasoning_content}[/bold orange1]",
                end="",
                highlight=False,
                soft_wrap=True,
            )
        elif state["in_think_block"]:
            console.print("[bold orange1]</think>[/bold orange1]", end="")
            state["in_think_block"] = False

        if hasattr(delta, "content") and delta.content:
            console.print(
                f"[bold cyan]{delta.content}[/bold cyan]",
                end="",
                highlight=False,
                soft_wrap=True,
            )

    return on_chunk, state


def _make_generate_chunk_handler() -> tuple[Callable[[Any], None], dict[str, Any]]:
    """Create a per-chunk callback for raw text completion streaming."""
    state: dict[str, Any] = {"first_token_at": None}

    def on_chunk(chunk: Any) -> None:
        if not chunk.choices:
            return
        delta = chunk.choices[0].text
        if delta:
            if state["first_token_at"] is None:
                state["first_token_at"] = time.perf_counter()
            console.print(
                f"[bold cyan]{delta}[/bold cyan]",
                end="",
                highlight=False,
                soft_wrap=True,
            )

    return on_chunk, state


def _print_chat_header(
    *,
    model: str,
    image_source: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    tools: list[dict[str, Any]] | None,
    resolved_fmt: Any,
    trace_url: str | None = None,
) -> None:
    """Print the pre-stream header block via rich logger + console."""
    logger.info("─" * 60)
    logger.info(f"🖼️  Image source : {image_source or '(none — text-only)'}")
    logger.info(f"🤖 Model        : {model}")
    logger.info(f"🎛️  Sampling     : temp={temperature} top_p={top_p} top_k={top_k}")
    if tools:
        tool_names = [t.get("function", {}).get("name", "?") for t in tools]
        logger.info(f"🔧 Tools        : {tool_names}")
    if resolved_fmt.output_format != OutputFormat.TEXT:
        logger.info(f"📐 Response fmt : {resolved_fmt.output_format.value}")
    if trace_url:
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")


def _print_chat_footer(
    result: StreamCompletionResult,
    total_secs: float,
    ttft: float | None,
    trace_url: str | None = None,
) -> None:
    """Print the post-stream summary block via rich logger + console."""
    logger.info("─" * 60)
    logger.info("📊 Summary")
    if result.usage:
        tok_per_sec = (
            result.usage.get("completion_tokens", 0) / total_secs
            if total_secs > 0
            else 0.0
        )
        logger.info(
            f"   Tokens           : {result.usage.get('prompt_tokens', 0)}p / "
            f"{result.usage.get('completion_tokens', 0)}c / "
            f"{result.usage.get('total_tokens', 0)}t"
        )
        logger.info(f"   Throughput       : {tok_per_sec:.1f} tok/s")
        logger.info(f"   Duration         : {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        logger.info(f"   Response length  : {len(result.content)} chars")
        if result.finish_reason:
            logger.info(f"   Finish reason    : {result.finish_reason}")
        if result.has_tool_calls:
            logger.info(f"   Tool calls       : {len(result.tool_calls)}")
        if result.structured:
            status = "✅" if result.structured.success else "⚠️"
            logger.info(
                f"   Structured       : {status} {result.structured.format_used.value}"
            )
    if trace_url:
        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
    logger.info("─" * 60)


def _execute_traced_call(
    func: Callable[..., StreamCompletionResult]
    | Coroutine[Any, Any, StreamCompletionResult],
    is_async: bool,
    span_name: str,
    span_attributes: dict[str, Any],
    print_header: Callable,
    print_footer: Callable,
    phx_url: str,
    project_name: str,
    **kwargs,
) -> StreamCompletionResult:
    """Generic executor for both sync and async traced calls."""
    _ensure_telemetry(project_name, phx_url)

    tracer = otel_trace.get_tracer(__name__)

    with tracer.start_as_current_span(span_name, attributes=span_attributes) as span:
        trace_url = get_trace_url(phx_url) if project_name else None

        # Extract args needed for header/footer from kwargs
        print_header(
            model=kwargs.get("model", "unknown"),
            image_source=kwargs.get("image_source"),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.8),
            top_k=kwargs.get("top_k", 20),
            tools=kwargs.get("tools"),
            resolved_fmt=kwargs.get("resolved_fmt"),
            trace_url=trace_url,
        )

        # Determine chunk handler based on function name or explicit arg
        if "generate" in span_name:
            on_chunk, chunk_state = _make_generate_chunk_handler()
        else:
            on_chunk, chunk_state = _make_chat_chunk_handler()

        t_start = time.perf_counter()
        console.print("[bold cyan]Response:[/bold cyan] ", end="")

        # Handle sync vs async execution
        if is_async:
            # If we are already in an event loop, we should await.
            # For a library wrapper, we often use asyncio.run if called from sync context,
            # but here we assume the caller handles the loop if it's an async function.
            # However, since this wrapper is called by sync/async public APIs,
            # we handle the coroutine here.
            if isinstance(func, Coroutine):
                result = asyncio.run(
                    func(
                        on_chunk=on_chunk,
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k
                            not in [
                                "resolved_fmt",
                                "image_source",
                                "tools",
                                "temperature",
                                "top_p",
                                "top_k",
                                "model",
                            ]
                        },
                    )
                )
            else:
                # Fallback if passed a sync func to async wrapper (shouldn't happen)
                result = func(
                    on_chunk=on_chunk,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in [
                            "resolved_fmt",
                            "image_source",
                            "tools",
                            "temperature",
                            "top_p",
                            "top_k",
                            "model",
                        ]
                    },
                )
        else:
            result = func(
                on_chunk=on_chunk,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "resolved_fmt",
                        "image_source",
                        "tools",
                        "temperature",
                        "top_p",
                        "top_k",
                        "model",
                    ]
                },
            )

        if chunk_state.get("in_think_block"):
            console.print("[bold orange1]</think>[/bold orange1]", end="")
        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        _record_result(span, result, total_secs, ttft)
        print_footer(result, total_secs, ttft, trace_url)

    return result


# --- Public API ---


def run_chat_stream(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    project_name: str = "chat-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
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
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Traced synchronous chat streaming using jet_telemetry."""
    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None
    messages = _extract_messages(prompt_or_messages)

    span_kind = (
        OpenInferenceSpanKindValues.AGENT
        if is_agentic
        else OpenInferenceSpanKindValues.LLM
    )
    span_name = "agent.workflow" if is_agentic else "llm.chat_stream"

    attributes = _prepare_span_attributes(
        model, messages, is_agentic, session_id, max_tool_rounds, span_kind
    )

    return _execute_traced_call(
        func=_pure_run_chat_stream,
        is_async=False,
        span_name=span_name,
        span_attributes=attributes,
        print_header=_print_chat_header,
        print_footer=_print_chat_footer,
        phx_url=phoenix_url,
        project_name=project_name,
        prompt_or_messages=prompt_or_messages,
        model=model,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        tool_registry=tool_registry,
        response_format=response_format,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
        resolved_fmt=resolved_fmt,
    )


async def run_chat_stream_async(
    prompt_or_messages: str
    | list[dict[str, Any]] = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    project_name: str = "achat-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
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
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Traced asynchronous chat streaming using jet_telemetry."""
    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None
    messages = _extract_messages(prompt_or_messages)

    span_kind = (
        OpenInferenceSpanKindValues.AGENT
        if is_agentic
        else OpenInferenceSpanKindValues.LLM
    )
    span_name = "agent.workflow.async" if is_agentic else "llm.chat_stream.async"

    attributes = _prepare_span_attributes(
        model, messages, is_agentic, session_id, max_tool_rounds, span_kind
    )

    # Note: In a true async environment, we shouldn't use asyncio.run inside an already running loop.
    # We pass the coroutine directly to the helper which will handle it.
    coro = _pure_run_chat_stream_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        tool_registry=tool_registry,
        response_format=response_format,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
    )

    return _execute_traced_call(
        func=coro,
        is_async=True,
        span_name=span_name,
        span_attributes=attributes,
        print_header=_print_chat_header,
        print_footer=_print_chat_footer,
        phx_url=phoenix_url,
        project_name=project_name,
        prompt_or_messages=prompt_or_messages,
        model=model,
        image_source=image_source,
        client=client,
        enable_thinking=enable_thinking,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
        tool_registry=tool_registry,
        response_format=response_format,
        max_tool_rounds=max_tool_rounds,
        extra_body_params=extra_body_params,
        resolved_fmt=resolved_fmt,
    )


def run_generate_stream(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "generate-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
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
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Traced synchronous raw text completion using jet_telemetry."""
    messages = [{"role": "user", "content": prompt}]
    attributes = _prepare_span_attributes(
        model, messages, False, session_id, None, OpenInferenceSpanKindValues.LLM
    )

    # Custom header for generate mode
    def gen_header(**kwargs):
        trace_url = kwargs.get("trace_url")
        logger.info("─" * 60)
        logger.info(f"📝 Text Completion Mode | Model: {model}")
        if trace_url:
            console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

    def gen_footer(result, total_secs, ttft, trace_url):
        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        if trace_url:
            console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    return _execute_traced_call(
        func=_pure_run_generate_stream,
        is_async=False,
        span_name="llm.generate_stream",
        span_attributes=attributes,
        print_header=gen_header,
        print_footer=gen_footer,
        phx_url=phoenix_url,
        project_name=project_name,
        prompt=prompt,
        model=model,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body_params=extra_body_params,
        resolved_fmt=None,
    )


async def run_generate_stream_async(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "agenerate-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
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
    session_id: str | None = None,
) -> StreamCompletionResult:
    """Traced asynchronous raw text completion using jet_telemetry."""
    messages = [{"role": "user", "content": prompt}]
    attributes = _prepare_span_attributes(
        model, messages, False, session_id, None, OpenInferenceSpanKindValues.LLM
    )

    def gen_header(**kwargs):
        trace_url = kwargs.get("trace_url")
        logger.info("─" * 60)
        logger.info(f"📝 Async Text Completion Mode | Model: {model}")
        if trace_url:
            console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

    def gen_footer(result, total_secs, ttft, trace_url):
        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        if trace_url:
            console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    coro = _pure_run_generate_stream_async(
        prompt=prompt,
        model=model,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body_params=extra_body_params,
    )

    return _execute_traced_call(
        func=coro,
        is_async=True,
        span_name="llm.generate_stream.async",
        span_attributes=attributes,
        print_header=gen_header,
        print_footer=gen_footer,
        phx_url=phoenix_url,
        project_name=project_name,
        prompt=prompt,
        model=model,
        client=client,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        seed=seed,
        stop=stop,
        extra_body_params=extra_body_params,
        resolved_fmt=None,
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream chat completions with Phoenix observability."
    )
    parser.add_argument(
        "prompt", type=str, nargs="?", default="What is OpenTelemetry in one sentence?"
    )
    parser.add_argument("-i", "--image-source", type=str, default=None)
    parser.add_argument("--project", type=str, default="chat-stream-obs")
    parser.add_argument("--phoenix-url", type=str, default=PHOENIX_URL)
    parser.add_argument(
        "--no-capture-content", action="store_false", dest="capture_content"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1"),
    )
    parser.add_argument("--model", type=str, default=MODEL)
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
    parser.add_argument("--session-id", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    from jet.adapters.llama_cpp.factory import get_llm_client

    args = get_args()
    parsed_logit_bias: dict[str, int] | None = None
    if args.logit_bias:
        try:
            parsed_logit_bias = json.loads(args.logit_bias)
            logger.info(f"🎯 Logit bias applied: {parsed_logit_bias}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid logit_bias JSON: {e}")
            raise SystemExit(1)

    parsed_tools: list[dict[str, Any]] | None = None
    if args.tools_json:
        try:
            parsed_tools = json.loads(args.tools_json)
            logger.info(f"🔧 Loaded {len(parsed_tools)} tool definition(s)")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid tools JSON: {e}")
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
            logger.info(f"📐 Response format: {parsed_response_format}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid response_format JSON: {e}")
            raise SystemExit(1)

    logger.info("🚀 Startup config")
    logger.info(f"   Base URL     : {args.base_url}")
    logger.info(f"   Model        : {args.model}")
    logger.info(f"   Phoenix URL  : {args.phoenix_url}")
    logger.info(f"   Project      : {args.project}")

    client = get_llm_client(base_url=args.base_url, timeout=args.timeout)

    result = run_chat_stream(
        args.prompt,
        model=args.model,
        project_name=args.project,
        capture_content=args.capture_content,
        phoenix_url=args.phoenix_url,
        client=client,
        image_source=args.image_source,
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
        session_id=args.session_id,
    )

    if result.has_tool_calls:
        logger.info(
            f"📋 Result: {len(result.tool_calls)} tool call(s), finish_reason={result.finish_reason}"
        )
    else:
        logger.info(
            f"📋 Result: {len(result.content)} chars, finish_reason={result.finish_reason}"
        )
