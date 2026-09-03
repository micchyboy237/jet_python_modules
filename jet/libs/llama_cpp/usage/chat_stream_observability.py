"""Observability Wrapper for chat_stream.py.

Adds OpenTelemetry tracing, Phoenix integration, rich console logging,
and PII redaction around the pure streaming engine. All actual LLM logic
is delegated to chat_stream.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Callable

from jet.libs.llama_cpp.usage.chat_stream import (
    MODEL,
)
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
from jet.libs.llama_cpp.usage.observability_utils import (
    PHOENIX_URL,
    setup_observability,
)
from jet.libs.llama_cpp.usage.structured_output import (
    OutputFormat,
    resolve_response_format,
)
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from rich.console import Console
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("vision-stream-obs")

PII_PATTERNS = ["ssn", "password", "api_key", "secret", "token"]


def _redact(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    lower = text.lower()
    for pattern in PII_PATTERNS:
        if pattern in lower:
            return "[REDACTED]"
    return text


def format_trace_id(trace_id: int) -> str:
    return format(trace_id, "032x")


def build_phoenix_trace_url(phoenix_url: str, trace_id: int) -> str:
    return f"{phoenix_url.rstrip('/')}/redirects/traces/{format_trace_id(trace_id)}"


def _make_chat_chunk_handler() -> tuple[
    Callable[[ChatCompletionChunk], None],
    dict[str, Any],
]:
    """Create a per-chunk callback that flushes tokens to console in real-time.

    Returns:
        Tuple of (callback_fn, state_dict) where state_dict tracks
        first_token_at and in_think_block for post-stream metrics.
    """
    state: dict[str, Any] = {
        "first_token_at": None,
        "in_think_block": False,
    }

    def on_chunk(chunk: ChatCompletionChunk) -> None:
        if not chunk.choices:
            return

        delta = chunk.choices[0].delta
        if not delta:
            return

        # Track first token time
        if state["first_token_at"] is None and (
            getattr(delta, "content", None)
            or getattr(delta, "reasoning_content", None)
            or getattr(delta, "tool_calls", None)
        ):
            state["first_token_at"] = time.perf_counter()

        # Handle reasoning/thinking content
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

        # Handle regular content
        if hasattr(delta, "content") and delta.content:
            console.print(
                f"[bold cyan]{delta.content}[/bold cyan]",
                end="",
                highlight=False,
                soft_wrap=True,
            )

    return on_chunk, state


def _make_generate_chunk_handler() -> tuple[
    Callable[[Any], None],
    dict[str, Any],
]:
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
    """Traced synchronous chat streaming. Delegates to chat_stream.run_chat_stream."""
    if project_name:
        setup_observability(
            project_name=project_name,
            capture_content=capture_content,
            phoenix_url=phoenix_url,
        )

    tracer = trace.get_tracer(__name__)
    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None

    span_name = "agent_workflow" if is_agentic else "chat_completion"
    root_span_kind = (
        OpenInferenceSpanKindValues.AGENT.value
        if is_agentic
        else OpenInferenceSpanKindValues.CHAIN.value
    )

    input_val = prompt_or_messages if isinstance(prompt_or_messages, str) else ""
    if isinstance(prompt_or_messages, list) and prompt_or_messages:
        last_msg = prompt_or_messages[-1]
        content = last_msg.get("content")
        if isinstance(content, str):
            input_val = content
        elif isinstance(content, list):
            input_val = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

    with tracer.start_as_current_span(span_name) as root_span:
        root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, root_span_kind)
        root_span.set_attribute(
            SpanAttributes.INPUT_VALUE, _redact(str(input_val)[:3000])
        )
        if session_id is not None:
            root_span.set_attribute(SpanAttributes.SESSION_ID, session_id)

        trace_id = root_span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)

        root_span.set_attribute("llm.model", model)
        root_span.set_attribute(
            "agent.mode", "agentic" if is_agentic else "single_turn"
        )
        if tools:
            root_span.set_attribute("llm.tools.count", len(tools))

        logger.info("─" * 60)
        logger.info(f"🖼️  Image source : {image_source or '(none — text-only)'}")
        logger.info(f"🤖 Model        : {model}")
        logger.info(f"🎛️  Sampling     : temp={temperature} top_p={top_p} top_k={top_k}")
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            logger.info(f"🔧 Tools        : {tool_names}")
        if resolved_fmt.output_format != OutputFormat.TEXT:
            logger.info(f"📐 Response fmt : {resolved_fmt.output_format.value}")
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        # Build real-time chunk handler for console flushing
        on_chunk, chunk_state = _make_chat_chunk_handler()

        t_start = time.perf_counter()
        console.print("[bold cyan]Response:[/bold cyan] ", end="")

        result = _pure_run_chat_stream(
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
            on_chunk=on_chunk,
        )

        # Close any open think block and print newline
        if chunk_state["in_think_block"]:
            console.print("[bold orange1]</think>[/bold orange1]", end="")
        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        root_span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact(result.content[:3000])
        )
        if result.usage:
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                result.usage.get("prompt_tokens", 0),
            )
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                result.usage.get("completion_tokens", 0),
            )
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                result.usage.get("total_tokens", 0),
            )
        root_span.set_attribute("llm.latency.total_s", round(total_secs, 4))
        if ttft is not None:
            root_span.set_attribute("llm.latency.time_to_first_token_s", round(ttft, 4))
        root_span.set_status(Status(StatusCode.OK))

        logger.info("─" * 60)
        logger.info(f"📊 Summary")
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

        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
        logger.info("─" * 60)

    return result


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
    """Traced asynchronous chat streaming. Delegates to chat_stream.run_chat_stream_async."""
    if project_name:
        setup_observability(
            project_name=project_name,
            capture_content=capture_content,
            phoenix_url=phoenix_url,
        )

    tracer = trace.get_tracer(__name__)
    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None

    span_name = "agent_workflow" if is_agentic else "chat_completion"
    root_span_kind = (
        OpenInferenceSpanKindValues.AGENT.value
        if is_agentic
        else OpenInferenceSpanKindValues.CHAIN.value
    )

    input_val = prompt_or_messages if isinstance(prompt_or_messages, str) else ""
    if isinstance(prompt_or_messages, list) and prompt_or_messages:
        last_msg = prompt_or_messages[-1]
        content = last_msg.get("content")
        if isinstance(content, str):
            input_val = content
        elif isinstance(content, list):
            input_val = " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )

    with tracer.start_as_current_span(span_name) as root_span:
        root_span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, root_span_kind)
        root_span.set_attribute(
            SpanAttributes.INPUT_VALUE, _redact(str(input_val)[:3000])
        )
        if session_id is not None:
            root_span.set_attribute(SpanAttributes.SESSION_ID, session_id)

        trace_id = root_span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)

        root_span.set_attribute("llm.model", model)
        root_span.set_attribute(
            "agent.mode", "agentic" if is_agentic else "single_turn"
        )
        if tools:
            root_span.set_attribute("llm.tools.count", len(tools))

        logger.info("─" * 60)
        logger.info(f"🖼️  Image source : {image_source or '(none — text-only)'}")
        logger.info(f"🤖 Model        : {model}")
        logger.info(f"🎛️  Sampling     : temp={temperature} top_p={top_p} top_k={top_k}")
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            logger.info(f"🔧 Tools        : {tool_names}")
        if resolved_fmt.output_format != OutputFormat.TEXT:
            logger.info(f"📐 Response fmt : {resolved_fmt.output_format.value}")
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        # Build real-time chunk handler for console flushing
        on_chunk, chunk_state = _make_chat_chunk_handler()

        t_start = time.perf_counter()
        console.print("[bold cyan]Response:[/bold cyan] ", end="")

        result = await _pure_run_chat_stream_async(
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
            on_chunk=on_chunk,
        )

        # Close any open think block and print newline
        if chunk_state["in_think_block"]:
            console.print("[bold orange1]</think>[/bold orange1]", end="")
        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        root_span.set_attribute(
            SpanAttributes.OUTPUT_VALUE, _redact(result.content[:3000])
        )
        if result.usage:
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                result.usage.get("prompt_tokens", 0),
            )
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                result.usage.get("completion_tokens", 0),
            )
            root_span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                result.usage.get("total_tokens", 0),
            )
        root_span.set_attribute("llm.latency.total_s", round(total_secs, 4))
        if ttft is not None:
            root_span.set_attribute("llm.latency.time_to_first_token_s", round(ttft, 4))
        root_span.set_status(Status(StatusCode.OK))

        logger.info("─" * 60)
        logger.info(f"📊 Summary")
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

        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
        logger.info("─" * 60)

    return result


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
    """Traced synchronous raw text completion. Delegates to chat_stream.run_generate_stream."""
    if project_name:
        setup_observability(
            project_name=project_name,
            capture_content=capture_content,
            phoenix_url=phoenix_url,
        )

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("text_completion") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)

        trace_id = span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)

        span.set_attribute("llm.model", model)
        if session_id is not None:
            span.set_attribute("session.id", session_id)

        logger.info("─" * 60)
        logger.info(f"📝 Text Completion Mode | Model: {model}")
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        # Build real-time chunk handler for console flushing
        on_chunk, chunk_state = _make_generate_chunk_handler()

        t_start = time.perf_counter()
        console.print("[bold cyan]Response:[/bold cyan] ", end="")

        result = _pure_run_generate_stream(
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
            on_chunk=on_chunk,
        )

        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.content)
        if result.usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                result.usage.get("prompt_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                result.usage.get("completion_tokens", 0),
            )
        span.set_attribute("llm.latency.total_s", round(total_secs, 4))
        if ttft is not None:
            span.set_attribute("llm.latency.time_to_first_token_s", round(ttft, 4))
        span.set_status(Status(StatusCode.OK))

        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    return result


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
    """Traced asynchronous raw text completion. Delegates to chat_stream.run_generate_stream_async."""
    if project_name:
        setup_observability(
            project_name=project_name,
            capture_content=capture_content,
            phoenix_url=phoenix_url,
        )

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("text_completion") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, prompt)

        trace_id = span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)

        span.set_attribute("llm.model", model)
        if session_id is not None:
            span.set_attribute("session.id", session_id)

        logger.info("─" * 60)
        logger.info(f"📝 Async Text Completion Mode | Model: {model}")
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        # Build real-time chunk handler for console flushing
        on_chunk, chunk_state = _make_generate_chunk_handler()

        t_start = time.perf_counter()
        console.print("[bold cyan]Response:[/bold cyan] ", end="")

        result = await _pure_run_generate_stream_async(
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
            on_chunk=on_chunk,
        )

        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.content)
        if result.usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                result.usage.get("prompt_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                result.usage.get("completion_tokens", 0),
            )
        span.set_attribute("llm.latency.total_s", round(total_secs, 4))
        if ttft is not None:
            span.set_attribute("llm.latency.time_to_first_token_s", round(ttft, 4))
        span.set_status(Status(StatusCode.OK))

        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    return result


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream chat completions with Phoenix observability."
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="What is OpenTelemetry in one sentence?",
        help="Prompt for the chat/image analysis.",
    )
    parser.add_argument(
        "-i",
        "--image-source",
        type=str,
        default=None,
        help="Path or URL to an image to analyze. Omit for text-only.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="chat-stream-obs",
        help="Phoenix project name to log traces under.",
    )
    parser.add_argument(
        "--phoenix-url",
        type=str,
        default=PHOENIX_URL,
        help="Phoenix server base URL (env: LLM_OBS_PHOENIX_URL).",
    )
    parser.add_argument(
        "--no-capture-content",
        action="store_false",
        dest="capture_content",
        help="Disable capturing prompt/response text in traces.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1"),
        help="OpenAI-compatible server base URL (env: LLAMA_CPP_VISION_URL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to request (env: LLAMA_CPP_VISION_MODEL).",
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
        help="JSON array of tool definitions for function calling.",
    )
    parser.add_argument(
        "--tool-choice",
        type=str,
        default=None,
        help='"auto", "none", "required", or JSON object.',
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default=None,
        help='JSON response format, e.g. \'{"type": "json_object"}\'.',
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID to group traces as a conversation thread in Phoenix.",
    )
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
            f"📋 Result: {len(result.tool_calls)} tool call(s), "
            f"finish_reason={result.finish_reason}"
        )
    else:
        logger.info(
            f"📋 Result: {len(result.content)} chars, "
            f"finish_reason={result.finish_reason}"
        )
