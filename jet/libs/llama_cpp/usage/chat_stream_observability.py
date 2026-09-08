"""Observability Wrapper for chat_stream.py.
Adds OpenTelemetry tracing, Phoenix integration, rich console logging,
and PII redaction around the pure streaming engine. All actual LLM logic
is delegated to chat_stream_utils + chat_stream pure functions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Callable

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
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
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


# ---------------------------------------------------------------------------
# Rich chunk handlers (observability-specific)
# ---------------------------------------------------------------------------


def _make_chat_chunk_handler() -> tuple[
    Callable[[ChatCompletionChunk], None], dict[str, Any]
]:
    """Create a per-chunk callback that flushes tokens to rich console."""
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


# ---------------------------------------------------------------------------
# Shared span attribute setter
# ---------------------------------------------------------------------------


def _set_chat_span_attributes(
    span: Any,
    *,
    model: str,
    is_agentic: bool,
    tools: list[dict[str, Any]] | None,
    session_id: str | None,
    input_val: str,
) -> None:
    """Set common span attributes for chat/agent spans."""
    root_span_kind = (
        OpenInferenceSpanKindValues.AGENT.value
        if is_agentic
        else OpenInferenceSpanKindValues.CHAIN.value
    )
    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, root_span_kind)
    span.set_attribute(SpanAttributes.INPUT_VALUE, _redact(str(input_val)[:3000]))
    if session_id is not None:
        span.set_attribute(SpanAttributes.SESSION_ID, session_id)
    span.set_attribute("llm.model", model)
    span.set_attribute("agent.mode", "agentic" if is_agentic else "single_turn")
    if tools:
        span.set_attribute("llm.tools.count", len(tools))


def _set_completion_span_results(
    span: Any,
    result: StreamCompletionResult,
    total_secs: float,
    ttft: float | None,
) -> None:
    """Set output, token counts, latency, and status on a completed span."""
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, _redact(result.content[:3000]))
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
    span.set_status(Status(StatusCode.OK))


def _print_chat_header(
    *,
    model: str,
    image_source: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    tools: list[dict[str, Any]] | None,
    resolved_fmt: Any,
    trace_url: str,
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
    console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")


def _print_chat_footer(
    result: StreamCompletionResult,
    total_secs: float,
    ttft: float | None,
    trace_url: str,
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
    console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
    logger.info("─" * 60)


def _extract_input_value(prompt_or_messages: str | list[dict[str, Any]]) -> str:
    """Extract a string input value from prompt_or_messages for span attributes."""
    if isinstance(prompt_or_messages, str):
        return prompt_or_messages
    if isinstance(prompt_or_messages, list) and prompt_or_messages:
        last_msg = prompt_or_messages[-1]
        content = last_msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content if p.get("type") == "text"
            )
    return ""


# ---------------------------------------------------------------------------
# Traced sync chat
# ---------------------------------------------------------------------------


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
    input_val = _extract_input_value(prompt_or_messages)

    with tracer.start_as_current_span(span_name) as root_span:
        _set_chat_span_attributes(
            root_span,
            model=model,
            is_agentic=is_agentic,
            tools=tools,
            session_id=session_id,
            input_val=input_val,
        )
        trace_id = root_span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)
        _print_chat_header(
            model=model,
            image_source=image_source,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tools=tools,
            resolved_fmt=resolved_fmt,
            trace_url=trace_url,
        )

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

        if chunk_state["in_think_block"]:
            console.print("[bold orange1]</think>[/bold orange1]", end="")
        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        _set_completion_span_results(root_span, result, total_secs, ttft)
        _print_chat_footer(result, total_secs, ttft, trace_url)

    return result


# ---------------------------------------------------------------------------
# Traced async chat
# ---------------------------------------------------------------------------


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
    input_val = _extract_input_value(prompt_or_messages)

    with tracer.start_as_current_span(span_name) as root_span:
        _set_chat_span_attributes(
            root_span,
            model=model,
            is_agentic=is_agentic,
            tools=tools,
            session_id=session_id,
            input_val=input_val,
        )
        trace_id = root_span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)
        _print_chat_header(
            model=model,
            image_source=image_source,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tools=tools,
            resolved_fmt=resolved_fmt,
            trace_url=trace_url,
        )

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

        if chunk_state["in_think_block"]:
            console.print("[bold orange1]</think>[/bold orange1]", end="")
        console.print()

        total_secs = time.perf_counter() - t_start
        ttft = chunk_state.get("first_token_at")
        if ttft is not None:
            ttft = ttft - t_start

        _set_completion_span_results(root_span, result, total_secs, ttft)
        _print_chat_footer(result, total_secs, ttft, trace_url)

    return result


# ---------------------------------------------------------------------------
# Traced sync generate
# ---------------------------------------------------------------------------


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
    """Traced synchronous raw text completion."""
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

        _set_completion_span_results(span, result, total_secs, ttft)
        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    return result


# ---------------------------------------------------------------------------
# Traced async generate
# ---------------------------------------------------------------------------


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
    """Traced asynchronous raw text completion."""
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

        _set_completion_span_results(span, result, total_secs, ttft)
        logger.info(f"📊 Done: {len(result.content)} chars in {total_secs:.2f}s")
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        console.print(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
