"""Observability Wrapper for chat_stream.py using jet_telemetry.
Refactored to use granular decorators (@agent, @llm, @tool, @evaluator)
instead of manual span management. Aligns with telemetry_utils/examples.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any, Callable

# Initialize telemetry EARLY so decorators have access to Phoenix tracer
from jet_telemetry import initialize_telemetry

PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")
initialize_telemetry(service_name="chat-stream-obs", endpoint=PHOENIX_URL)

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
    StructuredResult,
    resolve_response_format,
)
from jet_telemetry import (
    agent,
    chain,
    evaluator,
    get_trace_url,
    llm,
    redact,
    tool,
)
from openai import AsyncOpenAI, OpenAI
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as otel_trace
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


# ---------------------------------------------------------------------------
# Granular Observability Wrappers
# ---------------------------------------------------------------------------


@tool(name="encode_image_input")
def observe_image_encoding(image_source: str | None) -> tuple[str, str] | None:
    """Wraps image encoding in a TOOL span for visibility."""
    if not image_source:
        return None
    from jet.libs.llama_cpp.usage.chat_stream_utils import encode_image_to_base64

    try:
        return encode_image_to_base64(image_source)
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise


@evaluator(name="parse_structured_output")
def observe_structured_parsing(
    content: str, resolved_fmt: Any
) -> StructuredResult | None:
    """Wraps structured output parsing in an EVALUATOR span."""
    if resolved_fmt.output_format == OutputFormat.TEXT:
        return None

    from jet.libs.llama_cpp.usage.structured_output import parse_structured_content

    result = parse_structured_content(content, resolved_fmt)

    # Add specific attributes to the current span
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute("evaluator.format", resolved_fmt.output_format.value)
        span.set_attribute("evaluator.success", result.success)
        if result.error:
            span.set_attribute("evaluator.error", redact(result.error))

    return result


@llm(model_name="unknown")
def observe_llm_chat_stream(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    on_chunk: Callable,
    client: OpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """
    Decorated LLM span for synchronous chat streaming.
    Delegates to pure engine while capturing semantic attributes.
    """
    result = _pure_run_chat_stream(
        prompt_or_messages=prompt_or_messages,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    # Post-hoc attribute setting for streaming metrics
    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
        if result.usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                result.usage.get("total_tokens", 0),
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                result.usage.get("completion_tokens", 0),
            )
    return result


@llm(model_name="unknown")
async def observe_llm_chat_stream_async(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    on_chunk: Callable,
    client: AsyncOpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Decorated LLM span for asynchronous chat streaming."""
    result = await _pure_run_chat_stream_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    span = otel_trace.get_current_span()
    if span.is_recording():
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
        if result.usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                result.usage.get("total_tokens", 0),
            )
    return result


@chain(name="llm.generate_session")
def observe_generate_stream(
    prompt: str,
    model: str,
    on_chunk: Callable,
    client: OpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Wraps raw text generation in a CHAIN span."""
    return _pure_run_generate_stream(
        prompt=prompt,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )


@chain(name="llm.generate_session.async")
async def observe_generate_stream_async(
    prompt: str,
    model: str,
    on_chunk: Callable,
    client: AsyncOpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Wraps async raw text generation in a CHAIN span."""
    return await _pure_run_generate_stream_async(
        prompt=prompt,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# High-Level Orchestrators
# ---------------------------------------------------------------------------


@agent(name="agent.chat_loop")
def run_agentic_chat(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    tool_registry: dict[str, Callable],
    resolved_fmt: Any,
    on_chunk: Callable,
    client: OpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """
    Top-level AGENT span for agentic loops.
    Handles image encoding, tool execution loops, and structured parsing.
    """
    # 1. Handle Image Encoding (if present in kwargs)
    image_source = kwargs.get("image_source")
    if image_source:
        observe_image_encoding(image_source)

    # 2. Execute the Pure Chat Stream (which contains the internal tool loop)
    result = observe_llm_chat_stream(
        prompt_or_messages=prompt_or_messages,
        model=model,
        tool_registry=tool_registry,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    # 3. Observe Structured Output Parsing
    if result.content:
        structured_result = observe_structured_parsing(result.content, resolved_fmt)
        if structured_result:
            result.structured = structured_result

    return result


@agent(name="agent.chat_loop.async")
async def run_agentic_chat_async(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    tool_registry: dict[str, Callable],
    resolved_fmt: Any,
    on_chunk: Callable,
    client: AsyncOpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Async top-level AGENT span for agentic loops."""
    image_source = kwargs.get("image_source")
    if image_source:
        # Note: For async, we'd ideally use an async image encoder wrapper
        # For now, reusing sync wrapper inside async agent is acceptable
        # as image encoding is usually fast or IO-bound via httpx internally
        observe_image_encoding(image_source)

    result = await observe_llm_chat_stream_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        tool_registry=tool_registry,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    if result.content:
        structured_result = observe_structured_parsing(result.content, resolved_fmt)
        if structured_result:
            result.structured = structured_result

    return result


@chain(name="llm.chat_session")
def run_simple_chat(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    resolved_fmt: Any,
    on_chunk: Callable,
    client: OpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Top-level CHAIN span for non-agentic chat."""
    image_source = kwargs.get("image_source")
    if image_source:
        observe_image_encoding(image_source)

    result = observe_llm_chat_stream(
        prompt_or_messages=prompt_or_messages,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    if result.content:
        structured_result = observe_structured_parsing(result.content, resolved_fmt)
        if structured_result:
            result.structured = structured_result

    return result


@chain(name="llm.chat_session.async")
async def run_simple_chat_async(
    prompt_or_messages: str | list[dict[str, Any]],
    model: str,
    resolved_fmt: Any,
    on_chunk: Callable,
    client: AsyncOpenAI | None = None,
    **kwargs,
) -> StreamCompletionResult:
    """Async top-level CHAIN span for non-agentic chat."""
    image_source = kwargs.get("image_source")
    if image_source:
        observe_image_encoding(image_source)

    result = await observe_llm_chat_stream_async(
        prompt_or_messages=prompt_or_messages,
        model=model,
        on_chunk=on_chunk,
        client=client,
        **kwargs,
    )

    if result.content:
        structured_result = observe_structured_parsing(result.content, resolved_fmt)
        if structured_result:
            result.structured = structured_result

    return result


# ---------------------------------------------------------------------------
# Console & Summary Helpers
# ---------------------------------------------------------------------------


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


def _print_header_footer(
    result: StreamCompletionResult,
    total_secs: float,
    ttft: float | None,
    model: str,
    trace_url: str | None,
    is_agentic: bool = False,
):
    """Unified printing logic for stream summaries."""
    logger.info("─" * 60)
    if is_agentic:
        logger.info(f"🤖 Agent Mode | Model: {model}")
    else:
        logger.info(f"💬 Chat Mode | Model: {model}")

    if trace_url:
        console.print(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

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

    if result.structured:
        status = "✅" if result.structured.success else "⚠️"
        logger.info(
            f"   Structured       : {status} {result.structured.format_used.value}"
        )

    logger.info("─" * 60)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_chat_stream(
    prompt_or_messages: str | list[dict[str, Any]] = "What is OpenTelemetry?",
    model: str = MODEL,
    *,
    project_name: str = "chat-stream-obs",
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
    """Traced synchronous chat streaming using jet_telemetry decorators."""
    _ensure_telemetry(project_name, phoenix_url)

    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None

    on_chunk, chunk_state = _make_chat_chunk_handler()

    console.print("[bold cyan]Response:[/bold cyan] ", end="")
    t_start = time.perf_counter()

    common_kwargs = {
        "model": model,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "seed": seed,
        "stop": stop,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tool_rounds": max_tool_rounds,
        "extra_body_params": extra_body_params,
        "image_source": image_source,
    }

    if is_agentic:
        result = run_agentic_chat(
            prompt_or_messages=prompt_or_messages,
            tool_registry=tool_registry,
            resolved_fmt=resolved_fmt,
            on_chunk=on_chunk,
            client=client,
            **common_kwargs,
        )
    else:
        result = run_simple_chat(
            prompt_or_messages=prompt_or_messages,
            resolved_fmt=resolved_fmt,
            on_chunk=on_chunk,
            client=client,
            **common_kwargs,
        )

    total_secs = time.perf_counter() - t_start
    ttft = chunk_state.get("first_token_at")
    if ttft is not None:
        ttft = ttft - t_start

    trace_url = get_trace_url(phoenix_url) if project_name else None
    _print_header_footer(result, total_secs, ttft, model, trace_url, is_agentic)

    return result


async def run_chat_stream_async(
    prompt_or_messages: str | list[dict[str, Any]] = "What is OpenTelemetry?",
    model: str = MODEL,
    *,
    project_name: str = "achat-stream-obs",
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
    """Traced asynchronous chat streaming using jet_telemetry decorators."""
    _ensure_telemetry(project_name, phoenix_url)

    resolved_fmt = resolve_response_format(response_format)
    is_agentic = tool_registry is not None

    on_chunk, chunk_state = _make_chat_chunk_handler()

    console.print("[bold cyan]Response:[/bold cyan] ", end="")
    t_start = time.perf_counter()

    common_kwargs = {
        "model": model,
        "enable_thinking": enable_thinking,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "seed": seed,
        "stop": stop,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tool_rounds": max_tool_rounds,
        "extra_body_params": extra_body_params,
        "image_source": image_source,
    }

    if is_agentic:
        result = await run_agentic_chat_async(
            prompt_or_messages=prompt_or_messages,
            tool_registry=tool_registry,
            resolved_fmt=resolved_fmt,
            on_chunk=on_chunk,
            client=client,
            **common_kwargs,
        )
    else:
        result = await run_simple_chat_async(
            prompt_or_messages=prompt_or_messages,
            resolved_fmt=resolved_fmt,
            on_chunk=on_chunk,
            client=client,
            **common_kwargs,
        )

    total_secs = time.perf_counter() - t_start
    ttft = chunk_state.get("first_token_at")
    if ttft is not None:
        ttft = ttft - t_start

    trace_url = get_trace_url(phoenix_url) if project_name else None
    _print_header_footer(result, total_secs, ttft, model, trace_url, is_agentic)

    return result


def run_generate_stream(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "generate-stream-obs",
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
) -> StreamCompletionResult:
    """Traced synchronous raw text completion."""
    _ensure_telemetry(project_name, phoenix_url)

    on_chunk, chunk_state = _make_chat_chunk_handler()

    console.print("[bold cyan]Response:[/bold cyan] ", end="")
    t_start = time.perf_counter()

    result = observe_generate_stream(
        prompt=prompt,
        model=model,
        on_chunk=on_chunk,
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

    total_secs = time.perf_counter() - t_start
    ttft = chunk_state.get("first_token_at")
    if ttft is not None:
        ttft = ttft - t_start

    trace_url = get_trace_url(phoenix_url) if project_name else None
    _print_header_footer(result, total_secs, ttft, model, trace_url, is_agentic=False)

    return result


async def run_generate_stream_async(
    prompt: str,
    model: str = MODEL,
    *,
    project_name: str = "agenerate-stream-obs",
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
) -> StreamCompletionResult:
    """Traced asynchronous raw text completion."""
    _ensure_telemetry(project_name, phoenix_url)

    on_chunk, chunk_state = _make_chat_chunk_handler()

    console.print("[bold cyan]Response:[/bold cyan] ", end="")
    t_start = time.perf_counter()

    result = await observe_generate_stream_async(
        prompt=prompt,
        model=model,
        on_chunk=on_chunk,
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

    total_secs = time.perf_counter() - t_start
    ttft = chunk_state.get("first_token_at")
    if ttft is not None:
        ttft = ttft - t_start

    trace_url = get_trace_url(phoenix_url) if project_name else None
    _print_header_footer(result, total_secs, ttft, model, trace_url, is_agentic=False)

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
    )
    parser.add_argument("-i", "--image-source", type=str, default=None)
    parser.add_argument("--project", type=str, default="chat-stream-obs")
    parser.add_argument("--phoenix-url", type=str, default=PHOENIX_URL)
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
    parser.add_argument("--generate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    from jet.adapters.llama_cpp.factory import get_llm_client

    args = get_args()

    parsed_logit_bias: dict[str, int] | None = None
    if args.logit_bias:
        try:
            parsed_logit_bias = json.loads(args.logit_bias)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid logit_bias JSON: {e}")
            raise SystemExit(1)

    parsed_tools: list[dict[str, Any]] | None = None
    if args.tools_json:
        try:
            parsed_tools = json.loads(args.tools_json)
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
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid response_format JSON: {e}")
            raise SystemExit(1)

    client = get_llm_client(base_url=args.base_url, timeout=args.timeout)

    if args.generate:
        result = run_generate_stream(
            args.prompt,
            model=args.model,
            project_name=args.project,
            phoenix_url=args.phoenix_url,
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
        result = run_chat_stream(
            args.prompt,
            model=args.model,
            project_name=args.project,
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
