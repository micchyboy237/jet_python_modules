from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from opentelemetry import trace
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.trace import Status, StatusCode
from phoenix.otel import register
from requests.exceptions import RequestException
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("vision-stream-merged")

LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = "qwen3.5-uncensored:2b"
MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", DEFAULT_MODEL)
PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


# ──────────────────────────────────────────────────────────────────────────────
# Structured Result Data Classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCallResult:
    """A fully accumulated tool call from streaming deltas."""

    id: str
    type: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str  # Original JSON string for passthrough / debugging


@dataclass
class StreamCompletionResult:
    """Structured result from a streamed chat completion.

    Replaces raw `str` return so callers can programmatically inspect
    tool calls, usage, and finish reason without re-parsing.
    """

    content: str
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    usage: dict[str, int] | None = None
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Observability Setup
# ──────────────────────────────────────────────────────────────────────────────


def setup_observability(
    project_name: str = "vision-stream-obs",
    capture_content: bool = True,
    phoenix_url: str = PHOENIX_URL,
):
    """Configure OpenTelemetry to export traces to a remote Phoenix server.

    Uses phoenix.otel.register(), which sets the correct
    `openinference.project.name` resource attribute automatically —
    building the Resource by hand (e.g. with `service.name`) causes
    traces to silently land in the "default" project instead.
    """
    if capture_content:
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
        )

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{phoenix_url}/v1/traces",
        batch=False,
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

    logger.info(f"🔭 Observability enabled → [link={phoenix_url}]{phoenix_url}[/link]")
    logger.info(f"📁 Phoenix project name: {project_name}")
    return tracer_provider


def format_trace_id(trace_id: int) -> str:
    """Format an OTel trace id int as the 32-char hex string Phoenix expects."""
    return format(trace_id, "032x")


def build_phoenix_trace_url(phoenix_url: str, trace_id: int) -> str:
    """Direct link to view this specific trace in the Phoenix UI."""
    return f"{phoenix_url.rstrip('/')}/redirects/traces/{format_trace_id(trace_id)}"


# ──────────────────────────────────────────────────────────────────────────────
# Client & Image Helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_client(base_url: str = LLAMA_CPP_BASE_URL, timeout: float = 120.0) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="sk-1234",
        timeout=timeout,
    )


def fetch_remote_image_bytes(url: str, headers: dict | None = None) -> bytes:
    """Fetch image bytes from a remote URL with browser-like headers."""
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }
    headers = headers or default_headers
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except RequestException as exc:
        error_detail = ""
        if hasattr(exc, "response") and exc.response is not None:
            error_detail = (
                f" (status={exc.response.status_code}, reason={exc.response.reason})"
            )
        raise ValueError(
            f"Failed to fetch image from {url}{error_detail}: {exc}"
        ) from exc


def encode_image_to_base64(image_source: str | Path | bytes) -> tuple[str, str]:
    """Convert image (local path, remote URL, or bytes) to base64 string + mime type."""
    if isinstance(image_source, (str, Path)):
        source = str(image_source)
        if source.startswith(("http://", "https://")):
            img_bytes = fetch_remote_image_bytes(source)
            mime = "image/jpeg"
            lower = source.lower()
            if lower.endswith(".png"):
                mime = "image/png"
            elif lower.endswith((".jpg", ".jpeg")):
                mime = "image/jpeg"
            elif lower.endswith(".webp"):
                mime = "image/webp"
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


# ──────────────────────────────────────────────────────────────────────────────
# Encapsulated Tool Execution with Observability
# ──────────────────────────────────────────────────────────────────────────────


def execute_tool_with_span(
    tool_name: str,
    tool_arguments: dict[str, Any] | str,
    executor: Callable[..., Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Execute a tool function within its own observability span.

    Records tool name, arguments, result, duration, and any exceptions.
    Automatically links to the active parent chat span for end-to-end
    trace visibility in Phoenix.

    Args:
        tool_name: Name of the tool being executed.
        tool_arguments: Parsed dict or raw JSON string of arguments.
            Accepts str to handle cases where streaming accumulator
            hasn't pre-parsed the arguments yet.
        executor: Callable that implements the tool logic.
        strict: If True, raise on argument parse / execution failure.
            If False (default), return error dict instead of raising.

    Returns:
        Tool result dict, or error dict if strict=False and execution fails.
    """
    tracer = trace.get_tracer(__name__)

    # Normalize arguments: accept raw JSON string from stream accumulator
    if isinstance(tool_arguments, str):
        try:
            tool_arguments = json.loads(tool_arguments)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse tool arguments for {tool_name}: {e}")
            if strict:
                raise
            return {"error": f"Invalid JSON arguments: {e}", "tool": tool_name}

    with tracer.start_as_current_span(
        f"tool_execution.{tool_name}",
        attributes={
            "tool.name": tool_name,
            "tool.arguments": json.dumps(tool_arguments, default=str),
        },
    ) as span:
        t0 = time.perf_counter()
        logger.info(
            f"🔧 Executing tool: {tool_name}"
            f"({json.dumps(tool_arguments, default=str)[:120]})"
        )
        try:
            result = executor(**tool_arguments)
            duration = time.perf_counter() - t0
            span.set_attribute("tool.result", json.dumps(result, default=str))
            span.set_attribute("tool.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.OK))
            logger.info(
                f"   ✅ {tool_name} completed in {duration:.3f}s → "
                f"{json.dumps(result, default=str)[:150]}"
            )
            return result
        except TypeError as exc:
            # Catches mismatched kwargs (e.g., model hallucinated a param)
            duration = time.perf_counter() - t0
            error_msg = f"Argument mismatch: {exc}"
            span.set_attribute("tool.error", error_msg)
            span.set_attribute("tool.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.ERROR))
            logger.error(f"   ⚠️  {tool_name} argument error: {exc}")
            if strict:
                raise
            return {"error": error_msg, "tool": tool_name}
        except Exception as exc:
            duration = time.perf_counter() - t0
            span.record_exception(exc)
            span.set_attribute("tool.duration_s", round(duration, 4))
            span.set_status(Status(StatusCode.ERROR))
            logger.exception(f"   ❌ {tool_name} failed after {duration:.3f}s")
            if strict:
                raise
            return {"error": str(exc), "tool": tool_name}


# ──────────────────────────────────────────────────────────────────────────────
# Core Streaming Chat Completion
# ──────────────────────────────────────────────────────────────────────────────


def run_chat_stream(
    client: OpenAI,
    image_source: str | None = None,
    prompt: str = "What is OpenTelemetry in one sentence?",
    model: str = MODEL,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 32768,
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
    response_format: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    phoenix_url: str = PHOENIX_URL,
) -> StreamCompletionResult:
    """Stream a chat completion with full tool + structured output observability.

    Supports vision, function calling, JSON mode, grammar-constrained output,
    and all llama.cpp-compatible sampling parameters.

    If `messages` is provided, it takes precedence over `prompt`/`image_source`.
    This enables multi-turn agent loops where tool results are appended and
    the function is called again with the updated conversation history.

    Returns:
        StreamCompletionResult with content, parsed tool calls, usage, and
        finish reason — replacing raw str for programmatic consumption.
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("vision_chat_stream") as span:
        trace_id = span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)

        # ── Record all parameters on span for observability ────────────
        span.set_attribute("llm.model", model)
        span.set_attribute(
            "llm.image_source", str(image_source) if image_source else "none"
        )
        span.set_attribute("llm.sampling.temperature", temperature)
        span.set_attribute("llm.sampling.top_p", top_p)
        span.set_attribute("llm.sampling.top_k", top_k)
        span.set_attribute("llm.sampling.min_p", min_p)
        span.set_attribute("llm.sampling.repeat_penalty", repeat_penalty)
        span.set_attribute("llm.sampling.presence_penalty", presence_penalty)
        span.set_attribute("llm.sampling.frequency_penalty", frequency_penalty)
        span.set_attribute("llm.sampling.max_tokens", max_tokens)
        span.set_attribute("llm.sampling.enable_thinking", enable_thinking)
        if seed is not None:
            span.set_attribute("llm.sampling.seed", seed)
        if logit_bias:
            span.set_attribute("llm.sampling.logit_bias", json.dumps(logit_bias))
        if stop:
            span.set_attribute("llm.sampling.stop_sequences", json.dumps(stop))
        if tools:
            span.set_attribute("llm.tools.count", len(tools))
            span.set_attribute(
                "llm.tools.names",
                json.dumps([t.get("function", {}).get("name") for t in tools]),
            )
        if response_format:
            span.set_attribute(
                "llm.response_format.type",
                response_format.get("type", "unknown"),
            )

        # ── Startup logs ───────────────────────────────────────────────
        logger.info("─" * 60)
        logger.info(f"🖼️  Image source : {image_source or '(none — text-only)'}")
        logger.info(f"🤖 Model        : {model}")
        logger.info(
            f"🎛️  Sampling     : temp={temperature} top_p={top_p} top_k={top_k} "
            f"min_p={min_p} rep_pen={repeat_penalty}"
        )
        logger.info(
            f"   freq_pen={frequency_penalty} pres_pen={presence_penalty} "
            f"seed={seed} stop={stop}"
        )
        if logit_bias:
            logger.info(f"   logit_bias={logit_bias}")
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            logger.info(f"🔧 Tools        : {tool_names} (choice={tool_choice})")
        if response_format:
            logger.info(f"📐 Response fmt : {response_format}")
        logger.info(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        # ── Build messages ─────────────────────────────────────────────
        if messages is not None:
            logger.info(
                f"📨 Using pre-built message history ({len(messages)} messages)"
            )
        elif image_source:
            t0 = time.perf_counter()
            base64_img, mime_type = encode_image_to_base64(image_source)
            logger.info(
                f"📦 Image encoded in {time.perf_counter() - t0:.2f}s ({mime_type})"
            )
            content: Any = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
                },
            ]
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # ── Build extra_body for llama.cpp-specific params ─────────────
        extra_body_params: dict[str, Any] = {
            "top_k": top_k,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if min_p > 0.0:
            extra_body_params["min_p"] = min_p
        if repeat_penalty != 1.1:
            extra_body_params["repeat_penalty"] = repeat_penalty

        # ── Build API kwargs ───────────────────────────────────────────
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
            "extra_body": extra_body_params,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            api_kwargs["tools"] = tools
        if tool_choice is not None:
            api_kwargs["tool_choice"] = tool_choice
        if response_format:
            api_kwargs["response_format"] = response_format

        logger.info(
            f"➡️  Sending request (thinking={enable_thinking}, "
            f"tools={bool(tools)}, format={response_format})"
        )
        t_request_start = time.perf_counter()

        collected_content: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        usage = None
        first_token_at: float | None = None
        finish_reason: str | None = None

        try:
            stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                **api_kwargs
            )

            in_think_block = False

            console.print("[bold cyan]Response:[/bold cyan] ", end="")

            for chunk in stream:
                if not chunk.choices:
                    usage = getattr(chunk, "usage", None)
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                # Capture finish reason from final chunk
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                if first_token_at is None and (
                    getattr(delta, "content", None)
                    or getattr(delta, "reasoning_content", None)
                    or getattr(delta, "tool_calls", None)
                ):
                    first_token_at = time.perf_counter()

                # Handle reasoning/thinking tokens
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not in_think_block:
                        console.print("[bold orange1]<think>[/bold orange1]", end="")
                        in_think_block = True
                    console.print(
                        f"[bold orange1]{delta.reasoning_content}[/bold orange1]",
                        end="",
                        highlight=False,
                        soft_wrap=True,
                    )
                    collected_content.append(delta.reasoning_content)
                elif in_think_block:
                    console.print("[bold orange1]</think>[/bold orange1]", end="")
                    in_think_block = False

                # Handle regular content tokens
                if hasattr(delta, "content") and delta.content:
                    console.print(
                        f"[bold cyan]{delta.content}[/bold cyan]",
                        end="",
                        highlight=False,
                        soft_wrap=True,
                    )
                    collected_content.append(delta.content)

                # Handle streamed tool call deltas (accumulate across chunks)
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

            if in_think_block:
                console.print("[bold orange1]</think>[/bold orange1]", end="")

        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            logger.exception("❌ Streaming failed")
            raise
        finally:
            console.print()
            total_secs = time.perf_counter() - t_request_start
            ttft = (first_token_at - t_request_start) if first_token_at else None
            full_response = "".join(collected_content)

            # Build structured tool call results
            parsed_tool_calls: list[ToolCallResult] = []
            if tool_calls_acc:
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    fn = tc["function"]
                    try:
                        parsed_args = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        parsed_args = {}
                    parsed_tool_calls.append(
                        ToolCallResult(
                            id=tc.get("id", ""),
                            type=tc.get("type", "function"),
                            name=fn.get("name", ""),
                            arguments=parsed_args,
                            raw_arguments=fn.get("arguments", ""),
                        )
                    )

            # Log accumulated tool calls
            if parsed_tool_calls:
                logger.info(f"🔧 Tool calls received: {len(parsed_tool_calls)}")
                for tc in parsed_tool_calls:
                    args_preview = tc.raw_arguments[:120]
                    if len(tc.raw_arguments) > 120:
                        args_preview += "..."
                    logger.info(f"   → {tc.name}({args_preview})")
                span.set_attribute(
                    "llm.tool_calls",
                    json.dumps(
                        [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "name": tc.name,
                                "arguments": tc.raw_arguments,
                            }
                            for tc in parsed_tool_calls
                        ],
                        default=str,
                    ),
                )

            # ── Completion summary ─────────────────────────────────────
            logger.info("─" * 60)
            logger.info("📊 Completion summary")
            if usage:
                tok_per_sec = (
                    usage.completion_tokens / total_secs if total_secs > 0 else 0.0
                )
                logger.info(f"   Prompt tokens      : {usage.prompt_tokens}")
                logger.info(f"   Completion tokens  : {usage.completion_tokens}")
                logger.info(f"   Total tokens       : {usage.total_tokens}")
                logger.info(f"   Throughput         : {tok_per_sec:.1f} tok/s")
                span.set_attribute("llm.usage.prompt_tokens", usage.prompt_tokens)
                span.set_attribute(
                    "llm.usage.completion_tokens", usage.completion_tokens
                )
            if ttft is not None:
                logger.info(f"   Time to first token: {ttft:.2f}s")
            logger.info(f"   Total duration     : {total_secs:.2f}s")

            # Distinguish between content responses and tool-call-only responses
            if parsed_tool_calls:
                logger.info(
                    f"   Response type      : tool_calls ({len(parsed_tool_calls)} call(s))"
                )
            else:
                logger.info(f"   Response length    : {len(full_response)} chars")

            if finish_reason:
                logger.info(f"   Finish reason      : {finish_reason}")

            logger.info(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
            logger.info("─" * 60)
            span.set_status(Status(StatusCode.OK))

        return StreamCompletionResult(
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


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream vision-model chat completions with Phoenix observability."
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
        help="Path or URL to an image to analyze. Omit for a text-only chat request.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="vision-stream-obs",
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
        help="Disable capturing prompt/response text in traces (metadata only).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=LLAMA_CPP_BASE_URL,
        help="OpenAI-compatible server base URL (env: LLAMA_CPP_VISION_URL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="Model name to request (env: LLAMA_CPP_VISION_MODEL).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Client request timeout in seconds.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model's reasoning/thinking output.",
    )
    # ── Standard OpenAI sampling params (supported by llama.cpp) ──────
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Penalize tokens based on frequency (-2.0 to 2.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--stop",
        type=str,
        nargs="+",
        default=None,
        help="Stop sequences (up to 4).",
    )
    parser.add_argument(
        "--logit-bias",
        type=str,
        default=None,
        help="JSON dict of token_id:bias pairs, e.g. '{\"1234\": -100}'",
    )
    # ── llama.cpp-specific params (via extra_body) ────────────────────
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Minimum probability threshold (llama.cpp specific).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (llama.cpp native parameter).",
    )
    # ── Tools & Structured Output ─────────────────────────────────────
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
        help='Tool choice: "auto", "none", "required", or JSON object.',
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default=None,
        help='JSON response format, e.g. \'{"type": "json_object"}\'.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # ── Parse JSON CLI arguments ───────────────────────────────────────
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
            pass  # Keep as string literal ("auto", "none", "required")

    parsed_response_format: dict[str, Any] | None = None
    if args.response_format:
        try:
            parsed_response_format = json.loads(args.response_format)
            logger.info(f"📐 Response format: {parsed_response_format}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid response_format JSON: {e}")
            raise SystemExit(1)

    # ── Startup config ─────────────────────────────────────────────────
    logger.info("🚀 Startup config")
    logger.info(f"   Base URL     : {args.base_url}")
    logger.info(f"   Model        : {args.model}")
    logger.info(f"   Phoenix URL  : {args.phoenix_url}")
    logger.info(f"   Project      : {args.project}")

    setup_observability(
        project_name=args.project,
        capture_content=args.capture_content,
        phoenix_url=args.phoenix_url,
    )

    client = get_client(base_url=args.base_url, timeout=args.timeout)

    result = run_chat_stream(
        client,
        image_source=args.image_source,
        prompt=args.prompt,
        model=args.model,
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
        phoenix_url=args.phoenix_url,
    )

    # CLI prints structured result info
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
