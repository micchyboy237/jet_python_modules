from __future__ import annotations

import argparse
import base64
import logging
import os
import time
from pathlib import Path

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

# ────────────────────────────────────────────────
# Logging (rich-formatted)
# ────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("vision-stream-merged")

# ────────────────────────────────────────────────
# Config (env vars, existing defaults preserved)
# ────────────────────────────────────────────────
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = "qwen3.5-uncensored:2b"
MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", DEFAULT_MODEL)
PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


# ────────────────────────────────────────────────
# Observability setup
# ────────────────────────────────────────────────
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
        # Valid enum values: NO_CONTENT, SPAN_ONLY, EVENT_ONLY, SPAN_AND_EVENT
        os.environ.setdefault(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
        )

    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{phoenix_url}/v1/traces",
        batch=False,  # flush immediately — good for short-lived scripts
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


# ────────────────────────────────────────────────
# Client
# ────────────────────────────────────────────────
def get_client(base_url: str = LLAMA_CPP_BASE_URL, timeout: float = 120.0) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key="sk-1234",  # dummy — llama.cpp/vLLM ignores it
        timeout=timeout,
    )


# ────────────────────────────────────────────────
# Image handling (local path or remote URL)
# ────────────────────────────────────────────────
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


# ────────────────────────────────────────────────
# Core streaming call
# ────────────────────────────────────────────────
def run_chat_stream_vl(
    client: OpenAI,
    image_source: str,
    prompt: str = "Describe this image in detail, including colors, objects, text, and overall scene.",
    model: str = MODEL,
    *,
    enable_thinking: bool = False,
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    phoenix_url: str = PHOENIX_URL,
) -> str:
    """
    Stream image analysis from a vision-capable llama.cpp/vLLM server.
    Wrapped in its own span so we can capture the trace_id and build a
    direct Phoenix UI link — OpenAIInstrumentor's auto-created span
    becomes a child of this one and shares the same trace_id.
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("vision_chat_stream") as span:
        trace_id = span.get_span_context().trace_id
        trace_url = build_phoenix_trace_url(phoenix_url, trace_id)
        span.set_attribute("llm.image_source", str(image_source))
        span.set_attribute("llm.model", model)

        logger.info("─" * 60)
        logger.info(f"🖼️  Image source : {image_source}")
        logger.info(f"🤖 Model        : {model}")
        logger.info(f"🔗 Trace URL    : [link={trace_url}]{trace_url}[/link]")

        t0 = time.perf_counter()
        base64_img, mime_type = encode_image_to_base64(image_source)
        logger.info(
            f"📦 Image encoded in {time.perf_counter() - t0:.2f}s ({mime_type})"
        )

        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_img}"},
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_content,
                ],
            }
        ]

        logger.info(f"➡️  Sending request (thinking={enable_thinking})")
        t_request_start = time.perf_counter()

        collected: list[str] = []
        usage = None

        try:
            stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body={
                    "top_k": top_k,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
                stream=True,
                stream_options={"include_usage": True},
            )

            in_think_block = False
            first_token_at: float | None = None

            console.print("[bold cyan]Response:[/bold cyan] ", end="")

            for chunk in stream:
                if not chunk.choices:
                    # Final usage-only chunk
                    usage = getattr(chunk, "usage", None)
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                if first_token_at is None and (
                    getattr(delta, "content", None)
                    or getattr(delta, "reasoning_content", None)
                ):
                    first_token_at = time.perf_counter()

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
                    collected.append(delta.reasoning_content)
                elif in_think_block:
                    console.print("[bold orange1]</think>[/bold orange1]", end="")
                    in_think_block = False

                if hasattr(delta, "content") and delta.content:
                    console.print(
                        f"[bold cyan]{delta.content}[/bold cyan]",
                        end="",
                        highlight=False,
                        soft_wrap=True,
                    )
                    collected.append(delta.content)

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
        full_response = "".join(collected)

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
            span.set_attribute("llm.usage.completion_tokens", usage.completion_tokens)
        if ttft is not None:
            logger.info(f"   Time to first token: {ttft:.2f}s")
        logger.info(f"   Total duration     : {total_secs:.2f}s")
        logger.info(f"   Response length    : {len(full_response)} chars")
        logger.info(f"🔗 View trace: [link={trace_url}]{trace_url}[/link]")
        logger.info("─" * 60)

        span.set_status(Status(StatusCode.OK))
        return full_response


# ────────────────────────────────────────────────
# CLI args
# ────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream vision-model chat completions with Phoenix observability."
    )
    parser.add_argument(
        "image_source",
        type=str,
        nargs="?",
        default="https://picsum.photos/800/600",
        help="Path or URL to the image to analyze.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Describe this image in detail: mention the main subjects, and any interesting details you notice.",
        help="Prompt for image analysis.",
    )

    # Observability
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

    # Model / server
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

    # Generation params
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable the model's reasoning/thinking output.",
    )
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=1.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

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
    run_chat_stream_vl(
        client,
        image_source=args.image_source,
        prompt=args.prompt,
        model=args.model,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        phoenix_url=args.phoenix_url,
    )
