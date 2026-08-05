from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import requests
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openinference.semconv.resource import ResourceAttributes

# Observability imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from requests.exceptions import RequestException
from rich.console import Console
from rich.logging import RichHandler

# ────────────────────────────────────────────────
# Setup rich logging
# ────────────────────────────────────────────────
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger("vision-stream-obs")

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_VISION_URL", "http://localhost:8080/v1")
DEFAULT_MODEL = "qwen3.5-uncensored:2b"
MODEL = os.getenv("LLAMA_CPP_VISION_MODEL", DEFAULT_MODEL)
PHOENIX_URL = os.getenv("LLM_OBS_PHOENIX_URL", "http://localhost:6006")


# ────────────────────────────────────────────────
# Observability Setup
# ────────────────────────────────────────────────
def setup_observability(project_name: str = "vision-stream-obs"):
    """Configure OpenTelemetry to export traces to remote Phoenix server."""
    endpoint = f"{PHOENIX_URL}/v1/traces"
    resource = Resource.create(
        {
            ResourceAttributes.PROJECT_NAME: project_name,  # ← was "service.name"
        }
    )
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    OpenAIInstrumentor().instrument(disable_content_capture=True)
    logger.info(f"🔭 Observability enabled → [link={PHOENIX_URL}]{PHOENIX_URL}[/link]")
    logger.info(f"📁 Phoenix project name: {project_name}")


# ────────────────────────────────────────────────
# Client & Image Helpers
# ────────────────────────────────────────────────
def get_client() -> OpenAI:
    return OpenAI(
        base_url=LLAMA_CPP_BASE_URL,
        api_key="sk-1234",  # Dummy key — llama.cpp ignores it
        timeout=120.0,
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
            if lower.endswith((".png", ".PNG")):
                mime = "image/png"
            elif lower.endswith((".jpg", ".jpeg", ".JPG", ".JPEG")):
                mime = "image/jpeg"
            elif lower.endswith((".webp", ".WEBP")):
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
# Streaming Vision Chat
# ────────────────────────────────────────────────
def run_chat_stream_vl(
    client: OpenAI,
    image_source: str,
    prompt: str = "Describe this image in detail, including colors, objects, text, and overall scene.",
    model: str = MODEL,
) -> str:
    base64_img, mime_type = encode_image_to_base64(image_source)

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

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        max_tokens=32768,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={"top_k": 20},
        stream=True,
    )

    full_response = ""
    console.print(
        f"[bold cyan]Streaming response from {model} analyzing image:[/bold cyan] ",
        end="",
    )

    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            if hasattr(delta, "reasoning_content") and getattr(
                delta, "reasoning_content"
            ):
                content = delta.reasoning_content
                full_response += content
                console.print(
                    f"[bold orange1]{content}[/bold orange1]",
                    end="",
                    highlight=False,
                    soft_wrap=True,
                )
            elif hasattr(delta, "content") and getattr(delta, "content"):
                content = delta.content
                full_response += content
                console.print(
                    f"[bold cyan]{content}[/bold cyan]",
                    end="",
                    highlight=False,
                    soft_wrap=True,
                )

        usage = getattr(part, "usage", None)
        if usage is not None:
            logger.info("\n\n=== Completion Details (llama.cpp aligned) ===")
            logger.info(f"Prompt tokens     : {usage.prompt_tokens}")
            logger.info(f"Completion tokens : {usage.completion_tokens}")
            logger.info(f"Total tokens      : {usage.total_tokens}")

    console.print()
    logger.info("[Stream complete] Full response length: %d chars", len(full_response))
    return full_response


# ────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    # Initialize observability FIRST, before client creation
    setup_observability()

    parser = argparse.ArgumentParser(
        description="Stream image analysis from llama.cpp server with Phoenix observability."
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
    args = parser.parse_args()

    client = get_client()
    run_chat_stream_vl(
        client,
        image_source=args.image_source,
        prompt=args.prompt,
    )

    # Local file example (unchanged behavior)
    # run_chat_stream_vl(
    #     client,
    #     image_source="~/Downloads/my_chart.png",
    #     prompt="Summarize the key data trends shown in this chart.",
    # )
