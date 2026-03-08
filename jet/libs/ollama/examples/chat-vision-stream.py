from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import requests
from openai import OpenAI
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
logger = logging.getLogger("ollama-vision-stream")

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_LLM_URL", "http://localhost:11434/v1")
MODEL = "ministral-3:3b"


def get_client() -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # dummy — ignored by Ollama
    )


def fetch_remote_image_bytes(url: str, headers: dict | None = None) -> bytes:
    """Fetch image bytes from a remote URL with better headers to avoid blocks."""
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
        response = requests.get(url, headers=headers, timeout=12)
        response.raise_for_status()
        return response.content
    except RequestException as exc:
        # Better debug info
        error_detail = ""
        if hasattr(exc, "response") and exc.response is not None:
            error_detail = (
                f" (status={exc.response.status_code}, reason={exc.response.reason})"
            )
        raise ValueError(
            f"Failed to fetch image from {url}{error_detail}: {exc}"
        ) from exc


def encode_image_to_base64(image_source: str | Path | bytes) -> tuple[str, str]:
    """
    Convert image (local path, remote URL, or bytes) to base64 string + mime type.
    Returns (base64_data, mime_type)
    """
    if isinstance(image_source, (str, Path)):
        source = str(image_source)
        if source.startswith(("http://", "https://")):
            img_bytes = fetch_remote_image_bytes(source)  # now with UA header
            # Improved MIME guessing (still fallback to jpeg)
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
        mime = "image/jpeg"  # fallback
    else:
        raise ValueError("image_source must be str/Path (local/remote) or bytes")

    base64_data = base64.b64encode(img_bytes).decode("utf-8")
    return base64_data, mime


def stream_analyze_image(
    client: OpenAI,
    image_source: str,  # local path or remote URL
    prompt: str = "Describe this image in detail, including colors, objects, text, and overall scene.",
    model: str = MODEL,
) -> str:
    """
    Stream image analysis from Ollama vision model (ministral-3b etc.).
    Handles both local files and remote URLs by converting to base64.
    Prints each chunk live + logs them. Returns full response.
    """
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

    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        stream=True,
        max_tokens=600,
        temperature=0.7,
    )

    full_response = ""
    console.print(
        f"[bold cyan]Streaming response from {model} analyzing image:[/bold cyan] ",
        end="",
    )

    for chunk in stream:
        if chunk.choices and (delta := chunk.choices[0].delta).content is not None:
            content = delta.content
            full_response += content

            # Live typewriter print
            print(content, end="", flush=True)

            # Structured logging of chunks (visible at DEBUG level)
            logger.debug("Chunk: %s", repr(content))

    console.print()  # final newline
    logger.info("[Stream complete] Full response length: %d chars", len(full_response))

    return full_response


# ────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    client = get_client()

    # Remote URL example (will be fetched → base64 → sent)
    stream_analyze_image(
        client,
        image_source="https://picsum.photos/800/600",  # Stable alternative: random high-quality photo from picsum.photos (no rate limits, direct hotlink OK)
        # For PNG alpha/transparency-specific test, alternative:
        # image_source="https://www.w3.org/Graphics/PNG/images/alphatest.png"  # Small W3C alpha test image with fading bars (real transparency)
        prompt="Describe this image in detail: mention the main subjects, colors, lighting, composition, and any interesting details you notice.",
    )

    # Local file example (unchanged behavior)
    # stream_analyze_image(
    #     client,
    #     image_source="~/Downloads/my_chart.png",
    #     prompt="Summarize the key data trends shown in this chart.",
    # )
