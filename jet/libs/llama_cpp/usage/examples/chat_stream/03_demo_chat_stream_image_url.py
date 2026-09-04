"""Demo: Stream vision chat completion using a remote image URL.
Demonstrates:
  1. Remote image fetching with browser-like headers
  2. Base64 encoding and MIME type detection from URL extension
  3. Vision model streaming with Phoenix observability
  4. Structured StreamCompletionResult usage
"""

from __future__ import annotations

import logging
from pathlib import Path

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    run_chat_stream,
    setup_observability,
)
from rich.console import Console
from rich.logging import RichHandler

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(Path(__file__).stem)

IMAGE_URL = "https://picsum.photos/800/600"


def main():
    setup_observability(project_name="vision-image-url-demo")
    client = get_llm_client()
    prompt = (
        "Describe this image in detail. Include colors, objects, composition, "
        "and any text visible. Be specific and thorough."
    )
    logger.info(f"🌐 Analyzing remote image: {IMAGE_URL}")
    result = run_chat_stream(
        prompt,
        client=client,
        image_source=IMAGE_URL,
        model=MODEL,
        temperature=0.7,
        max_tokens=4096,
    )
    logger.info(f"📋 Finish reason: {result.finish_reason}")
    if result.usage:
        logger.info(
            f"📊 Tokens: {result.usage['prompt_tokens']} prompt + "
            f"{result.usage['completion_tokens']} completion = "
            f"{result.usage['total_tokens']} total"
        )
    logger.info(f"📝 Response length: {len(result.content)} chars")


if __name__ == "__main__":
    main()
