"""Demo: Stream vision chat completion using a local image file.

Demonstrates:
  1. Local file reading with automatic MIME type detection from extension
  2. Base64 encoding of local image bytes
  3. Vision model streaming with Phoenix observability
  4. Structured StreamCompletionResult usage
"""

from __future__ import annotations

import logging
from pathlib import Path

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    get_client,
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

LOCAL_IMAGE_PATH = (
    "/Users/jethroestrada/Desktop/External_Projects/AI/curated/"
    "awesome-ai-apps/memory_agents/ai_consultant_agent/demo.png"
)


def main():
    setup_observability(project_name="vision-image-local-demo")
    client = get_client()

    # Verify file exists before sending request
    image_path = Path(LOCAL_IMAGE_PATH)
    if not image_path.exists():
        logger.error(f"❌ Image file not found: {LOCAL_IMAGE_PATH}")
        raise SystemExit(1)

    logger.info(f"📂 Analyzing local image: {LOCAL_IMAGE_PATH}")
    logger.info(f"   File size: {image_path.stat().st_size / 1024:.1f} KB")

    prompt = (
        "Analyze this screenshot or diagram. Describe what it shows, "
        "identify key components, and explain the overall purpose or workflow depicted."
    )

    result = run_chat_stream(
        prompt,
        client=client,
        image_source=str(image_path),
        model=MODEL,
        temperature=0.7,
        max_tokens=4096,
    )

    # ── Structured result inspection ───────────────────────────────────
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
