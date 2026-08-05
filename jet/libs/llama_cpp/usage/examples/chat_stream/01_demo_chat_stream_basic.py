"""Demo: Basic text-only chat completion with Phoenix observability.
Demonstrates:
  1. Simple text-only chat streaming
  2. Phoenix observability integration
  3. Structured StreamCompletionResult usage
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


def main():
    # Setup observability
    setup_observability(project_name="chat-stream-basic-demo")

    # Initialize the client
    client = get_client()

    # Define the prompt
    prompt = "Write a 3 sentence romantic short story"

    # Run the chat stream
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        temperature=0.7,
        max_tokens=32768,
    )

    # Log results
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
