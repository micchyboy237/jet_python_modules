"""Demo: Basic text-only chat completion with Phoenix observability.
Demonstrates:
1. Simple text-only chat streaming
2. Phoenix observability integration (auto-initialized via project_name)
3. Structured StreamCompletionResult usage
4. Exported trace spans to JSONL
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.factory import get_llm_client
from jet.adapters.llama_cpp.llm_utils_observed import chat
from jet.libs.llama_cpp.usage.chat_stream_utils import MODEL
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

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    client = get_llm_client()
    prompt = "Write a 3 sentence romantic short story"

    result = chat(
        prompt,
        client=client,
        model=MODEL,
        project_name="chat-stream-basic-demo",
        temperature=0.7,
        max_tokens=16384,
        output_dir=OUTPUT_DIR,
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
