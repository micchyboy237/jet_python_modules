"""Demo: Tool-calling agent loop with Phoenix observability.
Demonstrates:
  1. LLM decides to call a tool
  2. Automatic tool execution via tool_registry
  3. Multi-turn loop handled inside run_chat_stream
  4. Full trace visible in Phoenix
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'San Francisco' or 'Tokyo'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
}


def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    return {
        "temp": 28 if location.lower() == "tokyo" else 18,
        "condition": "humid" if location.lower() == "tokyo" else "foggy",
        "unit": unit,
        "location": location,
    }


TOOL_REGISTRY: dict[str, callable] = {
    "get_weather": get_weather,
}


def main():
    setup_observability(project_name="tool-calling-demo")
    client = get_llm_client()
    prompt = "What's the weather like in Tokyo right now? Use celsius."

    logger.info("🚀 Starting tool-calling demo with automatic execution")
    logger.info(f"   Prompt: {prompt}")
    logger.info(f"   Tools: {list(TOOL_REGISTRY.keys())}")

    result = run_chat_stream(
        prompt,
        client=client,
        model=MODEL,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        tool_registry=TOOL_REGISTRY,
        max_tool_rounds=5,
        temperature=0.0,
    )

    logger.info("")
    logger.info("═══ Final Result ═══")
    logger.info(f"📋 Finish reason : {result.finish_reason}")
    logger.info(f"📝 Content length: {len(result.content)} chars")
    if result.has_tool_calls:
        logger.info(f"🔧 Tool calls    : {len(result.tool_calls)}")
    if result.usage:
        logger.info(
            f"📊 Tokens: {result.usage['prompt_tokens']} prompt + "
            f"{result.usage['completion_tokens']} completion = "
            f"{result.usage['total_tokens']} total"
        )
    logger.info("")
    logger.info("✅ Tool-calling demo complete. Check Phoenix for full trace.")


if __name__ == "__main__":
    main()
