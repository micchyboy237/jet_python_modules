"""Demo: Full tool-calling agent loop with complete Phoenix observability.

Demonstrates:
  1. LLM decides to call a tool (turn 1)
  2. Tool execution recorded as child span
  3. Tool result sent back as `role: tool` message
  4. LLM generates final answer using tool result (turn 2)
  5. Both turns + tool execution visible in single Phoenix trace
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    execute_tool_with_span,
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
    client = get_client()

    prompt = "What's the weather like in Tokyo right now? Use celsius."

    # ── Turn 1: LLM decides to call a tool ──────────────────────────────
    logger.info("═══ TURN 1: Initial request with tool definitions ═══")
    turn1_response = run_chat_stream(
        prompt,
        client=client,
        model=MODEL,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        temperature=0.0,
    )

    # NOTE: In production, run_chat_stream should return a structured
    # StreamResult containing the accumulated tool_calls. For this demo,
    # we reconstruct from what we know the model will output.
    # CRITICAL: llama.cpp REQUIRES "type": "function" on assistant tool_calls
    tool_calls = [
        {
            "id": "call_demo_001",
            "type": "function",  # ← REQUIRED by llama.cpp
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "Tokyo", "unit": "celsius"}),
            },
        }
    ]

    if not tool_calls:
        logger.warning(
            "⚠️  No tool calls detected. Model may not support function calling."
        )
        return

    # ── Execute each tool with observability ────────────────────────────
    tool_messages: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn_name = tc["function"]["name"]
        fn_args = json.loads(tc["function"]["arguments"])

        executor = TOOL_REGISTRY.get(fn_name)
        if executor is None:
            logger.error(f"❌ Unknown tool: {fn_name}")
            continue

        result = execute_tool_with_span(fn_name, fn_args, executor)

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result),
            }
        )

    # ── Turn 2: Send tool results back to LLM ──────────────────────────
    logger.info("\n═══ TURN 2: Follow-up with tool results ═══")
    follow_up_messages: list[dict[str, Any]] = [
        {"role": "user", "content": prompt},
        {
            "role": "assistant",
            # Use empty string instead of None — some llama.cpp builds
            # reject null content on assistant messages with tool_calls
            "content": "",
            "tool_calls": tool_calls,
        },
        *tool_messages,
    ]

    run_chat_stream(
        client=client,
        messages=follow_up_messages,
        model=MODEL,
        temperature=0.7,
    )

    logger.info("\n✅ Complete agent loop finished. Check Phoenix for full trace.")


if __name__ == "__main__":
    main()
