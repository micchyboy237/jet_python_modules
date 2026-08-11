"""Demo: Automatic tool execution via tool_registry with Phoenix observability.

Demonstrates:
  1. Passing a tool_registry dict to run_chat_stream for automatic execution
  2. Multi-turn agent loop handled entirely inside run_chat_stream
  3. Tool execution spans nested under tool_execution_loop parent span
  4. Graceful handling of unknown tools (error returned to model, no crash)
  5. Single Phoenix trace showing full LLM → tool → LLM round-trip
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


# ──────────────────────────────────────────────────────────────────────────────
# Tool Definitions & Implementations
# ──────────────────────────────────────────────────────────────────────────────

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a given city.",
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
                    "description": "Temperature unit. Defaults to celsius.",
                },
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a simple arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    },
}


def get_weather(location: str, unit: str = "celsius") -> dict[str, Any]:
    """Simulated weather lookup."""
    data = {
        "tokyo": {"temp": 28, "condition": "humid"},
        "san francisco": {"temp": 14, "condition": "foggy"},
        "london": {"temp": 11, "condition": "rainy"},
    }
    info = data.get(location.lower(), {"temp": 20, "condition": "clear"})
    return {
        "location": location,
        "temp": info["temp"],
        "condition": info["condition"],
        "unit": unit,
    }


def calculate(expression: str) -> dict[str, Any]:
    """Safe arithmetic evaluator (demo only — never use eval in production)."""
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return {"error": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression)  # noqa: S307 — demo-only sandboxed check above
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}


# Registry maps tool names → callables. This is the single integration point.
TOOL_REGISTRY: dict[str, Any] = {
    "get_weather": get_weather,
    "calculate": calculate,
}

TOOLS = [WEATHER_TOOL, CALCULATOR_TOOL]


# ──────────────────────────────────────────────────────────────────────────────
# Demo Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def main():
    setup_observability(project_name="tool-registry-auto-demo")
    client = get_client()

    # Prompt designed to trigger at least one tool call
    prompt = (
        "What's the weather in Tokyo right now in celsius? Also, what is 28 * 3 + 15?"
    )

    logger.info("🚀 Starting auto-tool-execution demo")
    logger.info(f"   Prompt: {prompt}")
    logger.info(f"   Tools registered: {list(TOOL_REGISTRY.keys())}")
    logger.info("")

    # The key difference from demo 04: pass tool_registry directly.
    # run_chat_stream handles the full LLM → tool → LLM loop internally.
    result = run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        tools=TOOLS,
        tool_choice="auto",
        tool_registry=TOOL_REGISTRY,
        max_tool_rounds=5,
        temperature=0.0,
        max_tokens=4096,
    )

    logger.info("")
    logger.info("═══ Final Result ═══")
    logger.info(f"📋 Finish reason : {result.finish_reason}")
    logger.info(f"📝 Content length: {len(result.content)} chars")
    if result.usage:
        logger.info(
            f"📊 Tokens: {result.usage['prompt_tokens']} prompt + "
            f"{result.usage['completion_tokens']} completion = "
            f"{result.usage['total_tokens']} total"
        )
    logger.info("")
    logger.info("✅ Auto-tool demo complete. Check Phoenix for full trace hierarchy:")
    logger.info(
        "   tool_execution_loop → chat_stream (per round) → tool_execution.<name>"
    )


if __name__ == "__main__":
    main()
