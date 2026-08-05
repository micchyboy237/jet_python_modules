"""Demo: Stream tool/function calling from llama.cpp with Phoenix observability.

Demonstrates defining tools, streaming the LLM's tool call decisions,
and executing the function locally. Note: This demo shows ONE turn;
for multi-turn agent loops, append the tool result message and re-call.
"""

import logging
from pathlib import Path
from typing import Any

from jet.libs.llama_cpp.usage.chat_stream_vl_observability import (
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

# Define tools compatible with llama.cpp's OpenAI API
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


def execute_get_weather(location: str, unit: str = "celsius") -> dict:
    """Simulated weather function execution."""
    # In production, call a real weather API here
    mock_data = {
        "san francisco": {"temp": 18, "condition": "foggy"},
        "tokyo": {"temp": 28, "condition": "humid"},
        "new york": {"temp": 22, "condition": "clear"},
    }
    result = mock_data.get(location.lower(), {"temp": 20, "condition": "unknown"})
    if unit == "fahrenheit":
        result["temp"] = round(result["temp"] * 9 / 5 + 32)
    result["unit"] = unit
    result["location"] = location
    return result


def main():
    setup_observability(project_name="tool-calling-demo")
    client = get_client()

    prompt = "What's the weather like in Tokyo right now? Use celsius."

    logger.info("🔧 Sending request with tool definitions...")
    # First turn: LLM decides to call a tool
    run_chat_stream(
        client,
        prompt=prompt,
        model=MODEL,
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        temperature=0.0,  # Deterministic for reliable tool selection
    )

    # NOTE: The above call streams the tool call decision.
    # To complete the agent loop, you would:
    # 1. Parse the tool call from the response (logged in trace)
    # 2. Execute the function locally
    # 3. Append assistant + tool messages to conversation
    # 4. Call run_chat_stream again with updated messages

    # Simulating what step 2-4 would look like:
    logger.info("\n🔄 Simulating tool execution and follow-up...")
    weather_result = execute_get_weather("Tokyo", "celsius")
    logger.info(f"   Tool result: {weather_result}")

    # For a full multi-turn demo, extend run_chat_stream to accept
    # pre-built message lists instead of just prompts.
    console.print(f"\n[bold green]🌤️  Weather in Tokyo:[/bold green] {weather_result}")


if __name__ == "__main__":
    main()
