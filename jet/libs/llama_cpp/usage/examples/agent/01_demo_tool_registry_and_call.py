"""Demo: Stateful Agent Class with Tool Registry and Unified Tracing.

Demonstrates:
  1. Instantiating a stateful Agent class
  2. Registering tools dynamically
  3. Running multi-turn conversations with automatic history management
  4. Subclassing the Agent to add custom hooks (e.g., logging/approvals)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jet.libs.llama_cpp.usage.agent import Agent
from jet.libs.llama_cpp.usage.chat_stream_observability import (
    MODEL,
    get_client,
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

# --- 1. Define Tool Schemas & Implementations ---
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
    }


# --- 2. Create a Custom Agent Subclass (Optional but powerful) ---
class VerboseAgent(Agent):
    """Custom agent that adds extra logging when tools are called."""

    def on_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        console.print(
            f"[bold magenta]🛑 INTERCEPTED TOOL CALL:[/bold magenta] {tool_name}({arguments})"
        )
        # Call parent implementation to actually execute the tool
        return super().on_tool_call(tool_name, arguments)


def main():
    setup_observability(project_name="tool-registry-and-call-demo")
    client = get_client()

    # Instantiate the custom agent
    agent = VerboseAgent(
        client=client,
        model=MODEL,
        max_turns=3,
        system_prompt="You are a helpful travel assistant. Always use tools to get weather data.",
        temperature=0.0,
    )

    # Register tools
    agent.register_tool(WEATHER_TOOL, get_weather)

    # --- Turn 1 ---
    logger.info("🚀 Starting Turn 1")
    res1 = agent.run(prompt="What's the weather like in Tokyo right now? Use celsius.")
    console.print(f"\n[bold cyan]Agent Turn 1 Response:[/bold cyan] {res1.content}")

    # --- Turn 2 (Continuing the conversation using internal history) ---
    logger.info("🚀 Starting Turn 2 (Follow-up)")
    res2 = agent.run(
        prompt="Thanks! Based on that, what should I pack for my trip there tomorrow?"
    )
    console.print(f"\n[bold cyan]Agent Turn 2 Response:[/bold cyan] {res2.content}")

    logger.info(f"📊 Total messages in agent history: {len(agent.history)}")


if __name__ == "__main__":
    main()
