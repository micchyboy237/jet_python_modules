"""Demo: Human-in-the-Loop Agent with Tool Approval.

Demonstrates:
  1. InteractiveApproval for terminal-based approval.
  2. CallbackApproval for automated decisions via callback.
  3. AutoApproval for no human involvement (default).
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
from jet.libs.llama_cpp.usage.human_in_the_loop import (
    AutoApproval,
    CallbackApproval,
    InteractiveApproval,
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
    """Mock weather function."""
    return {
        "temp": 28 if location.lower() == "tokyo" else 18,
        "condition": "humid" if location.lower() == "tokyo" else "foggy",
        "unit": unit,
    }


def custom_approval_callback(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Custom logic for approving/rejecting tool calls."""
    console.print(
        f"\n[bold yellow]🔍 Custom Approval Callback:[/bold yellow] {tool_name}({arguments})"
    )
    if tool_name == "delete_file":
        console.print("[bold red]❌ Rejected:[/bold red] 'delete_file' is not allowed.")
        return False
    if tool_name == "get_weather":
        allowed_locations = ["tokyo", "new york", "london"]
        location = arguments.get("location", "").lower()
        if location in allowed_locations:
            console.print(
                f"[bold green]✅ Auto-approved:[/bold green] {tool_name} for {location}"
            )
            return True
        else:
            console.print(
                f"[bold red]❌ Rejected:[/bold red] Unknown location: {location}"
            )
            return False
    console.print(f"[bold green]✅ Auto-approved:[/bold green] {tool_name}")
    return True


def demo_interactive_approval():
    """Demo: Interactive approval for tool calls."""
    console.print("\n" + "=" * 60)
    console.print("[bold blue]🎯 Demo 1: Interactive Approval[/bold blue]")
    console.print("=" * 60)
    setup_observability(project_name="human-in-the-loop-interactive")
    client = get_client()

    # Use InteractiveApproval strategy
    agent = Agent(
        client=client,
        model=MODEL,
        max_turns=3,
        system_prompt="You are a helpful assistant. Use tools to fetch data.",
        approval=InteractiveApproval(),
    )
    agent.register_tool(WEATHER_TOOL, get_weather)
    console.print(
        "\n[bold cyan]🚀 Running agent with interactive approval...[/bold cyan]"
    )
    console.print("[dim]Type 'y' to approve tool calls or 'n' to reject.[/dim]\n")
    result = agent.run(prompt="What's the weather like in Tokyo? Use celsius.")
    console.print(f"\n[bold green]✅ Result:[/bold green] {result.content}")


def demo_custom_approval_callback():
    """Demo: Custom approval callback for automated decisions."""
    console.print("\n" + "=" * 60)
    console.print("[bold blue]🎯 Demo 2: Custom Approval Callback[/bold blue]")
    console.print("=" * 60)
    setup_observability(project_name="human-in-the-loop-custom-callback")
    client = get_client()

    # Use CallbackApproval strategy with custom logic
    agent = Agent(
        client=client,
        model=MODEL,
        max_turns=3,
        system_prompt="You are a helpful assistant. Use tools to fetch data.",
        approval=CallbackApproval(custom_approval_callback),
    )
    agent.register_tool(WEATHER_TOOL, get_weather)
    console.print(
        "\n[bold cyan]🚀 Running agent with custom approval callback...[/bold cyan]\n"
    )
    result = agent.run(prompt="What's the weather like in Tokyo? Use celsius.")
    console.print(f"\n[bold green]✅ Result (Tokyo):[/bold green] {result.content}")
    result = agent.run(prompt="What's the weather like in Mars? Use celsius.")
    console.print(f"\n[bold green]✅ Result (Mars):[/bold green] {result.content}")


def demo_no_approval():
    """Demo: Agent without approval (default behavior)."""
    console.print("\n" + "=" * 60)
    console.print("[bold blue]🎯 Demo 3: No Approval (Default)[/bold blue]")
    console.print("=" * 60)
    setup_observability(project_name="human-in-the-loop-no-approval")
    client = get_client()

    # Use AutoApproval strategy (explicit, or omit for default)
    agent = Agent(
        client=client,
        model=MODEL,
        max_turns=3,
        system_prompt="You are a helpful assistant. Use tools to fetch data.",
        approval=AutoApproval(),
    )
    agent.register_tool(WEATHER_TOOL, get_weather)
    console.print("\n[bold cyan]🚀 Running agent without approval...[/bold cyan]\n")
    result = agent.run(prompt="What's the weather like in Tokyo? Use celsius.")
    console.print(f"\n[bold green]✅ Result:[/bold green] {result.content}")


def main():
    """Run all demos."""
    demo_interactive_approval()
    demo_custom_approval_callback()
    demo_no_approval()


if __name__ == "__main__":
    main()
