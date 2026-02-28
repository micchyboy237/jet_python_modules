# demo_different_models_local.py
"""
Demonstrations of using different model backends with smolagents

Shows:
- Local llama.cpp (OpenAI-compatible)
- OpenRouter (OpenAI-compatible remote)
- Google Gemini (via OpenAI-compatible)
- xAI Grok (via LiteLLM)
"""

import os
import time
from typing import Callable, Optional

from jet.libs.smolagents.utils.model_utils import create_local_model
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from smolagents import REMOVE_PARAMETER, CodeAgent, LiteLLMModel, OpenAIModel

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Other model factories (optional / env-var driven)
# ──────────────────────────────────────────────────────────────────────────────


def create_openrouter_model(
    model_id: str = "openai/gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
) -> OpenAIModel:
    """OpenRouter – requires OPENROUTER_API_KEY env var."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    return OpenAIModel(
        model_id=model_id,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_gemini_model(
    model_id: str = "gemini-1.5-flash",
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
) -> OpenAIModel:
    """Google Gemini via OpenAI-compatible endpoint – requires GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    return OpenAIModel(
        model_id=model_id,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_grok_model(
    model_name: str = "grok-beta",  # or "grok-4", "grok-3-mini"
    temperature: float = 0.7,
    max_tokens: Optional[int] = 1024,
) -> LiteLLMModel:
    """xAI Grok via LiteLLM – requires XAI_API_KEY."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise ValueError("XAI_API_KEY environment variable not set")

    return LiteLLMModel(
        model_id=f"xai/{model_name}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=REMOVE_PARAMETER,  # Grok models do not support stop sequences
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reusable agent factory
# ──────────────────────────────────────────────────────────────────────────────


def create_simple_agent(
    model_factory: Callable,
    max_steps: int = 6,
    verbosity_level: int = 1,
) -> CodeAgent:
    """Creates a minimal agent using the provided model factory."""
    model = model_factory()
    return CodeAgent(
        tools=[],
        model=model,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Common task used in all demos
# ──────────────────────────────────────────────────────────────────────────────

COMMON_TASK = "What is the 12th Fibonacci number? Please compute it step by step."


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────


def demo_model_1_local_llamacpp():
    """Demo 1: Your local llama.cpp server (default)"""
    console.rule("Demo 1 – Local llama.cpp", style="blue")

    agent = create_simple_agent(create_local_model, max_steps=5)

    console.print(f"\n[bold cyan]Task:[/bold cyan] {COMMON_TASK}")
    start = time.time()

    try:
        result = agent.run(COMMON_TASK)
        console.print(Panel(result, title="Answer", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
    finally:
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")


def demo_model_2_openrouter():
    """Demo 2: OpenRouter (fast & cheap models)"""
    console.rule("Demo 2 – OpenRouter", style="blue")

    try:
        agent = create_simple_agent(
            lambda: create_openrouter_model(model_id="anthropic/claude-3.5-haiku"),
            max_steps=5,
        )

        console.print(f"\n[bold cyan]Task:[/bold cyan] {COMMON_TASK}")
        start = time.time()
        result = agent.run(COMMON_TASK)
        console.print(Panel(result, title="Answer", border_style="green"))
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")

    except ValueError as e:
        console.print(f"[yellow]Skipped: {str(e)}[/yellow]")
        console.print("[dim]Set OPENROUTER_API_KEY env var to enable[/dim]")


def demo_model_3_gemini():
    """Demo 3: Google Gemini via OpenAI-compatible API"""
    console.rule("Demo 3 – Google Gemini", style="blue")

    try:
        agent = create_simple_agent(
            lambda: create_gemini_model(model_id="gemini-1.5-flash"),
            max_steps=5,
        )

        console.print(f"\n[bold cyan]Task:[/bold cyan] {COMMON_TASK}")
        start = time.time()
        result = agent.run(COMMON_TASK)
        console.print(Panel(result, title="Answer", border_style="green"))
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")

    except ValueError as e:
        console.print(f"[yellow]Skipped: {str(e)}[/yellow]")
        console.print("[dim]Set GEMINI_API_KEY env var to enable[/dim]")


def demo_model_4_grok():
    """Demo 4: xAI Grok via LiteLLM"""
    console.rule("Demo 4 – xAI Grok", style="blue")

    try:
        agent = create_simple_agent(
            lambda: create_grok_model(model_name="grok-beta"),
            max_steps=5,
        )

        console.print(f"\n[bold cyan]Task:[/bold cyan] {COMMON_TASK}")
        start = time.time()
        result = agent.run(COMMON_TASK)
        console.print(Panel(result, title="Answer", border_style="green"))
        console.print(f"[dim]Duration: {time.time() - start:.1f}s[/dim]")

    except ValueError as e:
        console.print(f"[yellow]Skipped: {str(e)}[/yellow]")
        console.print("[dim]Set XAI_API_KEY env var to enable[/dim]")


def main():
    console.rule("Different Model Backends – smolagents Demos", style="bold magenta")

    console.print(
        "[dim]Each demo runs the same simple task with a different model provider[/dim]\n"
        "[dim]Only local llama.cpp is guaranteed to work without API keys[/dim]\n"
    )

    table = Table(title="Available Demos")
    table.add_column("Demo", style="cyan")
    table.add_column("Model Type", style="magenta")
    table.add_column("Requires Env Var")
    table.add_row("1", "Local llama.cpp", "—")
    table.add_row("2", "OpenRouter", "OPENROUTER_API_KEY")
    table.add_row("3", "Google Gemini", "GEMINI_API_KEY")
    table.add_row("4", "xAI Grok", "XAI_API_KEY")
    console.print(table)
    console.print()

    # Run selected demos
    demo_model_1_local_llamacpp()
    # demo_model_2_openrouter()
    # demo_model_3_gemini()
    # demo_model_4_grok()

    console.rule("Done", style="bold green")


if __name__ == "__main__":
    main()
