# demo_multi_agent_web_local.py
"""
Multi-agent web browser demos using smolagents with LOCAL llama.cpp server.
Reuses create_local_model() from previous examples.
Shows manager (CodeAgent) orchestrating a web sub-agent (ToolCallingAgent).
"""

import re
import shutil
import time
from pathlib import Path

import requests
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.step_callbacks import save_step_state
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from markdownify import markdownify
from requests.exceptions import RequestException
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    tool,
)

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Reuse from previous file
# ──────────────────────────────────────────────────────────────────────────────


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
    agent_name: str | None = None,
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        agent_name=agent_name,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────


@tool
def visit_webpage(url: str) -> str:
    """Fetches a webpage and converts it to clean markdown.

    Args:
        url: The URL to visit.

    Returns:
        Markdown content or error message.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        md = markdownify(response.text).strip()
        md = re.sub(r"\n{3,}", "\n\n", md)  # collapse excessive newlines
        md = re.sub(r"\s{2,}", " ", md)  # normalize spaces

        preview = md[:400] + "..." if len(md) > 400 else md
        console.print(f"[dim]Visited {url} → preview: {preview}[/dim]")

        return md

    except RequestException as e:
        return f"Error fetching page: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ──────────────────────────────────────────────────────────────────────────────
# Agent factories
# ──────────────────────────────────────────────────────────────────────────────


def create_web_sub_agent(
    max_steps: int = 10, verbosity_level: int = 2
) -> ToolCallingAgent:
    """Creates the web-specialized ToolCallingAgent."""
    model = create_local_model(temperature=0.65, agent_name="web_sub_agent")

    return ToolCallingAgent(
        tools=[
            SearXNGSearchTool(max_results=10),
            VisitWebpageTool(
                max_output_length=4096,
                top_k=None,
            ),
        ],
        model=model,
        max_steps=max_steps,
        name="web_agent",
        description="Performs web searches and visits result urls to get more details when needed.",
        verbosity_level=verbosity_level,
        step_callbacks=[
            save_step_state("web_agent"),
        ],
    )


def create_manager_agent(
    managed_agents: list,
    max_steps: int = 15,
    verbosity_level: int = 2,
    additional_imports: list[str] | None = None,
) -> CodeAgent:
    """Creates the top-level CodeAgent that orchestrates sub-agents."""
    model = create_local_model(temperature=0.7, agent_name="manager_agent")

    return CodeAgent(
        tools=[],
        model=model,
        managed_agents=managed_agents,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        additional_authorized_imports=additional_imports or ["time", "numpy", "pandas"],
        add_base_tools=True,
        step_callbacks=[
            save_step_state("manager_agent"),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────


def demo_multi_1_simple_delegation():
    """Demo 1: Simple question that requires web lookup"""
    console.rule("Demo 1: Basic delegation to web agent")

    web_agent = create_web_sub_agent(max_steps=8)
    manager = create_manager_agent([web_agent], max_steps=10)

    question = "What is the latest stable version of the Hugging Face Transformers library as of today?"
    # question = "Search for top 10 anime in 2026. Check some relevant urls until you can provide with 10 results."

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    start = time.time()

    answer = manager.run(question, reset=True)
    # answer = web_agent.run(question, reset=True)
    duration = time.time() - start
    console.print(
        Panel(answer, title="Final Answer", border_style="green", expand=False)
    )
    console.print(f"[dim]Completed in {duration:.1f} seconds[/dim]")


def demo_multi_2_calculation_plus_research():
    """Demo 2: Question needing both research + numeric calculation"""
    console.rule("Demo 2: Research + computation")

    web_agent = create_web_sub_agent(max_steps=12, verbosity_level=2)
    manager = create_manager_agent(
        [web_agent],
        max_steps=18,
        additional_imports=["time", "numpy", "pandas", "datetime"],
    )

    question = (
        "If AI training compute doubles every 6 months, and current largest runs use about 10,000 H100 GPUs "
        "(~5 MW total power), estimate the power in GW required for the biggest run in 2030. "
        "Compare to the average power consumption of a small European country."
    )

    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
    start = time.time()

    answer = manager.run(question, reset=True)
    duration = time.time() - start
    console.print(
        Panel(answer, title="Final Answer", border_style="green", expand=False)
    )
    console.print(f"[dim]Completed in {duration:.1f} seconds[/dim]")


def demo_multi_3_inspect_agents_and_memory():
    """Demo 3: Run a task then inspect managed agents & memory"""
    console.rule("Demo 3: Inspect hierarchy & memory after run")

    web_agent = create_web_sub_agent(max_steps=6)
    manager = create_manager_agent([web_agent], max_steps=12)

    question = "Who won the Nobel Prize in Physics in 2025 and what was it awarded for?"

    console.print(f"\n[bold cyan]Running:[/bold cyan] {question}\n")
    _ = manager.run(question, reset=True)

    # Inspect hierarchy
    table = Table(title="Agent Hierarchy")
    table.add_column("Agent", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description")
    table.add_row("Manager", "CodeAgent", "Orchestrates & computes")
    table.add_row("web_agent", "ToolCallingAgent", "Web search & page reading")
    console.print(table)

    # Memory summary
    if manager.memory.steps:
        console.print("\n[bold]Memory steps summary:[/bold]")
        from collections import Counter

        types = Counter(type(s).__name__ for s in manager.memory.steps)
        for t, c in types.most_common():
            console.print(f"  • {t:18} : {c:2d}x")

        console.print(f"\nTotal steps: {len(manager.memory.steps)}")
    else:
        console.print("[dim]No steps recorded.[/dim]")


def main():
    console.rule("Multi-Agent Web Browser Demos — LOCAL llama.cpp", style="bold blue")

    console.print(
        "[dim]Hierarchy: CodeAgent (manager) → ToolCallingAgent (web search & visit)[/dim]\n"
    )

    # Uncomment demos you want to run
    demo_multi_1_simple_delegation()
    # demo_multi_2_calculation_plus_research()
    # demo_multi_3_inspect_agents_and_memory()

    console.rule("Done", style="bold green")


if __name__ == "__main__":
    main()
