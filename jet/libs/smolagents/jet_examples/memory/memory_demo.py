# File: smolagents_memory_demo.py
from typing import Any

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from jet.libs.smolagents.utils.model_utils import create_local_model
from rich.console import Console
from rich.panel import Panel
from smolagents import (
    CodeAgent,
    tool,
)

console = Console()


@tool
def update_consolidated_answer(key: str, value: str) -> str:
    """
    Store or update a single consolidated key-value fact/answer in agent memory.

    This is a stateful tool — values persist across steps and across different .run() calls
    (as long as the same agent instance is used with reset=False).

    Args:
        key: Unique identifier for this piece of knowledge (e.g. "best_framework_2026")
        value: The consolidated answer / fact / decision to store

    Returns:
        str: Human-readable confirmation message showing old → new value
             (helps agent see what changed)

    Side effects:
        Updates internal persistent dictionary (simulated via function attribute)
    """
    if not hasattr(update_consolidated_answer, "store"):
        update_consolidated_answer.store: dict[str, str] = {}

    old_value = update_consolidated_answer.store.get(key, "<not set>")
    update_consolidated_answer.store[key] = value

    return f"Consolidated answer updated: '{key}'\n  Old: {old_value}\n  New: {value}"


@tool
def get_all_consolidated_answers() -> str:
    """
    Return a readable summary of all currently stored consolidated answers.

    Returns:
        str: Formatted string with all key-value pairs, sorted by key.
             Returns friendly message if store is empty.
    """
    if (
        not hasattr(update_consolidated_answer, "store")
        or not update_consolidated_answer.store
    ):
        return "No consolidated answers have been stored yet."

    lines = [
        f"• {key}: {value}"
        for key, value in sorted(update_consolidated_answer.store.items())
    ]
    return "Current consolidated knowledge:\n" + "\n".join(lines)


@tool
def add_debug_note(note: str) -> str:
    """
    Add a short debug / thinking / reminder note that will appear in memory.
    Useful when the agent wants to leave breadcrumbs without updating the main knowledge base.

    Args:
        note: The short message / observation / reminder to record

    Returns:
        str: Confirmation that the note was added (visible in next observations)
    """
    return f"[DEBUG / REMINDER ADDED] {note}"


def memory_size_watcher(memory_step: Any, agent: CodeAgent) -> None:
    """Simple step callback: warn when memory grows large.

    Compatible with smolagents callback calling pattern: cb(memory_step, agent=agent)

    """
    # We don't rely on memory_step being a specific type here
    try:
        steps = agent.memory.get_full_steps()
        size = len(steps)
        if size > 12:
            console.print(
                f"[yellow bold]Memory size warning[/yellow bold]: now at {size} steps — "
                f"consider summarizing or using reset soon"
            )
        # Optional: show progress every 5 steps
        elif size > 0 and size % 5 == 0:
            console.print(f"[dim]Memory step count: {size}[/dim]")
    except Exception as exc:
        console.print(f"[red]Memory watcher failed: {exc}[/red]")


def create_research_agent(
    model_id: LLAMACPP_LLM_KEYS = "qwen3-instruct-2507:4b",
) -> CodeAgent:
    """Factory for consistent agent setup."""
    model = create_local_model(
        model_id=model_id,
        temperature=0.4,
        max_tokens=1400,
    )

    tools = [
        update_consolidated_answer,
        get_all_consolidated_answers,
        add_debug_note,
        # Add real tools here in production, e.g. DuckDuckGoSearchTool(), etc.
        SearXNGSearchTool(max_results=10),
        VisitWebpageTool(
            max_output_length=3800,
            top_k=12,
            chunk_target_tokens=450,
        ),
    ]

    agent = CodeAgent(
        model=model,
        instructions=(
            "\nYou are a careful researcher. Use update_consolidated_answer(key, value) "
            "whenever you reach a confident, consolidated finding. "
            "Always check get_all_consolidated_answers() first to avoid duplication."
        ),
        tools=tools,
        add_base_tools=False,  # we provide our own
        step_callbacks=[memory_size_watcher],
    )

    return agent


def example_1_single_long_run() -> None:
    """Pattern A: one long .run() — agent consolidates internally over many steps."""
    console.rule("Example 1 — Single long run")
    agent = create_research_agent()

    task = (
        "Research the current (2026) most popular lightweight Python web framework "
        "for building APIs (not full-stack websites). "
        "Compare at least 3 candidates. "
        "Update consolidated_answer 'best_lightweight_api_framework_2026' "
        "with your final reasoned choice + short justification."
    )

    console.print(Panel(task, title="Task", border_style="cyan"))

    # Single call — memory grows inside the loop
    result = agent.run(task, max_steps=18, reset=True)

    console.print("\n[bold green]Final answer:[/bold green]")
    console.print(result)

    console.print("\n[bold]Consolidated store (from tool):[/bold]")
    print(get_all_consolidated_answers())


def example_2_multi_call_iterative() -> None:
    """Pattern B: multiple .run(..., reset=False) calls — like a chat/refinement session."""
    console.rule("Example 2 — Multi-turn / iterative refinement")

    agent = create_research_agent()

    questions = [
        "What are the top Python web frameworks in early 2026 for high-performance APIs?",
        "Now focus only on the ones with < 5k lines of code (very lightweight). Rank them.",
        "Which one has the best type hints / modern Python support in 2026? Update the consolidated answer.",
        "Final summary: give me the consolidated answer + why it won.",
    ]

    consolidated_key = "best_ultra_light_api_framework_2026"

    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold cyan]Turn {i}/{len(questions)}[/bold cyan]")
        console.print(Panel(q, title="User", expand=False))

        # Only first call resets memory
        reset = i == 1

        answer = agent.run(
            q,
            reset=reset,
            max_steps=12,
        )

        console.print("[dim]→ Agent:[/dim]")
        console.print(answer.strip())

    # After all turns → show final consolidated state
    console.print(
        "\n[bold magenta]Final consolidated knowledge after multi-turn:[/bold magenta]"
    )
    console.print(get_all_consolidated_answers())


def jet_example_1() -> None:
    """Pattern A: one long .run() — agent consolidates internally over many steps."""
    console.rule("Example 1 — Single long run")
    agent = create_research_agent()

    task = (
        "As of February 07, 2026, research the **top 10 most popular / trending ongoing anime** that are currently airing "
        "(weekly releases, active episodes this season – Winter 2026). "
        "Focus on shows that have new episodes releasing right now or very recently. "
        "\n\n"
        "For each of at least 10 anime, try to collect: "
        "- English title (and Japanese / Romaji if relevant) "
        "- Studio / Production company "
        "- Current episode count (how many episodes have aired so far in this season) "
        "- Scheduled date/time of the next episode (if known and upcoming) "
        "- Main streaming platform(s) (Crunchyroll, Netflix, HIDIVE, Amazon, etc.) "
        "- Indicators of popularity (trending rank on X/Twitter, MyAnimeList score & members, AniList ranking, "
        "  discussion volume, viewership estimates, social media buzz) "
        "\n\n"
        "Use search tools, seasonal charts, MyAnimeList, AniList, livechart.me, or similar sources to get accurate, up-to-date information. "
        "Store findings using update_consolidated_answer with clear, consistent keys, for example: "
        "  'anime_rank_1_title', 'anime_rank_1_current_episodes', 'anime_rank_1_next_episode', etc. "
        "or structured keys like 'top_ongoing_anime_winter_2026' if storing multiple in one value. "
        "\n\n"
        "At the end, update the key 'top_10_ongoing_anime_feb_2026' with a concise ranked summary "
        "(at least top 10 titles + brief reason for their position / popularity)."
    )

    console.print(Panel(task, title="Task", border_style="cyan"))

    # Single call — memory grows inside the loop
    result = agent.run(task, max_steps=18, reset=True)

    console.print("\n[bold green]Final answer:[/bold green]")
    console.print(result)

    console.print("\n[bold]Consolidated store (from tool):[/bold]")
    print(get_all_consolidated_answers())


if __name__ == "__main__":
    console.print(
        "[bold green]smolagents memory + consolidation demo[/bold green] (2026 style)\n"
    )

    # Run both patterns
    # example_1_single_long_run()
    # example_2_multi_call_iterative()
    jet_example_1()
