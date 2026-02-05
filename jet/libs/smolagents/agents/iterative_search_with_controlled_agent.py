"""
iterative_search_with_controlled_agent.py

Demonstrates how to use ControlledCodeAgent / ControlledToolCallingAgent
in an iterative deep-research loop until a condition is satisfied.

This script includes support for semantic content search and accumulation into
long-term research memory via the appropriate tools.

Assumes:
- The file containing ControlledCodeAgent, ControlledToolCallingAgent,
  LastNTurnsController, SummaryPlusRecentController
- smolagents with @tool support (tool decorator)
- rich
- jet.libs.smolagents.tools.semantic_content_search_tool and accumulate_relevant_content_tool are available

Run with:
    python iterative_search_with_controlled_agent.py
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.agents.controlled_messages_agent import (
    ControlledCodeAgent,
    ControlledToolCallingAgent,
    LastNTurnsController,
    SummaryPlusRecentController,
)
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from rich.console import Console
from rich.panel import Panel

console = Console()

# ── Long-term content memory & tools ──────────────────────────────
from jet.libs.smolagents.components.long_term_content_memory import (
    LongTermContentMemory,
)
from jet.libs.smolagents.tools.accumulate_relevant_content_tool import (
    add_to_research_knowledge,
    attach_memory_to_accumulate_tool,
    set_current_step,
)
from jet.libs.smolagents.tools.semantic_content_search_tool import (
    attach_memory_to_search_tool,
    search_relevant_content,
)
from smolagents import tool

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
#               Search-oriented Tools
# ────────────────────────────────────────────────


# ────────────────────────────────────────────────
#          Real implementations using available tools
# ────────────────────────────────────────────────


@tool
def x_keyword_search(
    query: str, limit: int = 5, mode: Literal["Latest", "Top"] = "Latest"
) -> str:
    """
    Search real posts on X/Twitter using keyword or advanced search syntax.

    Args:
        query: X advanced search query string.
               Supports operators like:
               - keywords: (puppy OR kitten) cute
               - from:user, since:2025-01-01, filter:images, min_faves:50
               - -filter:replies, lang:en, geocode:...
        limit: Maximum number of posts to return. Clamped to 1–15. Default is 5.
        mode: Sort order — "Latest" (chronological) or "Top" (most relevant/engaged).
              Case-insensitive; normalized to title-case.

    Returns:
        A markdown-formatted string containing:
        - The query and mode used
        - Number of posts found
        - Numbered list of posts showing:
          - @username • date
          - post text (truncated to ~340 characters)

        Returns an error message string if the query is invalid or search fails.
    """
    if not query or not isinstance(query, str):
        return "Query cannot be empty."

    limit = max(1, min(15, limit))  # sensible bounds
    mode_clean = mode.strip().capitalize()  # enforce "Latest" or "Top"
    if mode_clean not in ("Latest", "Top"):
        mode_clean = "Latest"

    try:
        # Call the real tool
        posts = x_keyword_search(query=query.strip(), limit=limit, mode=mode_clean)

        if not posts or not isinstance(posts, (list, tuple)):
            return f"No posts returned for query: **{query}** (mode: {mode_clean})"

        lines = [
            f"**X / Twitter search** — query: **{query}**  •  mode: {mode_clean}  •  {len(posts)} posts\n"
        ]

        for i, post in enumerate(posts, 1):
            user = post.get("username", post.get("user", {}).get("screen_name", "—"))
            text = post.get(
                "text", post.get("content", post.get("full_text", "No text"))
            )[:340]
            date = post.get("created_at", post.get("timestamp", "—"))
            lines.append(f"{i}. @{user}  •  {date}\n   {text}\n")

        return "\n".join(lines).strip()

    except Exception as e:
        return f"Error during X search: {str(e)[:220]}"


# ────────────────────────────────────────────────
#      Simple iterative research orchestrator
# ────────────────────────────────────────────────


@dataclass
class IterativeResearchLoop:
    """
    Runs a controlled agent repeatedly, refining the task until
    the stop_condition returns True or max_rounds is reached.
    Adds long-term semantic content memory and content accumulation.
    """

    agent_type: Literal["code", "toolcalling"]
    model: Any  # e.g. HfApiModel(...)
    controller_type: Literal["last_n", "summary"] = "summary"
    max_rounds: int = 12
    sleep_between_rounds: float = 1.8
    memory: LongTermContentMemory = None  # Will be created if not provided

    def __post_init__(self):
        # Prepare controller
        if self.controller_type == "last_n":
            self.controller = LastNTurnsController(keep_last_turns=6)
        else:
            self.controller = SummaryPlusRecentController(
                summary_every_n_steps=4,  # slightly more frequent summaries
                keep_last_turns=5,
                max_summary_chars=1600,
            )

        # Prepare agent
        tools = [
            SearXNGSearchTool(max_results=10),
            VisitWebpageTool(
                max_output_length=3800,
                top_k=12,
                chunk_target_tokens=450,
            ),
            x_keyword_search,
        ]

        # ── Long-term memory ─────────────────────────────────
        if self.memory is None:
            self.memory = LongTermContentMemory(
                embed_model="nomic-embed-text",
                max_chunks=1200,
            )

        # Add semantic memory tools
        tools.append(search_relevant_content)
        attach_memory_to_search_tool(search_relevant_content, self.memory)

        # Add accumulation/knowledge tool and link memory
        tools.append(add_to_research_knowledge)
        attach_memory_to_accumulate_tool(add_to_research_knowledge, self.memory)

        if self.agent_type == "code":
            self.agent = ControlledCodeAgent(
                model=self.model,
                tools=tools,
                # You can pass system prompt here if you want to guide accumulation behavior
                message_controller=self.controller,
                max_steps=15,  # per-run limit
            )
        else:
            self.agent = ControlledToolCallingAgent(
                model=self.model,
                tools=tools,
                # Consider adding guidance prompt about using memory tools
                message_controller=self.controller,
                max_steps=15,
            )

    def run(
        self,
        initial_task: str,
        stop_condition: Callable[[str], bool],
        refine_prompt_template: str = "Previous attempt:\n{previous}\n\nThe answer is still incomplete or inaccurate. Please continue researching and improve the answer.\nFocus on: {focus}",
    ) -> str:
        console.rule(
            f"[bold cyan]Starting iterative research — {self.agent_type} agent[/]"
        )
        current_step = 0

        current_task = initial_task
        previous_answer = ""
        round_number = 0

        while round_number < self.max_rounds:
            round_number += 1
            console.print(
                f"\n[bold]Round {round_number}/{self.max_rounds}[/] — task:",
                style="bold blue",
            )
            console.print(Panel(current_task, expand=False))

            current_step += 1
            # Update step number for accumulation tool so any content it logs can reference step
            set_current_step(add_to_research_knowledge, current_step)

            # Run one full agent pass
            with console.status("[yellow]Agent thinking & searching...[/]"):
                answer = self.agent.run(current_task)

            # In case answer is wrapped in a custom result type, cast to string if needed
            answer_str = str(answer) if not isinstance(answer, str) else answer

            console.print(
                Panel(answer_str, title="Agent's current answer", border_style="green")
            )

            # Check stopping condition
            if stop_condition(answer_str):
                console.print("[bold green]Goal reached — stopping.[/]")
                return answer_str

            # Optional: Print memory stats every 3 rounds
            if (
                round_number % 3 == 0
                and hasattr(self.memory, "chunks")
                and getattr(self.memory, "chunks", None)
            ):
                console.print(
                    f"[dim]Memory size: {len(self.memory.chunks)} chunks[/dim]"
                )

            # Refine the task for next round
            previous_answer = answer_str
            current_task = refine_prompt_template.format(
                previous=previous_answer[:900],
                focus="missing facts / conflicting information / more recent data",
            )

            console.print("[dim]Refining task for next round...[/dim]")
            time.sleep(self.sleep_between_rounds)

        console.print("[yellow]Max rounds reached — returning last answer.[/]")
        return previous_answer


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


# ────────────────────────────────────────────────
#               Example usage
# ────────────────────────────────────────────────


def answer_looks_complete(text: str) -> bool:
    """Very naive stopping condition — improve with LLM judge if needed"""
    text_lower = text.lower()
    return (
        len(text) > 400
        and "population" in text_lower
        and any(str(y) in text_lower for y in [2024, 2025, 2026])
        and any(k in text_lower for k in ["million", "paris"])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative Research Loop: Paris Population Example",
        allow_abbrev=False,
    )

    # Positional argument (optional)
    parser.add_argument(
        "task_positional",
        type=str,
        nargs="?",
        default=None,
        help="The research question/task (positional alternative to --task)",
    )

    # Classic named flag (with default)
    parser.add_argument(
        "--task",
        type=str,
        default="What is the current population of Paris, France in 2026? Include latest estimates, sources, and whether it includes the metropolitan area.",
        help="Initial research task/question. Can also be provided positionally.",
    )

    parser.add_argument(
        "--max_rounds",
        type=int,
        default=8,
        help="Maximum number of rounds for the research loop.",
    )

    parser.add_argument(
        "--agent_type",
        type=str,
        choices=["toolcalling", "code"],
        default="toolcalling",
        help="Type of agent to use (toolcalling or code).",
    )

    parser.add_argument(
        "--controller_type",
        type=str,
        choices=["summary", "last_n"],
        default="summary",
        help="Type of research controller.",
    )

    args = parser.parse_args()

    # Decide which task to use - positional wins if provided
    task = args.task_positional if args.task_positional is not None else args.task

    model = create_local_model()

    researcher = IterativeResearchLoop(
        agent_type=args.agent_type,
        model=model,
        controller_type=args.controller_type,
        max_rounds=args.max_rounds,
    )

    final_answer = researcher.run(
        initial_task=task,
        stop_condition=answer_looks_complete,
    )

    console.rule("Final Result")
    console.print(
        Panel(final_answer, title="Most complete answer found", border_style="green")
    )
