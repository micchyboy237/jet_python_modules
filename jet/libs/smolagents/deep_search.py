"""
smolagents v1.24.0+ – Managed agents (hierarchical/multi-agent) usage examples
Updated January 2026 – no separate ManagedAgent class anymore

Just give sub-agents .name and .description → pass them directly in managed_agents=[...]
"""

import os
import json
import datetime
from pathlib import Path
from typing import List, Optional

from smolagents import (
    CodeAgent,
    tool,
    LogLevel,
)

from jet.adapters.llama_cpp.models import LLAMACPP_MODEL_CONTEXTS
from jet.adapters.llama_cpp.tokens import count_tokens
from jet.adapters.llama_cpp.types import LLAMACPP_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.custom_tools import VisitWebpageTool, WebSearchTool

MODEL: LLAMACPP_KEYS = "qwen3-instruct-2507:4b"
MODEL_MAX_CONTEXT: int = LLAMACPP_MODEL_CONTEXTS[MODEL]


# ────────────────────────────────────────────────────────────────
# Your local model factory (as provided)
# ────────────────────────────────────────────────────────────────
def create_local_model(
    temperature: float = 0.3,
    max_tokens: Optional[int] = 4096,
    model_id: str = "local-model",
    logs_dir: str | Path | None = None,
) -> OpenAIModel:
    return OpenAIModel(
        model_id=model_id,
        api_base="http://shawn-pc.local:8080/v1",
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=True,
        logs_dir=str(logs_dir) if logs_dir else None,
    )


# ────────────────────────────────────────────────────────────────
# Progress logger (timestamp + step preview)
# ────────────────────────────────────────────────────────────────
def rich_progress_logger(step):
    # Compatible with ActionStep, PlanningStep, etc.
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thought_field = (
        getattr(step, "thought", None)
        or getattr(step, "plan", None)
        or getattr(step, "model_output", None)
        or getattr(step, "task", None)
        or getattr(step, "code_action", None)
        or ""
    )
    thought = str(thought_field).strip().replace("\n", " ")[:140] or "—"
    print(f"[{now}] [Step {getattr(step, 'step_number', '?'):2}] {thought} …")
    if hasattr(step, "code_action") and step.code_action:
        print(f"  └─ Code: {step.code_action[:88]}…")


# ────────────────────────────────────────────────────────────────
# Helper: create focused specialist agents
# ────────────────────────────────────────────────────────────────
def create_specialist(
    model, name: str, description: str, tools: List = None, extra_instructions: str = ""
) -> CodeAgent:
    system_add = (
        f"You are {name} – specialized in {description.lower()}. "
        "Stay in your domain. Be precise, concise, evidence-based. "
        "Never guess facts. "
        f"{extra_instructions}"
    )

    agent = CodeAgent(
        tools=tools or [],
        model=model,
        name=name,  # ← required for managed_agents
        description=description,  # ← required – becomes part of manager prompt
        max_steps=15,
        verbosity_level=LogLevel.INFO,
        add_base_tools=False,
        # If your version supports it:
        # additional_system_prompt=system_add,
    )
    return agent


# ────────────────────────────────────────────────────────────────
# Example 1: Simple manager + one web-research sub-agent
# ────────────────────────────────────────────────────────────────
def example_simple_hierarchy(out_dir: str | Path | None = None):
    out_dir = (
        Path(out_dir)
        if out_dir
        else Path(__file__).parent
        / "generated"
        / Path(__file__).stem
        / "simple_hierarchy"
    )
    model = create_local_model(temperature=0.4, logs_dir=out_dir / "llm_logs")

    # Sub-agent: focused web researcher
    web_agent = create_specialist(
        model=model,
        name="WebResearcher",
        description="Performs internet searches and reads webpage content when up-to-date information or external facts are needed.",
        tools=[
            WebSearchTool(max_results=10),
            VisitWebpageTool(max_output_length=7000),
        ],
    )

    # Manager that can delegate to the sub-agent
    manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[web_agent],  # ← just pass the agent directly
        name="Coordinator",
        description="Top-level coordinator that delegates research when needed.",
        max_steps=25,
        planning_interval=5,
        step_callbacks=[rich_progress_logger],
        verbosity_level=LogLevel.INFO,
    )

    task = (
        "What was the latest winning Lotto combination in the Philippines? "
        "Give the most recent one and the next draw date."
    )

    print("\n" + "═" * 90)
    print("Example 1 : Simple manager → web sub-agent")
    print("═" * 90 + "\n")

    result = manager.run(task)
    print("\nFinal result:\n", result)


# ────────────────────────────────────────────────────────────────
# Example 2: Advanced coordinator with multiple specialists
# (outline → delegate → verify pattern)
# ────────────────────────────────────────────────────────────────
def example_advanced_coordinator(out_dir: str | Path | None = None):
    out_dir = (
        Path(out_dir)
        if out_dir
        else Path(__file__).parent
        / "generated"
        / Path(__file__).stem
        / "advanced_coordinator"
    )
    model = create_local_model(
        temperature=0.35, max_tokens=2048, logs_dir=out_dir / "llm_logs"
    )

    specialists = [
        create_specialist(
            model,
            "Researcher",
            "gathers current facts, prices, news via web tools",
            tools=[
                WebSearchTool(max_results=12),
                VisitWebpageTool(max_output_length=8000),
            ],
        ),
        create_specialist(
            model,
            "Analyzer",
            "compares options, evaluates pros/cons, performs calculations",
        ),
        create_specialist(
            model, "Writer", "writes clear, structured final answers and summaries"
        ),
        create_specialist(
            model,
            "Verifier",
            "critically reviews outputs for accuracy, completeness, contradictions",
        ),
    ]

    coordinator = CodeAgent(
        tools=[],
        model=model,
        managed_agents=specialists,  # ← all specialists passed here
        name="PlannerCoordinator",
        description="""
You are a task orchestrator. Follow this protocol:

1. First: create a numbered high-level PLAN (4–10 tasks) with estimated owner.
   Example:
   1. Research latest data ................................ [Pending] [Researcher]
   2. ...

2. Then iteratively:
   - Pick next unfinished task
   - Delegate to the best matching agent (by name)
   - After result: evaluate → mark Done/Failed
   - Use Verifier for important outputs

3. Only when all tasks Done → synthesize FINAL ANSWER
""",
        max_steps=60,
        planning_interval=4,
        step_callbacks=[rich_progress_logger],
        verbosity_level=LogLevel.INFO,
    )

    query = (
        "Compare rent and quality of life in Quezon City vs BGC for a remote software engineer in 2026. "
        "Include 1-bedroom condo prices, internet speed, flood risk, coworking options. "
        "Give recommendation + pros/cons table."
    )

    print("\n" + "═" * 90)
    print("Example 2 : Advanced coordinator with multiple managed specialists")
    print("Query:", query)
    print("═" * 90 + "\n")

    result = coordinator.run(query)
    print("\nFinal comprehensive answer:\n", result)


# ────────────────────────────────────────────────────────────────
# Run examples (comment/uncomment as needed)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("smolagents managed_agents usage – updated 2026 (no ManagedAgent wrapper)\n")

    # Choose which example to run
    example_simple_hierarchy()
    # example_advanced_coordinator()
