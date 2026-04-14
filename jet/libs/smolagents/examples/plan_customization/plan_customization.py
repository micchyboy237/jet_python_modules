"""
Plan Customization Example – Updated 2026

Current reference date for example: 2026-01-29

This example demonstrates:
- Step callbacks to interrupt after plan creation for user review/modification
- Resuming execution with preserved memory (reset=False)
- Dynamic current date/time via tool (avoids stale data)
- Correct usage of custom AgentMemory with automatic context compression
- Optional Mem0 integration for long-term persistent memory
"""

import shutil
from datetime import datetime
from pathlib import Path

import pytz
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_memory import (
    AgentMemory,  # Your enhanced memory with compression + Mem0
)
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from smolagents import (
    CodeAgent,
    LocalPythonExecutor,
    PlanningStep,
    tool,
)

OUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


@tool
def get_current_datetime(timezone: str | None = "Asia/Manila") -> str:
    """Returns the current date and time in the specified timezone.

    Args:
        timezone: IANA timezone name (default: Asia/Manila for Quezon City, PH).
                  Examples: 'UTC', 'America/New_York', 'Asia/Tokyo'.

    Returns:
        Formatted string like '2026-01-29 21:40:00 PST' or equivalent.
    """
    try:
        tz = pytz.timezone(timezone or "UTC")
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error fetching time for '{timezone}': {str(e)}. Falling back to UTC."


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int = 12000,
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


def create_custom_memory(
    system_prompt: str = "",
    max_tokens_before_compress: int = 12000,
) -> AgentMemory:
    """Create and configure the enhanced AgentMemory (compression + optional Mem0)."""
    if not system_prompt:
        system_prompt = (
            "You are a helpful assistant that remembers important facts, "
            "decisions, and progress across steps. Use tools when needed."
        )

    memory = AgentMemory(system_prompt)

    # Tune compression settings (adjust based on your model's context window)
    memory.max_tokens_before_compress = max_tokens_before_compress
    memory.keep_recent_steps = 10

    # Optional: Enable Mem0 for long-term, searchable persistent memory across sessions
    # memory.set_scoping(user_id="plan_example_user", agent_id="planning_agent")

    return memory


def display_plan(plan_content: str):
    """Display the plan in a formatted way"""
    print("\n" + "=" * 60)
    print("🤖 AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice() -> int:
    """Get user's choice for plan approval (default to 1 if blank)"""
    while True:
        choice = input(
            "\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3) [default 1]: "
        ).strip()
        if choice == "":
            return 1
        if choice in {"1", "2", "3"}:
            return int(choice)
        print("Invalid choice. Please enter 1, 2, or 3.")


def get_modified_plan(original_plan: str) -> str:
    """Allow user to modify the plan (multi-line input, end with two empty lines)"""
    print("\n" + "-" * 40)
    print("MODIFY PLAN")
    print("-" * 40)
    print("Current plan:")
    print(original_plan)
    print("-" * 40)
    print("Enter your modified plan (press Enter twice to finish):")

    lines = []
    empty_count = 0
    while empty_count < 2:
        line = input()
        if not line.strip():
            empty_count += 1
        else:
            empty_count = 0
        lines.append(line)

    modified = "\n".join(lines[:-2]).strip()
    return modified if modified else original_plan


def interrupt_after_plan(memory_step, agent):
    """Step callback: interrupt after PlanningStep for user review/modification"""
    if isinstance(memory_step, PlanningStep):
        print("\n🛑 Agent interrupted after plan creation...")

        display_plan(memory_step.plan)

        choice = get_user_choice()

        if choice == 1:
            print("✅ Plan approved! Continuing...")
            return

        elif choice == 2:
            modified = get_modified_plan(memory_step.plan)
            memory_step.plan = modified
            print("\nPlan updated!")
            display_plan(modified)
            print("✅ Continuing with modified plan...")
            return

        elif choice == 3:
            print("❌ Execution cancelled by user.")
            agent.interrupt()
            return


def compress_memory_callback(memory_step, agent):
    """Callback that triggers compression when needed."""
    if not hasattr(agent, "memory") or not isinstance(agent.memory, AgentMemory):
        return

    # Compress after planning steps or periodically (every 5 steps)
    if isinstance(memory_step, PlanningStep) or len(agent.memory.steps) % 5 == 0:
        try:
            agent.memory.compress_old_steps(agent)
            # Optional: Push step to Mem0 for long-term semantic memory
            # agent.memory.add_step_to_mem0(memory_step)
        except Exception as e:
            print(f"Warning: Memory compression failed: {e}")


def parseargs():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plan Customization Example (with dynamic date tool and custom memory)"
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="?",
        default="Search for top romance and isekai anime this year. Include the plots for each. Include source for each.",
        help="Task prompt for the agent to solve.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for the local model (default: 0.7)",
    )
    # Note: max_tokens is intentionally NOT added to CLI yet (see below)
    parser.add_argument(
        "-n",
        "--agent_name",
        type=str,
        default="planning_agent",
        help="Agent name (default: planning_agent)",
    )
    parser.add_argument(
        "-m",
        "--max_steps",
        type=int,
        default=15,
        help="Maximum agent steps (default: 15)",
    )
    parser.add_argument(
        "-p",
        "--planning_interval",
        type=int,
        default=5,
        help="Planning interval (default: 5)",
    )
    parser.add_argument(
        "-v",
        "--verbosity_level",
        type=int,
        default=2,
        help="Verbosity level (default: 2)",
    )
    parser.add_argument(
        "-mt",
        "--max-tokens",
        type=int,
        default=12000,
        help="Maximum tokens for model generation (default: 12000)",
    )
    parser.add_argument(
        "-ct",
        "--compress-tokens",
        type=int,
        default=8000,
        help="Max tokens before memory compression (default: 8000)",
    )
    return parser.parse_args()


def get_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def main():
    print(
        "🚀 Starting Plan Customization Example (with dynamic date tool + custom memory)"
    )
    print("=" * 60)

    CODE_EXECUTION_TIME_SECONDS = 60

    args = parseargs()

    model = create_local_model(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        agent_name=args.agent_name,
    )

    # 1. Create your enhanced custom memory (compression + optional Mem0)
    additional_authorized_imports = []
    custom_memory = create_custom_memory(
        max_tokens_before_compress=args.compress_tokens,
    )

    # 2. Create the agent (do NOT pass memory= here)
    executor = LocalPythonExecutor(
        additional_authorized_imports, timeout_seconds=CODE_EXECUTION_TIME_SECONDS
    )
    agent = CodeAgent(
        model=model,
        tools=[
            # get_current_datetime,  # dynamic date/time tool
            SearXNGSearchTool(max_results=10),
            VisitWebpageTool(
                max_output_length=3500, chunk_target_tokens=500, top_k=None
            ),
        ],
        planning_interval=args.planning_interval,
        step_callbacks={
            PlanningStep: interrupt_after_plan,  # specific callback for plan review
            # You can add more specific callbacks here if needed
        },
        max_steps=args.max_steps,
        verbosity_level=args.verbosity_level,
        code_block_tags=("```python", "```"),
        executor=executor,
        instructions=f"Today's date is {get_today()}. Always use this as the reference for 'current date' or 'today' unless the user specifies otherwise.",
    )

    # 3. REPLACE the default memory with your custom one (this is the correct way)
    agent.memory = custom_memory

    # Optional: Register compression as a general callback if your smolagents version supports broad registration
    # For safety, we can add it via step_callbacks if needed, but the current callback already handles it via PlanningStep + periodic check

    task = args.task

    try:
        print(f"\n📋 Task: {task}")
        print(
            "\n🤖 Agent starting execution... (can call get_current_datetime when needed)"
        )

        result = agent.run(task)

        print("\n✅ Task completed successfully!")
        print("\n📄 Final Result:")
        print("-" * 40)
        print(result)

    except Exception as e:
        if "interrupted" in str(e).lower():
            print("\n🛑 Agent execution was cancelled by user.")
            print("\nTo resume later (preserving memory + tool access):")
            print("  agent.run(task, reset=False)\n")

            resume_choice = input("Show resume demonstration? (y/n): ").strip().lower()
            if resume_choice == "y":
                print("\n🔄 Resuming execution with preserved memory...")
                try:
                    agent.run(task, reset=False)
                    print("\n✅ Resume completed!")
                except Exception as resume_err:
                    print(f"Resume failed: {resume_err}")
        else:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
