"""
Plan Customization Example – Updated 2026

Current reference date for example: 2026-01-29

This example demonstrates:
- Step callbacks to interrupt after plan creation
- User approval/modification of plan
- Resuming with preserved memory
- Best practice: dynamic current date/time via tool (avoids stale data)
"""

import shutil
from datetime import datetime
from pathlib import Path

import pytz
from jet.adapters.llama_cpp.types import LLAMACPP_LLM_KEYS
from jet.libs.smolagents.custom_models import OpenAIModel
from jet.libs.smolagents.tools.searxng_search_tool import SearXNGSearchTool
from jet.libs.smolagents.tools.visit_webpage_tool import VisitWebpageTool
from smolagents import (
    CodeAgent,
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
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Error fetching time for '{timezone}': {str(e)}. Falling back to UTC."


def create_local_model(
    temperature: float = 0.4,
    max_tokens: int | None = 8000,
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


def display_plan(plan_content: str):
    """Display the plan in a formatted way"""
    print("\n" + "=" * 60)
    print("🤖 AGENT PLAN CREATED")
    print("=" * 60)
    print(plan_content)
    print("=" * 60)


def get_user_choice() -> int:
    """Get user's choice for plan approval"""
    while True:
        choice = input(
            "\nChoose an option:\n1. Approve plan\n2. Modify plan\n3. Cancel\nYour choice (1-3): "
        ).strip()
        if choice in ["1", "2", "3"]:
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


def main():
    print("🚀 Starting Plan Customization Example (with dynamic date tool)")
    print("=" * 60)

    model = create_local_model(temperature=0.7, agent_name="planning_agent")

    # Optional: static date injection (only if you really want it baked in)
    # today_str = datetime.now(pytz.timezone("Asia/Manila")).strftime("%Y-%m-%d")
    # custom_system = f"""You are a helpful assistant. The current date is {today_str}.
    # Always use get_current_datetime tool for precise time-sensitive questions."""

    agent = CodeAgent(
        model=model,
        tools=[
            # DuckDuckGoSearchTool(),
            get_current_datetime,  # ← dynamic date/time tool
            SearXNGSearchTool(max_results=20),
            VisitWebpageTool(max_output_length=3500, chunk_target_tokens=500, top_k=7),
        ],
        planning_interval=5,
        step_callbacks={PlanningStep: interrupt_after_plan},
        max_steps=15,
        verbosity_level=2,
        # system_prompt=custom_system,      # ← uncomment only if you want static injection
    )

    # task = """Search for recent developments in artificial intelligence and provide a summary of the top 3 most significant breakthroughs in 2025. Include the source of each breakthrough."""
    task = """Search for top anime this year. Include source for each."""

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
