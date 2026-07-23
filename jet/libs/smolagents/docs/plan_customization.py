# demo_human_in_the_loop_local.py
"""
Human-in-the-Loop (HITL) demos with smolagents using LOCAL llama.cpp server.
Shows plan interruption, modification, resume, memory inspection.
"""

from collections.abc import Callable

from jet.adapters.smolagents.factory import create_code_agent, create_llm_model
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reusable HITL callback helpers
# ──────────────────────────────────────────────────────────────────────────────


def create_plan_interrupt_callback(
    allow_edit: bool = True,
    show_steps_before: bool = False,
) -> Callable:
    """
    Factory that returns a step callback for PlanningStep.
    Pauses execution, shows the plan, lets user approve/edit/cancel.
    """

    def interrupt_after_plan(step: PlanningStep, agent: CodeAgent) -> None:
        if not isinstance(step, PlanningStep):
            return

        print("\n" + "=" * 70)
        print("🤖 AGENT PLAN CREATED".center(70))
        print("=" * 70)

        if show_steps_before and agent.memory.steps:
            print("\nPrevious steps in memory:")
            for i, s in enumerate(agent.memory.steps, 1):
                print(f"  {i:2d}. {type(s).__name__}")

        print("\nProposed plan:")
        print("-" * 60)

        plan_lines = step.plan.splitlines()
        if not plan_lines:
            print("  (empty plan)")
        else:
            for line in plan_lines:
                print(line)

        print("-" * 60)

        while True:
            print("\nOptions:")
            print("  1 = Approve & continue")
            if allow_edit:
                print("  2 = Modify plan")
            print("  3 = Cancel execution")
            choice = input("Your choice (1-3): ").strip()

            if choice == "1":
                print("→ Plan approved. Continuing...\n")
                break

            elif choice == "2" and allow_edit:
                print("\nEnter the new plan (one action per line).")
                print("End input with empty line + Enter.\n")
                new_plan_lines = []
                while True:
                    line = input("> ").rstrip()
                    if not line.strip():
                        break
                    new_plan_lines.append(line)

                if new_plan_lines:
                    step.plan = "\n".join(new_plan_lines).strip()
                    print("\n→ Plan updated.")
                else:
                    print("→ No changes made.")
                break

            elif choice == "3":
                print("→ Execution cancelled by user.")
                raise KeyboardInterrupt("User cancelled execution")

            else:
                print("Invalid choice. Please enter 1, 2 or 3.")

    return interrupt_after_plan


def create_rag_like_agent(
    planning_interval: int = 4,
    max_steps: int = 12,
    verbosity_level: int = 1,
    allow_plan_edit: bool = True,
) -> CodeAgent:
    """Create agent with local model + search tool + HITL callback."""
    model = create_llm_model(temperature=0.65)

    callback = create_plan_interrupt_callback(allow_edit=allow_plan_edit)

    agent = create_code_agent(
        tools=[DuckDuckGoSearchTool()],
        model=model,
        planning_interval=planning_interval,
        step_callbacks={PlanningStep: callback},
        max_steps=max_steps,
        verbosity_level=verbosity_level,
    )
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# Demos
# ──────────────────────────────────────────────────────────────────────────────


def demo_hitl_1_basic_interrupt():
    """Demo 1: Basic plan interrupt + approve/cancel only"""
    print("\n" + "=" * 78)
    print("Demo 1: Basic HITL – interrupt after plan, approve or cancel".center(78))
    print("=" * 78)

    agent = create_rag_like_agent(
        planning_interval=4,
        max_steps=10,
        verbosity_level=1,
        allow_plan_edit=False,  # only approve/cancel
    )

    task = (
        "What are the three most important AI research papers published in 2025 so far?"
    )

    print(f"\nTask: {task}\n")
    try:
        result = agent.run(task, reset=True)
        print("\nFinal result:", result)
    except KeyboardInterrupt:
        print("\nExecution stopped by user.")


def demo_hitl_2_modify_plan():
    """Demo 2: Allow user to edit the plan interactively"""
    print("\n" + "=" * 78)
    print("Demo 2: HITL with plan modification".center(78))
    print("=" * 78)

    agent = create_rag_like_agent(
        planning_interval=5,
        max_steps=12,
        verbosity_level=1,
        allow_plan_edit=True,
    )

    task = "Find recent news about open-source LLM releases in Asia this month."

    print(f"\nTask: {task}\n")
    try:
        result = agent.run(task, reset=True)
        print("\nFinal result:", result)
    except KeyboardInterrupt:
        print("\nExecution stopped by user.")


def demo_hitl_3_resume_after_interrupt():
    """Demo 3: Interrupt → modify/approve → resume with same agent (memory preserved)"""
    print("\n" + "=" * 78)
    print("Demo 3: Interrupt → review → resume execution (memory kept)".center(78))
    print("=" * 78)

    agent = create_rag_like_agent(
        planning_interval=4,
        max_steps=15,
        verbosity_level=1,
        allow_plan_edit=True,
    )

    task = (
        "Compare the parameter count, context length and release date "
        "of DeepSeek-V3, Qwen-2.5-Max, and Llama-3.1-405B"
    )

    print(f"\nTask: {task}\n")

    print("→ First partial run (will likely interrupt)...")
    try:
        agent.run(task, reset=True)
    except KeyboardInterrupt:
        print("\n→ Interrupted. You can now resume.")

    if input("\nResume execution? (y/n): ").strip().lower() == "y":
        print("\n→ Resuming with preserved memory...")
        try:
            result = agent.run(task, reset=False)
            print("\nFinal result:", result)
        except KeyboardInterrupt:
            print("\nResumed execution also cancelled by user.")


def demo_hitl_4_inspect_memory():
    """Demo 4: After any HITL session – inspect what is in memory"""
    print("\n" + "=" * 78)
    print("Demo 4: Inspect agent memory after HITL interaction".center(78))
    print("=" * 78)

    # Use the last agent from previous demo if you ran it,
    # otherwise create a new one and run something short
    agent = create_rag_like_agent(
        planning_interval=5,
        max_steps=8,
        verbosity_level=0,  # quiet
    )

    task = "What is the current status of Grok-3 development?"

    print(f"\nRunning short task to populate memory: {task}")
    try:
        agent.run(task, reset=True)
    except KeyboardInterrupt:
        print("→ Interrupted (expected)")

    # Inspect memory
    print("\nMemory inspection:")
    print(f"Total steps in memory: {len(agent.memory.steps)}")
    print("Steps breakdown:")
    from collections import Counter

    types = Counter(type(step).__name__ for step in agent.memory.steps)
    for t, count in types.most_common():
        print(f"  {t:18} : {count:2d}")

    if agent.memory.steps:
        print("\nLast 3 steps:")
        for step in agent.memory.steps[-3:]:
            step_type = type(step).__name__
            if isinstance(step, PlanningStep):
                print(f"  {step_type} – plan length: {len(step.plan)}")
            elif isinstance(step, ActionStep):
                if step.code_action:
                    action_preview = str(step.code_action)[:70]
                    print(f"  {step_type} – code: {action_preview}...")
                elif step.model_output:
                    output_preview = str(step.model_output)[:70]
                    print(f"  {step_type} – output: {output_preview}...")
                else:
                    print(f"  {step_type}")
            elif isinstance(step, TaskStep):
                task_preview = step.task[:70]
                print(f"  {step_type} – task: {task_preview}...")
            elif isinstance(step, SystemPromptStep):
                prompt_preview = step.system_prompt[:70]
                print(f"  {step_type} – prompt: {prompt_preview}...")
            elif isinstance(step, FinalAnswerStep):
                output_preview = str(step.output)[:70]
                print(f"  {step_type} – output: {output_preview}...")
            else:
                print(f"  {step_type}")


def main():
    print("=" * 90)
    print("  Human-in-the-Loop Demos with LOCAL llama.cpp  ".center(90))
    print("  Interrupt, review, edit plan, resume, inspect memory  ".center(90))
    print("=" * 90 + "\n")

    # Uncomment what you want to try
    # demo_hitl_1_basic_interrupt()
    # demo_hitl_2_modify_plan()
    # demo_hitl_3_resume_after_interrupt()
    demo_hitl_4_inspect_memory()

    print("\n" + "=" * 90)
    print("Done".center(90))
    print("=" * 90)


if __name__ == "__main__":
    main()
