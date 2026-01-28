# demo_memory_usage_local_llama.py
"""
Demonstration examples of working with smolagents memory
using a LOCAL llama.cpp OpenAI-compatible server
"""

import time
from smolagents import OpenAIModel, CodeAgent, ActionStep, TaskStep
from typing import Optional

from smolagents.memory import Timing


def create_local_model(
    temperature: float = 0.7,
    max_tokens: int | None = None,
    model_id: str = "local-model",
) -> OpenAIModel:
    """Factory for creating consistently configured local llama.cpp model."""
    return OpenAIModel(
        model_id=model_id,
        api_base="http://shawn-pc.local:8080/v1",
        api_key="not-needed",           # llama.cpp server ignores this
        temperature=temperature,
        max_tokens=max_tokens,
    )


def demo_1_simple_run_and_replay():
    """Demo 1: Basic run + replay last execution (local llama.cpp)"""
    print("\n" + "="*70)
    print("Demo 1: Simple run + agent.replay()  [LOCAL LLAMA.CPP]")
    print("="*70)

    model = create_local_model(temperature=0.7)

    agent = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=0
    )

    result = agent.run("What's the 20th Fibonacci number?")
    print(f"\nResult: {result}")

    print("\nReplaying last run:")
    agent.replay(detailed=True)


def demo_2_inspect_memory_after_run():
    """Demo 2: Looking into different parts of memory after run"""
    print("\n" + "="*70)
    print("Demo 2: Inspecting memory contents  [LOCAL LLAMA.CPP]")
    print("="*70)

    model = create_local_model(temperature=0.7)

    agent = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=0
    )

    agent.run("Calculate 8 factorial")

    system_prompt_step = agent.memory.system_prompt
    print("\nSystem prompt was:")
    prompt_preview = system_prompt_step.system_prompt[:400] + "..." \
        if len(system_prompt_step.system_prompt) > 400 else system_prompt_step.system_prompt
    print(prompt_preview)

    if agent.memory.steps:
        first_task = agent.memory.steps[0]
        print("\nFirst task was:")
        print(first_task.task)

    print("\nSteps overview:")
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            if step.error:
                print(f"Step {step.step_number:2d} → ERROR: {step.error.strip()}")
            else:
                obs = str(step.observations).strip()
                obs_preview = (obs[:100] + "...") if len(obs) > 100 else obs
                print(f"Step {step.step_number:2d} → {obs_preview}")


def demo_3_run_one_step_at_a_time():
    """Demo 3: Manual step-by-step execution with memory control"""
    print("\n" + "="*70)
    print("Demo 3: Step-by-step manual execution  [LOCAL LLAMA.CPP]")
    print("="*70)

    model = create_local_model(temperature=0.7)

    agent = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=1
    )

    # Optional: prepare tools in executor
    agent.python_executor.send_tools({**agent.tools})

    task = "What is the 8th Fibonacci number?"

    # Start new task
    agent.memory.steps.append(TaskStep(task=task, task_images=[]))

    final_answer: Optional[str] = None
    max_steps = 8
    step_number = 1

    print(f"\nStarting task: {task}\n")

    while final_answer is None and step_number <= max_steps:
        print(f"\n→ Step {step_number}")

        step_start_time = time.time()

        current_step = ActionStep(
            step_number=step_number,
            timing=Timing(start_time=step_start_time),
            observations_images=[],
        )

        # Execute one step
        final_answer = agent.step(current_step)

        # Append to memory
        agent.memory.steps.append(current_step)

        step_number += 1

        if final_answer:
            print(f"→ Final answer found: {final_answer}")
        else:
            print("→ Continuing...")

    if final_answer is None:
        print("\nReached max steps without final answer.")


def demo_4_get_full_steps_as_dicts():
    """Demo 4: Getting memory steps as plain dictionaries"""
    print("\n" + "="*70)
    print("Demo 4: agent.memory.get_full_steps()  [LOCAL LLAMA.CPP]")
    print("="*70)

    model = create_local_model(temperature=0.7)

    agent = CodeAgent(
        tools=[],
        model=model,
        verbosity_level=0
    )

    agent.run("2 ** 10 + 3 ** 5")

    print("\nAll steps as dictionaries:")
    full_steps = agent.memory.get_full_steps()

    for i, step_dict in enumerate(full_steps, 1):
        print(f"\nStep {i}:")
        print(f"  type: {step_dict.get('type')}")
        if 'task' in step_dict:
            print(f"  task: {step_dict['task']}")
        if 'observations' in step_dict:
            obs = str(step_dict['observations'])
            print(f"  observations: {obs[:120]}{'...' if len(obs) > 120 else ''}")
        if 'error' in step_dict and step_dict['error']:
            print(f"  ERROR: {step_dict['error']}")


def main():
    print("=" * 78)
    print("  smolagents memory examples  —  LOCAL llama.cpp server  ".center(78))
    print("  Endpoint: http://shawn-pc.local:8080/v1  ".center(78))
    print("=" * 78 + "\n")

    # Select which demos to run
    demo_1_simple_run_and_replay()
    demo_2_inspect_memory_after_run()
    demo_3_run_one_step_at_a_time()
    demo_4_get_full_steps_as_dicts()

    print("\n" + "="*78)
    print("Done".center(78))
    print("="*78)


if __name__ == "__main__":
    main()